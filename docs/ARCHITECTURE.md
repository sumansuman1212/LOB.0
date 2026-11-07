# Technical Architecture & Systems Design

This document details the internal systems architecture, data structures, and mathematical formulations implemented across the **LOB.0** engine.

---

## 1. Systems Architecture

LOB.0 separates **asynchronous learning** from **synchronous execution**:

```
┌────────────────────────────────────────────────────────┐
│             Python Reinforcement Learning              │
│       PPO Actor-Critic (MLP: 8 -> 256 -> 256 -> 3)     │
│       GAE-λ Advantage Estimation (TrajectoryBuffer)    │
│       RewardShaper (MtM + Inventory Penalty + Spread)  │
└───────────────────────────┬────────────────────────────┘
                            │ Serialized Graph (.pt + SHA-256)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   C++17 Execution Core                 │
│                                                        │
│  L2 Ticks ──► TickReplayer (SPSC Lock-Free Ring)       │
│                     │                                  │
│                     ▼                                  │
│            StrategyFramework                           │
│              ├── LobEngine (64B Cacheline PriceLevel)  │
│              ├── FeaturePipeline (8D, Zero-Alloc)      │
│              ├── PolicyInference (Atomic Hot-Reload)   │
│              └── QuoteGenerator (Halt Rule & Guard)    │
│                     │                                  │
│                     ▼                                  │
│                OrderGateway                            │
└────────────────────────────────────────────────────────┘
```

---

## 2. Limit Order Book (`LobEngine`)

### 2.1 Memory Layout & Cache Considerations
- **`PriceLevel` Alignment:** Cacheline aligned to 64 bytes (`alignas(64)`) with explicit byte padding to eliminate cacheline boundary splitting:
  ```cpp
  struct alignas(64) PriceLevel {
      int64_t  price;        // 8 bytes
      uint64_t total_qty;    // 8 bytes
      uint32_t order_count;  // 4 bytes
      uint32_t _pad0;        // 4 bytes
      uint8_t  _pad1[40];    // 40 bytes padding -> 64 bytes total
  };
  ```
- **Allocation Reality:** In `LobEngine`, `bids_` and `asks_` use `std::map`, and orders use `std::list`. Both standard library containers allocate dynamic node objects on the heap for every insertion. While `alignas(64)` ensures individual `PriceLevel` structs do not straddle a 64-byte cacheline boundary, tree traversals follow pointer nodes across disparate heap addresses. True contiguous cacheline traversal is achieved in `BookSnapshot::bids` and `BookSnapshot::asks` (`std::array<PriceLevel, 10>`), which reside contiguously on the stack or in flat structs. Furthermore, `alignas(64)` alone does not eliminate multi-threaded false sharing unless memory structures are explicitly segregated and allocated with separate cacheline padding per concurrent core.
- **Order Tracking & Removal Complexity:** `orders_` (`std::unordered_map`) allows $O(1)$ average iterator retrieval. Removing an order from `std::list` is $O(1)$. However, if that order was the final resting order at that price level, removing the now-empty level from `std::map` incurs an $O(\log M)$ red-black tree rebalancing deletion.
- **`OrderEvent` Packing:** Packed to exactly 32 bytes (`sizeof(OrderEvent) == 32`), allowing two complete order events to occupy a single 64-byte L1 cache line.

### 2.2 Data Structures
- **Bids:** `std::map<int64_t, PriceLevel, std::greater<int64_t>>` (descending sorted tree). Best bid is accessible in $O(1)$ at `bids_.begin()`.
- **Asks:** `std::map<int64_t, PriceLevel, std::less<int64_t>>` (ascending sorted tree). Best ask is accessible in $O(1)$ at `asks_.begin()`.
- **Order Index:** `std::unordered_map<uint64_t, OrderEntry>` provides $O(1)$ average order lookups for cancels, modifications, and partial/full executions.

### 2.3 Transaction Safety Invariants
1. **Timestamp Monotonicity:** Events with `timestamp_ns < last_timestamp_ns_` are rejected.
2. **Positive Pricing:** Events with `price <= 0` are rejected (except Cancels where price is unused).
3. **No Crossed Books:** Any Add or Modify event that would cause `bid_price >= best_ask` or `ask_price <= best_bid` is rejected with state left untouched.
4. **Duplicate Rejection:** Duplicate `order_id` on Add events is rejected to prevent orphaning aggregate price level quantities.

---

## 3. Microstructure Feature Vector

The `FeaturePipeline` calculates an 8-dimensional observation vector with zero dynamic heap allocations on the hot path:

| Index | Name | Formula / Definition | Range |
| :---: | :--- | :--- | :---: |
| 0 | `ofi_l1` | $\frac{Q_{\text{bid},0} - Q_{\text{ask},0}}{Q_{\text{bid},0} + Q_{\text{ask},0}}$ | $[-1, 1]$ |
| 1 | `ofi_l5` | $\frac{\sum_{i=0}^4 Q_{\text{bid},i} - \sum_{i=0}^4 Q_{\text{ask},i}}{\sum_{i=0}^4 Q_{\text{bid},i} + \sum_{i=0}^4 Q_{\text{ask},i}}$ | $[-1, 1]$ |
| 2 | `inventory_skew` | $\frac{q}{q_{\max}}$ | $[-1, 1]$ |
| 3 | `inventory_urgency` | $\frac{\|q\|}{q_{\max}}$ | $[0, 1]$ |
| 4 | `vol_regime` | $\frac{\text{RMS}_{\text{window}} - \mu_{\sigma}}{\text{std}_{\sigma}}$ (circular-buffer sliding RMS) | $[-5, 5]$ |
| 5 | `micro_price_offset` | $\frac{P_{\text{micro}} - P_{\text{mid}}}{\text{half\_spread}}$ | $[-5, 5]$ |
| 6 | `spread_ticks` | $\log(1 + (P_{\text{ask}} - P_{\text{bid}} - 1))$ | $[0, 5]$ |
| 7 | `session_progress` | $\frac{t - t_0}{T_{\text{session}}}$ | $[0, 1]$ |

*Note on Rolling Volatility Accumulator:* `FeaturePipeline` uses a circular buffer of log mid-price returns with rolling sum ($\sum r$) and sum-of-squares ($\sum r^2$) accumulators in double precision. Realized volatility is calculated as the root-mean-square (RMS) return $\sqrt{\frac{1}{W}\sum r_i^2}$. It is not a standard one-pass Welford recurrence, but a sliding-window subtract-and-add accumulator designed for $O(1)$ fixed-window updates.

---

## 4. Quote Generation & Guardrails

### 4.1 Continuous Action Space to Discrete Ticks
The PPO actor policy network emits a 3-dimensional continuous action $\mathbf{a} = (\Delta_{\text{bid}}, \Delta_{\text{ask}}, \text{size\_frac})$:
- $\Delta_{\text{bid}} \in [0.0, 20.0]$: Desired bid distance from mid-price in ticks.
- $\Delta_{\text{ask}} \in [0.0, 20.0]$: Desired ask distance from mid-price in ticks.
- $\text{size\_frac} \in [0.01, 1.0]$: Desired quoting size as a fraction of `max_order_size`.

These are converted into exchange-valid integer prices:
$$P_{\text{bid}} = P_{\text{mid}} - \lfloor \Delta_{\text{bid}} \cdot \text{tick\_size} \rceil, \quad P_{\text{ask}} = P_{\text{mid}} + \lfloor \Delta_{\text{ask}} \cdot \text{tick\_size} \rceil$$

### 4.2 Minimum Spread Invariant
If discretized prices collapse such that $P_{\text{ask}} - P_{\text{bid}} < \text{tick\_size}$, the ask is forced out:
$$P_{\text{ask}} = P_{\text{bid}} + \text{tick\_size}$$

### 4.3 Inventory Halt Rule (90% Soft Limit) & Microstructure Discussion
When inventory breaches $90\%$ of `max_inventory`:
- **Accumulation Halt:** Quoting quantity on the accumulating side is forced to zero ($Q_{\text{bid}} = 0$ if long, $Q_{\text{ask}} = 0$ if short), strictly preventing additional position accumulation.
- **Liquidation Quoting Behavior:** The opposite quote distance from mid is widened ($2\times \text{dist}$), preserving alpha signal adjustments.
  > *Microstructure Note:* In pure execution theory, increasing quote distance on the liquidation side moves quotes further away from the inside market, reducing fill probability. While it prevents trading at unfavorable adverse-selection prices, rapid inventory de-risking in live markets typically benefits from *tightening* the liquidation quote closer to mid (e.g. 1 tick) to attract fills.

---

## 5. Circuit Breaker & Fallback Policy (Inventory-Aware Quoting Heuristic)

If TorchScript neural inference throws a runtime exception, encounters NaN feature values, or breaches the latency budget ($800\text{ ns}$ for $>1\%$ of quotes), the engine routes inference to `SimplifiedPolicyInference`.

> **Theoretical Classification:** This policy implements an **inventory-aware quoting heuristic inspired by Avellaneda-Stoikov**. It is **not** the classical analytical Avellaneda-Stoikov model ($r(s,q) = s - q\gamma\sigma^2(T-t)$) because it omits explicit terminal horizon discounting $(T-t)$ and models volatility and OFI as linear additive offsets rather than continuous multiplicative diffusion variance.

$$\Delta_{\text{bid}} = 1.0 + 1.5 \max(0, \text{skew}) - 0.5 \max(0, -\text{skew}) - 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$
$$\Delta_{\text{ask}} = 1.0 + 1.5 \max(0, -\text{skew}) - 0.5 \max(0, \text{skew}) + 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$

- When `inventory_skew > 0` (long), $\Delta_{\text{bid}}$ widens and $\Delta_{\text{ask}}$ tightens.
- When `inventory_skew < 0` (short), $\Delta_{\text{ask}}$ widens and $\Delta_{\text{bid}}$ tightens.
- When `inventory_skew == 0`, quotes are symmetric ($\Delta_{\text{bid}} = \Delta_{\text{ask}} = 1.0$) **if and only if** order flow imbalance is neutral ($\text{OFI} = 0$) and volatility is at or below baseline ($\text{vol} \le 0$). Non-zero order flow imbalance will skew quotes even at zero inventory to account for directional order flow pressure.

---

## 6. Algorithmic Nuances & Optimization Bounds

1. **PPO Clipped Objective:**
   $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
   Clipping prevents catastrophically large policy divergences. However, unlike trust-region methods with exact KL constraints (TRPO), PPO clipping does **not** guarantee monotonic policy improvement; policy collapse remains possible under high learning rates or non-stationary reward distributions.

2. **Generalized Advantage Estimation (GAE):**
   $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
   When $\lambda = 1$, GAE approaches empirical Monte Carlo returns only when the rollout reaches an explicit terminal state or uses an unbiased value function bootstrap for infinite-horizon continuations.

3. **Reward Scaling vs. Sharpe Ratio:**
   Dividing step reward by running realized return standard deviation normalizes gradients across quiet and volatile regimes. It is an online variance-stabilization heuristic, not an annualized financial Sharpe ratio.

4. **Soft Risk Limit (90% Inventory Halt):**
   When $|q| \ge 0.9 \cdot q_{\max}$, quoting quantity on the accumulating side is set to 0. This halts additional quote generation on that side. It is a **soft risk limit** rather than a hard guarantee, because:
   - In-flight orders already acknowledged by the exchange can still execute.
   - Large incoming fills can breach the threshold before the next quote cycle updates.

---

## 7. Financial Accounting vs. Reinforcement Learning Reward

To ensure quantitative rigor, LOB.0 maintains strict mathematical and conceptual separation between **financial accounting ledgers** and **reinforcement learning surrogate rewards**:

### 7.1 Financial Accounting Identities (Ledger)
Every executed trade and price update mutates the fundamental financial state:
1. **Cash Ledger:**
   $$\text{Cash}_{t+1} = \text{Cash}_t - P_{\text{fill}} \cdot \Delta q - \text{Fee}$$
   (Decreases on buy fills; increases on sell fills).
2. **Inventory Balance:**
   $$\text{Inventory}_{t+1} = \text{Inventory}_t + \Delta q$$
3. **Portfolio Equity:**
   $$\text{Equity}_t = \text{Cash}_t + \text{Inventory}_t \cdot P_{\text{mid}, t}$$
4. **Total Financial PnL:**
   $$\text{PnL}_t = \text{Equity}_t - \text{InitialEquity} = \text{Realized PnL}_t + \text{Unrealized PnL}_t$$
   - **Realized PnL:** Closed-trade cash profit: $\sum (\text{ExitPrice} - \text{EntryPrice}) \cdot \text{ClosedQty}$.
   - **Unrealized PnL:** Mark-to-market floating gain on resting inventory: $\text{Inventory}_t \cdot (P_{\text{mid}, t} - \text{AvgCost})$.
5. **Transaction Costs & Fees:**
   Currently, exchange transaction fees and maker rebates are set to zero ($\text{fee\_rate} = 0.0$). If fees are active, $\text{Cash}$ is debited by $\text{Fee} = P_{\text{fill}} \cdot |\Delta q| \cdot \text{fee\_rate}$.

### 7.2 Reinforcement Learning Reward Signal (Shaping Heuristic)
The RL reward $R_t$ fed into PPO is an artificial optimization signal:
$$R_t = \Delta \text{MtM}_t - \gamma \left(\frac{|\text{Inventory}_t|}{q_{\max}}\right) + \text{SpreadCapture}_t$$
- $\Delta \text{MtM}_t = (P_{\text{mid}, t} - P_{\text{mid}, t-1}) \cdot \text{Inventory}_{t-1}$: One-step mark-to-market drift.
- $-\gamma \frac{|q|}{q_{\max}}$: Artificial holding penalty to incentivize inventory neutrality. **This is not cash flow.**
- $\text{SpreadCapture}_t = \min(|q_{t-1}|, \text{fill\_qty}) \cdot \text{half\_spread}$: Proxy credit awarded upon closing inventory.

Both quantities are independently tracked in `LobEnv.step()` and surfaced via the `info` dictionary (`info["accounting_pnl"]` vs `info["step_reward"]`).
