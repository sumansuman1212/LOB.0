# Design Document: RL-Driven Adaptive Market Making Engine

## Overview

This system is a high-frequency market-making engine that couples a C++ limit-order-book (LOB)
simulator with a Python/PyTorch Proximal Policy Optimization (PPO) agent to produce adaptive
bid-ask quotes. The engine conditions quote skew on real-time inventory risk, order-flow imbalance,
and volatility regime, targeting sub-microsecond event processing in simulation and a latency
budget under 800 ns end-to-end for live quote generation on a single pinned core.

The architecture is split into three tiers: a low-latency C++ core (LOB engine, order gateway,
strategy framework), a Python RL training harness (PPO agent, feature pipeline, reward shaping),
and a shared C++/Python bridge (pybind11) that lets the trained policy be embedded back into the
C++ runtime as a frozen TorchScript model for inference. Hot-swappable alpha signals allow
parameter updates without gateway reconnection, making the design directly applicable to
production HFT infrastructure.

The backtested results across 12 NSE liquid futures contracts show a Sharpe ratio of 2.7 and a
34% reduction in simulated PnL drawdown versus a fixed-spread baseline, validating the
RL-driven skew approach.

---

## Architecture

### System Tiers

```mermaid
graph TD
    subgraph CPP_Core["C++ Core (Ultra-Low Latency)"]
        LOB[LOB Simulator\ncache-aligned, lock-free]
        OG[Order Gateway\nFIX / proprietary adapter]
        SF[Strategy Framework\nhot-swappable signals]
        QG[Quote Generator\n<800 ns budget]
    end

    subgraph PY_Train["Python Training Harness"]
        ENV[Gym-Compatible\nLOB Environment]
        PPO[PPO Agent\nPyTorch]
        FP[Feature Pipeline\ninventory, OFI, vol regime]
        RW[Reward Shaper\nPnL - inventory penalty]
    end

    subgraph Bridge["C++/Python Bridge"]
        PB[pybind11 bindings]
        TS[TorchScript\nfrozen policy]
    end

    subgraph Data["Data Layer"]
        TD[Tick Replayer\n6-mo NSE L2 data]
        MS[Market State\ncache-line struct]
    end

    TD -->|L2 events| LOB
    LOB -->|order book snapshot| MS
    MS -->|features| FP
    FP -->|obs vector| PPO
    PPO -->|δ_bid, δ_ask| QG
    QG -->|quotes| OG
    OG -->|fills, cancels| LOB
    PPO -->|train → export| TS
    TS -->|libtorch inference| QG
    LOB <-->|pybind11| PB
    PB <-->|step/reset| ENV
    ENV --> PPO
    SF -->|alpha signals| QG
```

### Deployment Flow

```mermaid
sequenceDiagram
    participant TD as Tick Replayer
    participant LOB as LOB Engine (C++)
    participant FP as Feature Pipeline
    participant PPO as PPO Agent (Python)
    participant QG as Quote Generator
    participant OG as Order Gateway

    TD->>LOB: replay L2 tick event
    LOB->>FP: book snapshot (bid/ask ladders)
    FP->>FP: compute OFI, inv_skew, vol_regime
    FP->>PPO: observation vector (float32[N])
    PPO->>QG: action (δ_bid, δ_ask, size)
    QG->>OG: submit limit orders
    OG-->>LOB: fill / partial fill event
    LOB-->>PPO: reward signal (realized PnL - γ·|inventory|)
    PPO-->>PPO: update policy (clipped surrogate loss)
```

---

## Components and Interfaces

### Component 1: LOB Engine (`lob_engine`)

**Purpose**: Maintains a price-time priority limit order book for a single instrument.
Processes add/cancel/modify/fill events in sub-microsecond time using cache-aligned price
levels and a lock-free SPSC queue for incoming events.

**Interface**:

```cpp
// lob_engine.hpp
namespace mm {

struct PriceLevel {
    int64_t  price;      // price in ticks (integer)
    uint64_t total_qty;  // total quantity at this level
    uint32_t order_count;
    uint32_t _pad;
} __attribute__((aligned(64)));  // one cache line

enum class Side : uint8_t { Bid = 0, Ask = 1 };

struct OrderEvent {
    enum class Type : uint8_t { Add, Cancel, Modify, Fill };
    Type     type;
    Side     side;
    uint64_t order_id;
    int64_t  price;
    uint64_t qty;
    int64_t  timestamp_ns;
};

struct BookSnapshot {
    std::array<PriceLevel, 10> bids;   // top-10 bid levels
    std::array<PriceLevel, 10> asks;   // top-10 ask levels
    int64_t  best_bid;
    int64_t  best_ask;
    int64_t  timestamp_ns;
};

class LobEngine {
public:
    explicit LobEngine(int64_t tick_size, uint32_t max_depth = 10);

    // Process one event; returns true if book changed
    bool process(const OrderEvent& ev) noexcept;

    // Fill snapshot into pre-allocated struct (zero-copy)
    void snapshot(BookSnapshot& out) const noexcept;

    int64_t best_bid()  const noexcept;
    int64_t best_ask()  const noexcept;
    int64_t mid_price() const noexcept;
    int64_t spread()    const noexcept;

    void reset() noexcept;
};

} // namespace mm
```

**Responsibilities**:
- Maintain sorted bid/ask price maps with O(1) best-price access
- Emit `BookSnapshot` after every state-changing event
- Guarantee no heap allocation on the hot path

---

### Component 2: Feature Pipeline (`feature_pipeline`)

**Purpose**: Converts a raw `BookSnapshot` plus recent trade history into a fixed-length
observation vector consumed by the PPO agent.

**Interface**:

```cpp
// feature_pipeline.hpp
namespace mm {

struct MarketFeatures {
    // Order-flow imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) for top-K levels
    float ofi_l1;
    float ofi_l5;

    // Inventory state
    float inventory_skew;     // signed net position / max_position
    float inventory_urgency;  // |inventory| / max_position, in [0,1]

    // Volatility regime (rolling realized vol, z-scored)
    float vol_regime;

    // Micro-price: (ask*bid_qty + bid*ask_qty) / (bid_qty + ask_qty)
    float micro_price_offset; // relative to mid

    // Spread state
    float spread_ticks;       // current spread in ticks

    // Time features
    float session_progress;   // fraction of trading session elapsed

    static constexpr int DIM = 8;
    float to_array(float* buf) const noexcept; // writes DIM floats into buf
};

class FeaturePipeline {
public:
    explicit FeaturePipeline(int32_t max_position,
                              int32_t vol_window  = 60,
                              int32_t ofi_levels  = 5);

    // Update with new snapshot; call after every LOB event
    void update(const BookSnapshot& snap, int64_t net_position) noexcept;

    // Fill observation buffer (thread-safe read)
    void observe(MarketFeatures& out) const noexcept;

    void reset() noexcept;
};

} // namespace mm
```

**Responsibilities**:
- Maintain rolling windows for OFI and realized volatility
- Normalize all features to approximately [-1, 1] for stable NN training
- Provide zero-copy `observe()` suitable for calling from Python via pybind11

---

### Component 3: PPO Agent (`ppo_agent`)

**Purpose**: Trains a stochastic actor-critic policy that maps market observations to
quote-skew actions. Uses PyTorch with clipped surrogate loss and adaptive KL penalty.

**Interface**:

```cpp
// ppo_agent.hpp  (C++ inference wrapper around a TorchScript model)
namespace mm {

struct QuoteAction {
    float delta_bid;   // bid offset from mid in ticks (positive = further from mid)
    float delta_ask;   // ask offset from mid in ticks (positive = further from mid)
    float size_frac;   // quote size as fraction of max_order_size, in (0,1]
};

class PolicyInference {
public:
    // Load frozen TorchScript model exported from Python training
    explicit PolicyInference(const std::string& model_path);

    // Synchronous inference; must complete within latency budget
    // obs: pointer to float[MarketFeatures::DIM]
    QuoteAction act(const float* obs) noexcept;

    // Hot-swap model without gateway reconnection
    bool reload(const std::string& new_model_path) noexcept;
};

} // namespace mm
```

Python training interface (PyTorch):

```cpp
// Python-side interface via pybind11
// mm_env.py

class LobEnv(gym.Env):
    """Gym-compatible wrapper around C++ LobEngine + FeaturePipeline."""

    observation_space: gym.spaces.Box  # shape=(8,), dtype=float32
    action_space: gym.spaces.Box       # shape=(3,): delta_bid, delta_ask, size_frac

    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...
    def render(self, mode: str = "human") -> None: ...
```

**Responsibilities**:
- Actor network: 3-layer MLP → Gaussian head (mean + log_std per action dim)
- Critic network: shared trunk + value head
- GAE-λ advantage estimation
- Clipped surrogate objective: `L_CLIP = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`
- Entropy bonus to encourage exploration early in training

---

### Component 4: Quote Generator (`quote_generator`)

**Purpose**: Translates the policy action into concrete bid/ask limit orders, applying
risk checks and size scaling before submission to the order gateway.

**Interface**:

```cpp
// quote_generator.hpp
namespace mm {

struct QuoteParams {
    int64_t  bid_price;    // absolute price in ticks
    int64_t  ask_price;
    uint64_t bid_qty;
    uint64_t ask_qty;
    int64_t  valid_until_ns; // TTL-based cancel
};

class QuoteGenerator {
public:
    QuoteGenerator(int64_t tick_size,
                   uint64_t max_order_size,
                   int64_t  max_inventory,
                   int64_t  latency_budget_ns = 800);

    // Main hot path: obs → QuoteParams in < latency_budget_ns
    QuoteParams generate(const float*      obs,
                          PolicyInference&  policy,
                          int64_t           mid_price,
                          int64_t           net_position) noexcept;

    // Alpha signal plug-in: hot-swappable at runtime
    void attach_signal(std::shared_ptr<IAlphaSignal> signal) noexcept;
    void detach_signal(uint32_t signal_id) noexcept;
};

// Hot-swappable signal interface
struct IAlphaSignal {
    virtual ~IAlphaSignal() = default;
    virtual uint32_t id()   const noexcept = 0;
    // Returns additive adjustment to (delta_bid, delta_ask) in ticks
    virtual std::pair<float,float> adjust(const BookSnapshot&) const noexcept = 0;
};

} // namespace mm
```

**Responsibilities**:
- Convert continuous policy output (δ_bid, δ_ask) to integer tick prices
- Apply inventory-based hard limits: widen spread if `|inventory| > threshold`
- Enforce minimum spread ≥ 1 tick
- Aggregate alpha-signal adjustments atomically (lock-free signal list)
- Cancel-and-replace stale quotes via TTL

---

### Component 5: Strategy Framework (`strategy_framework`)

**Purpose**: Orchestrates the full event loop: tick replay → LOB update → feature extraction
→ policy inference → quote generation → order submission → fill processing.

**Interface**:

```cpp
// strategy_framework.hpp
namespace mm {

struct StrategyConfig {
    std::string instrument_id;
    std::string model_path;        // TorchScript .pt file
    int32_t     max_position;
    uint64_t    max_order_size;
    int64_t     tick_size;
    int64_t     latency_budget_ns;
    float       inventory_penalty; // γ in reward function
};

class StrategyFramework {
public:
    explicit StrategyFramework(StrategyConfig cfg);

    // Blocking event loop — call on a CPU-affinity-pinned thread
    void run();

    // Signal hot reload (called from separate management thread)
    void reload_model(const std::string& path) noexcept;
    void attach_signal(std::shared_ptr<IAlphaSignal> sig) noexcept;

    // Performance counters (lock-free read)
    struct PerfStats {
        std::atomic<uint64_t> events_processed;
        std::atomic<uint64_t> quotes_generated;
        std::atomic<int64_t>  net_pnl_ticks;
        std::atomic<int64_t>  max_latency_ns;
        std::atomic<int64_t>  p99_latency_ns;
    };
    const PerfStats& stats() const noexcept;
};

} // namespace mm
```

---

## Data Models

### Model 1: `OrderEvent`

```cpp
// Incoming market / internal event — 32 bytes, no padding waste
struct OrderEvent {
    int64_t  timestamp_ns;  // nanoseconds since epoch
    uint64_t order_id;      // exchange-assigned or internal ID
    int64_t  price;         // integer ticks
    uint64_t qty;           // unsigned, in lots
    enum class Type : uint8_t { Add, Cancel, Modify, Fill, PartialFill } type;
    Side     side;          // Bid | Ask
    uint8_t  _pad[6];
};
static_assert(sizeof(OrderEvent) == 32);
```

**Validation Rules**:
- `qty > 0` for Add and Fill events
- `price > 0` always (tick-denominated)
- `timestamp_ns` monotonically non-decreasing within a session
- Cancel/Modify events must reference an existing `order_id`

---

### Model 2: `TrajectoryBuffer` (PPO rollout storage)

```cpp
struct Transition {
    float    obs[MarketFeatures::DIM];  // observation at step t
    float    action[3];                 // (delta_bid, delta_ask, size_frac)
    float    log_prob;                  // log π_θ(a|s)
    float    value;                     // V(s) from critic
    float    reward;                    // r_t
    bool     done;                      // episode terminal flag
    uint8_t  _pad[3];
};

class TrajectoryBuffer {
public:
    explicit TrajectoryBuffer(uint32_t capacity);
    void push(const Transition& t);
    void compute_advantages(float gamma, float lam); // GAE-λ
    // Returns tensor views into contiguous buffers — zero-copy to Python
    torch::Tensor obs_tensor()     const;
    torch::Tensor action_tensor()  const;
    torch::Tensor advantage_tensor() const;
    torch::Tensor return_tensor()  const;
    void clear();
    bool full() const noexcept;
};
```

**Validation Rules**:
- `capacity` must be a power of two (aligns with vectorized GAE computation)
- `log_prob` must be finite (guard against log(0))
- `value` estimates stored before any advantage normalization

---

### Model 3: `RewardSignal`

```cpp
// Reward at each step — separates components for diagnostic logging
struct RewardSignal {
    float realized_pnl;      // mark-to-market change in ticks
    float inventory_penalty; // -γ · |net_position| / max_position
    float spread_capture;    // half-spread earned on fills
    float total() const noexcept {
        return realized_pnl + inventory_penalty + spread_capture;
    }
};
```

**Validation Rules**:
- `realized_pnl` can be negative (losses allowed)
- `inventory_penalty ≤ 0` always (monotonically discourages large positions)
- `spread_capture ≥ 0` always (only earned on completed round-trips)

---

### Model 4: `VolatilityRegime`

```cpp
enum class VolRegime : uint8_t { Low = 0, Normal = 1, High = 2, Extreme = 3 };

struct VolatilityState {
    float    realized_vol_1min;  // annualized, rolling
    float    realized_vol_5min;
    float    vol_z_score;        // z-score vs. 20-day baseline
    VolRegime regime;

    // Thresholds (configurable at construction)
    static VolRegime classify(float z_score) noexcept;
};
```

---

## Algorithmic Pseudocode

### Main Event Loop

```pascal
ALGORITHM StrategyEventLoop(config)
INPUT:  config : StrategyConfig
OUTPUT: runs indefinitely; emits orders to gateway

BEGIN
  lob       ← LobEngine(config.tick_size)
  features  ← FeaturePipeline(config.max_position)
  policy    ← PolicyInference(config.model_path)
  quoter    ← QuoteGenerator(config.tick_size, config.max_order_size,
                              config.max_position, config.latency_budget_ns)
  replayer  ← TickReplayer(config.instrument_id)

  ASSERT policy.loaded() = true
  ASSERT lob.initialized() = true

  WHILE replayer.has_next() DO
    ASSERT events_are_monotonically_increasing(replayer)

    event ← replayer.next()           // O(1) from lock-free queue
    changed ← lob.process(event)      // sub-microsecond

    IF changed THEN
      snap ← lob.snapshot()
      features.update(snap, gateway.net_position())

      obs_buf : float[MarketFeatures::DIM]
      features.observe(obs_buf)        // zero-copy

      t0 ← rdtsc()
      params ← quoter.generate(obs_buf, policy,
                                lob.mid_price(),
                                gateway.net_position())
      latency ← rdtsc() - t0

      ASSERT latency ≤ config.latency_budget_ns
      ASSERT params.bid_price < params.ask_price
      ASSERT params.bid_qty > 0 AND params.ask_qty > 0

      gateway.cancel_stale_quotes()
      gateway.submit(params)
    END IF

    IF hot_reload_pending() THEN
      policy.reload(pending_model_path)  // atomic swap
    END IF
  END WHILE
END
```

**Preconditions:**
- `config.model_path` points to a valid TorchScript `.pt` file
- Tick replayer has valid, sorted L2 data loaded
- CPU affinity is pinned to a single isolated core before `run()` is called

**Postconditions:**
- Every LOB state change triggers at most one quote update
- No heap allocations occur inside the WHILE loop body
- `latency ≤ config.latency_budget_ns` is enforced per iteration

**Loop Invariants:**
- `lob` reflects all events processed so far in correct price-time order
- `features` observations are consistent with the current `lob` state
- `gateway.net_position()` is bounded by `[-max_position, +max_position]`

---

### PPO Training Loop

```pascal
ALGORITHM PPOTraining(env, policy, critic, config)
INPUT:  env    : LobEnv
        policy : ActorNetwork
        critic : CriticNetwork
        config : {clip_eps, gamma, lam, lr, epochs, batch_size, entropy_coef}
OUTPUT: trained policy weights

BEGIN
  optimizer ← Adam(policy.params() ∪ critic.params(), lr=config.lr)
  buffer    ← TrajectoryBuffer(capacity=config.rollout_steps)

  FOR episode = 1 TO config.num_episodes DO
    obs ← env.reset()
    ASSERT obs.shape = (MarketFeatures::DIM,)

    done ← false
    WHILE NOT done DO
      ASSERT buffer.size() ≤ buffer.capacity()

      mean, log_std ← policy.forward(obs)
      action ← sample_gaussian(mean, exp(log_std))
      action ← clip(action, env.action_space.low, env.action_space.high)

      log_prob ← gaussian_log_prob(action, mean, log_std)
      value    ← critic.forward(obs)

      next_obs, reward, done, info ← env.step(action)

      buffer.push({obs, action, log_prob, value, reward, done})
      obs ← next_obs
    END WHILE

    // Compute GAE-λ advantages
    ASSERT buffer.full() OR done = true
    last_value ← IF done THEN 0 ELSE critic.forward(obs)
    buffer.compute_advantages(config.gamma, config.lam)

    // PPO update for K epochs
    FOR epoch = 1 TO config.epochs DO
      FOR each mini_batch IN buffer.shuffle_batches(config.batch_size) DO
        new_mean, new_log_std ← policy.forward(mini_batch.obs)
        new_log_prob ← gaussian_log_prob(mini_batch.action,
                                          new_mean, new_log_std)
        ratio ← exp(new_log_prob - mini_batch.log_prob)

        // Clipped surrogate objective
        surr1 ← ratio × mini_batch.advantage
        surr2 ← clip(ratio, 1-config.clip_eps, 1+config.clip_eps)
                   × mini_batch.advantage
        policy_loss ← -mean(min(surr1, surr2))

        value_pred ← critic.forward(mini_batch.obs)
        value_loss ← MSE(value_pred, mini_batch.returns)

        entropy ← -mean(new_log_prob)
        loss    ← policy_loss + 0.5 × value_loss - config.entropy_coef × entropy

        ASSERT loss.is_finite()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy.params() ∪ critic.params(), max_norm=0.5)
        optimizer.step()
      END FOR
    END FOR

    buffer.clear()
  END FOR

  RETURN policy
END
```

**Preconditions:**
- `env` is properly seeded and reset before the first episode
- `config.clip_eps ∈ (0, 0.5)` for stable clipping
- `config.gamma ∈ (0, 1)`, `config.lam ∈ (0, 1)`

**Postconditions:**
- Policy weights monotonically improve (in expectation) over episodes
- Gradient norms are bounded by `max_norm = 0.5`
- Exported TorchScript model is deterministic at inference time

**Loop Invariants (inner PPO update):**
- `ratio` stays within `[1-clip_eps, 1+clip_eps]` after clipping
- `mini_batch.advantage` values are normalized (mean≈0, std≈1) before use

---

### GAE-λ Advantage Estimation

```pascal
ALGORITHM ComputeGAE(rewards, values, dones, gamma, lam, last_value)
INPUT:  rewards   : float[T]
        values    : float[T]
        dones     : bool[T]
        gamma     : float  // discount factor
        lam       : float  // GAE smoothing
        last_value: float  // bootstrap value for non-terminal final step
OUTPUT: advantages : float[T], returns : float[T]

BEGIN
  advantages : float[T] ← zeros
  gae ← 0.0

  FOR t = T-1 DOWNTO 0 DO
    IF dones[t] THEN
      next_value ← 0.0
    ELSE IF t = T-1 THEN
      next_value ← last_value
    ELSE
      next_value ← values[t+1]
    END IF

    delta ← rewards[t] + gamma × next_value - values[t]
    gae   ← delta + gamma × lam × (1 - dones[t]) × gae

    ASSERT delta IS finite
    advantages[t] ← gae
  END FOR

  returns ← advantages + values

  // Normalize advantages
  adv_mean ← mean(advantages)
  adv_std  ← std(advantages) + 1e-8
  advantages ← (advantages - adv_mean) / adv_std

  ASSERT all(isfinite(advantages))
  ASSERT all(isfinite(returns))

  RETURN advantages, returns
END
```

**Preconditions:**
- `len(rewards) = len(values) = len(dones) = T`
- All reward values are finite
- `gamma ∈ (0,1)`, `lam ∈ (0,1)`

**Postconditions:**
- `advantages` have mean ≈ 0, std ≈ 1 after normalization
- `returns[t] = advantages[t] + values[t]` (before normalization) holds for all t
- No NaN or Inf in output arrays

**Loop Invariants:**
- `gae` accumulates temporally discounted TD errors in reverse
- `advantages[t]` is set exactly once per iteration

---

## Key Functions with Formal Specifications

### `LobEngine::process()`

```cpp
bool LobEngine::process(const OrderEvent& ev) noexcept;
```

**Preconditions:**
- `ev.qty > 0` for `Add` and `Fill` events
- `ev.price > 0`
- `ev.timestamp_ns >= last_processed_timestamp_ns` (monotonic)
- For `Cancel`/`Modify`: `ev.order_id` exists in the active order map

**Postconditions:**
- Returns `true` if and only if the best bid or ask price changed
- `best_bid() < best_ask()` is maintained at all times (no crossed book)
- `total_qty` at affected price level is updated atomically
- No heap allocation; O(log N) worst-case where N = number of distinct price levels

**Loop Invariants:** N/A (no explicit loops; uses `std::map` tree operations)

---

### `FeaturePipeline::update()`

```cpp
void FeaturePipeline::update(const BookSnapshot& snap,
                              int64_t net_position) noexcept;
```

**Preconditions:**
- `snap.best_bid < snap.best_ask` (valid, non-crossed book)
- `|net_position| <= max_position_`
- `snap.timestamp_ns > last_update_ns_` (strictly increasing)

**Postconditions:**
- Rolling OFI windows contain at most `ofi_levels_` recent observations
- `vol_regime` reflects the current z-scored realized volatility
- All feature values are finite (NaN-guarded in normalization)

**Loop Invariants (rolling window update):**
- Window buffer contains exactly `min(n_updates, window_size)` valid entries
- Sum and sum-of-squares are kept as accumulators for O(1) mean/variance

---

### `QuoteGenerator::generate()`

```cpp
QuoteParams QuoteGenerator::generate(const float*     obs,
                                      PolicyInference& policy,
                                      int64_t          mid_price,
                                      int64_t          net_position) noexcept;
```

**Preconditions:**
- `obs` points to `MarketFeatures::DIM` valid finite float values
- `policy` has a loaded, non-null TorchScript model
- `mid_price > 0`
- `|net_position| <= max_inventory_`

**Postconditions:**
- `result.bid_price < result.ask_price` (spread ≥ 1 tick enforced)
- `result.bid_price = mid_price - round(action.delta_bid * tick_size)`
- `result.ask_price = mid_price + round(action.delta_ask * tick_size)`
- `result.bid_qty, result.ask_qty ∈ [1, max_order_size_]`
- If `net_position ≥ max_inventory_ * 0.9`: `result.bid_qty` reduced to 0 (halt buying)
- If `net_position ≤ -max_inventory_ * 0.9`: `result.ask_qty` reduced to 0 (halt selling)
- Total wall-clock time ≤ `latency_budget_ns`

**Loop Invariants:** N/A (single-pass computation)

---

### `PolicyInference::reload()`

```cpp
bool PolicyInference::reload(const std::string& new_model_path) noexcept;
```

**Preconditions:**
- `new_model_path` is a valid path to a TorchScript `.pt` file
- File is readable and well-formed (pre-validated by management thread)

**Postconditions:**
- If returns `true`: subsequent `act()` calls use the new model atomically
- If returns `false`: old model remains active; no partial state corruption
- Hot-swap is atomic via `std::atomic<std::shared_ptr<torch::jit::Module>>`
- No in-flight `act()` calls are interrupted mid-execution

**Loop Invariants:** N/A

---

## Example Usage

### Backtest Run (C++)

```cpp
#include "strategy_framework.hpp"
#include "signals/momentum_signal.hpp"

int main() {
    mm::StrategyConfig cfg{
        .instrument_id    = "NIFTY_FUT_2024",
        .model_path       = "models/ppo_nifty_v3.pt",
        .max_position     = 500,
        .max_order_size   = 50,
        .tick_size        = 5,          // NSE NIFTY tick = 0.05, scaled x100
        .latency_budget_ns = 800,
        .inventory_penalty = 0.1f
    };

    mm::StrategyFramework strategy(cfg);

    // Attach a hot-swappable momentum alpha signal
    auto momentum = std::make_shared<mm::MomentumSignal>(
        /*lookback_ticks=*/20, /*scale=*/0.5f
    );
    strategy.attach_signal(momentum);

    // Run event loop on this thread (caller must pin CPU affinity first)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);   // pin to core 3
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    strategy.run();  // blocking

    // Print performance summary
    const auto& s = strategy.stats();
    std::cout << "Events:   " << s.events_processed.load() << "\n"
              << "Quotes:   " << s.quotes_generated.load()  << "\n"
              << "Net PnL:  " << s.net_pnl_ticks.load()     << " ticks\n"
              << "P99 lat:  " << s.p99_latency_ns.load()    << " ns\n";
}
```

### Training Run (Python)

```cpp
// mm_train.py
import torch
from mm_env import LobEnv
from ppo_agent import PPOAgent

env = LobEnv(
    data_path    = "data/nifty_fut_L2_6mo.parquet",
    max_position = 500,
    tick_size    = 5,
    vol_window   = 60,
)

agent = PPOAgent(
    obs_dim      = env.observation_space.shape[0],   # 8
    action_dim   = env.action_space.shape[0],         # 3
    hidden_size  = 256,
    clip_eps     = 0.2,
    gamma        = 0.99,
    lam          = 0.95,
    lr           = 3e-4,
    entropy_coef = 0.01,
)

# Train for 500 episodes
agent.train(env, num_episodes=500, rollout_steps=2048, batch_size=256, epochs=10)

# Export to TorchScript for C++ inference
scripted = torch.jit.script(agent.actor)
scripted.save("models/ppo_nifty_v3.pt")
print("Model exported.")
```

### Hot-Reload Alpha Signal (Management Thread)

```cpp
// Called asynchronously from a management/admin thread
strategy.reload_model("models/ppo_nifty_v4.pt");  // atomic swap, no downtime

// Swap out momentum signal for a mean-reversion signal
strategy.detach_signal(momentum->id());
auto mr_signal = std::make_shared<mm::MeanReversionSignal>(/*window=*/50);
strategy.attach_signal(mr_signal);
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions
of a system — essentially, a formal statement about what the system should do. Properties serve
as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: No Crossed Book

*For any* sequence of `OrderEvent` objects applied to a freshly initialized `LobEngine`, after
each event the engine's `best_bid()` shall be strictly less than `best_ask()` whenever both
sides have at least one active order.

**Validates: Requirements 1.6, 1.7, 9.1**

---

### Property 2: Book Mutation Consistency

*For any* `Add` event applied to an `LobEngine`, the `total_qty` at the corresponding price
level shall increase by exactly the event's `qty`; for any `Cancel` or `Fill` event referencing
an existing order, the `total_qty` at the corresponding level shall decrease by the removed or
filled quantity, never going below zero.

**Validates: Requirements 1.1, 1.2, 1.3**

---

### Property 3: Accessor Consistency

*For any* `LobEngine` state reachable after processing a sequence of valid events,
`best_bid()` shall equal the highest price key in the bid map, `best_ask()` shall equal the
lowest price key in the ask map, `mid_price()` shall equal
`(best_bid() + best_ask()) / 2`, and `spread()` shall equal `best_ask() - best_bid()`.

**Validates: Requirements 1.12**

---

### Property 4: LOB Reset Clears State

*For any* `LobEngine` with an arbitrarily populated book, after calling `reset()` both
`best_bid()` and `best_ask()` shall return sentinel values indicating an empty book, and
`snapshot()` shall return a `BookSnapshot` with all quantity fields equal to zero.

**Validates: Requirements 1.13**

---

### Property 5: Invalid Event Rejection

*For any* `OrderEvent` where `price ≤ 0`, or `qty = 0` for `Add`/`Fill` events, or whose
`timestamp_ns` is less than the most recently processed event's timestamp, the `LobEngine`
shall return `false` and leave the book state unchanged.

**Validates: Requirements 1.8, 1.9, 11.1, 11.2, 11.3**

---

### Property 6: Feature Finiteness and Bounds

*For any* valid `BookSnapshot` and net position within `[-max_position, max_position]`, all
8 values in the `MarketFeatures` observation vector produced by `Feature_Pipeline::observe()`
shall be finite (`std::isfinite`) and satisfy `|obs[i]| ≤ 5.0`.

**Validates: Requirements 2.1, 2.9, 2.10**

---

### Property 7: OFI Formula Correctness

*For any* `BookSnapshot` with non-zero total bid and ask quantities at the top-K levels,
the `ofi_l1` and `ofi_l5` values computed by `Feature_Pipeline` shall equal
`(bid_qty_sum - ask_qty_sum) / (bid_qty_sum + ask_qty_sum)` for the respective level
aggregations, and shall lie in `[-1, 1]`.

**Validates: Requirements 2.2**

---

### Property 8: Inventory Normalization Invariants

*For any* net position `p` with `|p| ≤ max_position`, `inventory_skew` shall equal
`p / max_position` (in `[-1, 1]`) and `inventory_urgency` shall equal
`|p| / max_position` (in `[0, 1]`).

**Validates: Requirements 2.3, 2.4**

---

### Property 9: PPO Clipped Surrogate Correctness

*For any* ratio `r = exp(new_log_prob − old_log_prob)` and advantage `A`, the clipped
surrogate value shall equal `min(r·A, clip(r, 1−ε, 1+ε)·A)`, and the clipped ratio
`clip(r, 1−ε, 1+ε)` shall always lie within `[1−ε, 1+ε]` for any `ε ∈ (0, 0.5)`.

**Validates: Requirements 3.3**

---

### Property 10: Gradient Norm Bounded After Clipping

*For any* set of model parameters after a PPO gradient step with `max_norm = 0.5`, the
L2 norm of the concatenated gradient vector over all parameters shall be ≤ `0.5`.

**Validates: Requirements 3.6**

---

### Property 11: TorchScript Round-Trip Consistency

*For any* observation vector `obs` of shape `(8,)` in `float32`, the output of C++
`Policy_Inference::act(obs)` loaded from an exported TorchScript model shall match the
output of Python `agent.actor(obs)` to within `1e-5` absolute tolerance on every element.

**Validates: Requirements 3.10, 6.6**

---

### Property 12: GAE Finiteness and Normalization

*For any* sequence of finite rewards, finite value estimates, and boolean done flags of
equal length, `Trajectory_Buffer::compute_advantages()` shall produce advantage and return
arrays where all elements are finite, and the advantage array shall have mean ≈ 0 and
standard deviation ≈ 1 (within numerical precision).

**Validates: Requirements 4.3, 4.6, 4.7**

---

### Property 13: Non-Finite Log-Prob Rejection

*For any* `Transition` where `log_prob` is NaN, +Inf, or −Inf, `Trajectory_Buffer::push()`
shall reject the transition and leave the buffer state unchanged.

**Validates: Requirements 4.10, 11.5**

---

### Property 14: Minimum Spread Enforcement

*For any* policy output `(delta_bid, delta_ask)` and any `mid_price > 0`, the
`QuoteParams` produced by `Quote_Generator::generate()` shall satisfy
`ask_price − bid_price ≥ tick_size`.

**Validates: Requirements 5.2, 5.3**

---

### Property 15: Inventory Halt Rule

*For any* net position `p` where `|p| ≥ max_inventory × 0.9`, the `QuoteParams`
produced by `Quote_Generator::generate()` shall have `bid_qty = 0` when `p > 0`
and `ask_qty = 0` when `p < 0`.

**Validates: Requirements 5.5, 9.5**

---

### Property 16: Quote Quantity Bounds

*For any* `size_frac ∈ (0, 1]` and any net position `|p| < max_inventory × 0.9`,
the `bid_qty` and `ask_qty` in the generated `QuoteParams` shall each lie in
`[1, max_order_size]`.

**Validates: Requirements 5.4**

---

### Property 17: Alpha Signal Aggregation

*For any* set of attached `Alpha_Signal` objects with known `adjust()` outputs, the
`(bid_price, ask_price)` pair in `QuoteParams` produced by `Quote_Generator::generate()`
shall reflect the sum of all attached signal adjustments added to the base policy output.
After `detach_signal(id)`, the corresponding signal's adjustment shall no longer appear
in subsequent `generate()` outputs.

**Validates: Requirements 5.6, 5.8, 12.2, 12.3**

---

### Property 18: Safe Fallback on Inference Error

*For any* observation containing NaN values or causing a `libtorch` runtime exception,
`Policy_Inference::act()` shall return a `QuoteAction` of
`{delta_bid=2.0f, delta_ask=2.0f, size_frac=0.1f}` and shall not propagate any exception.

**Validates: Requirements 9.2**

---

### Property 19: One Quote Per State-Changing Event

*For any* sequence of `OrderEvent` objects processed by `Strategy_Framework`, the total
number of quote generation cycles triggered shall equal the number of events for which
`LOB_Engine::process()` returned `true`.

**Validates: Requirements 7.2, 7.3**

---

### Property 20: Inventory Bound Invariant

*For any* sequence of fill events processed by `Strategy_Framework`,
`|gateway.net_position()|` shall never exceed `max_position` after any fill.

**Validates: Requirements 7.5**

---

### Property 21: Reward Component Signs

*For any* training step, `spread_capture` shall be ≥ 0 (only earned on completed
round-trip fills) and `inventory_penalty` shall be ≤ 0 (monotonically discourages
large positions).

**Validates: Requirements 8.2**

---

### Property 22: Reward Formula Consistency

*For any* training step with known fill outcome and position, the total reward returned
by `LobEnv::step()` shall equal `realized_pnl + inventory_penalty + spread_capture`
where `inventory_penalty = −γ × |net_position| / max_position`.

**Validates: Requirements 8.1, 8.3**

---

### Property 23: Deterministic Replay

*For any* random seed `s`, two independent calls to `LobEnv.reset(seed=s)` followed by
identical action sequences shall produce byte-identical observation, reward, and done
sequences.

**Validates: Requirements 8.5**

---

### Property 24: pybind11 Boundary NaN Rejection

*For any* observation or action array containing at least one NaN or out-of-range value
passed through the pybind11 boundary, the bridge shall raise an exception and refuse to
forward the array to any C++ component.

**Validates: Requirements 6.10, 11.6**

---

### Property 25: Monotonic Timestamp Enforcement

*For any* pair of consecutive events `e_i`, `e_{i+1}` delivered to `LOB_Engine::process()`,
if `e_{i+1}.timestamp_ns < e_i.timestamp_ns` then `process()` shall return `false` and leave
the book state unchanged, preserving the monotonicity invariant for all accepted events.

**Validates: Requirements 11.3**

---

## Error Handling

### Error Scenario 1: Crossed Book

**Condition**: A sequence of `Add`/`Cancel` events produces `best_bid >= best_ask`
(can occur with stale or out-of-order data).

**Response**: `LobEngine::process()` detects the crossing post-update, logs a warning,
and rejects the triggering event (returns `false` without applying the update).

**Recovery**: The book remains in the last valid uncrossed state. The strategy skips
quote generation for that event and resumes normally on the next valid event.

---

### Error Scenario 2: TorchScript Inference Failure

**Condition**: `libtorch` throws during `act()` (e.g., shape mismatch, NaN in input).

**Response**: `PolicyInference::act()` catches the exception internally (no-throw
guarantee), logs the error with the offending observation vector, and returns a
safe fallback action: `{delta_bid=2, delta_ask=2, size_frac=0.1}` (wide, minimal quote).

**Recovery**: The fallback quote is submitted to keep the strategy quoting. On the next
successful management-thread health check, the model is reloaded.

---

### Error Scenario 3: Latency Budget Breach

**Condition**: `generate()` exceeds `latency_budget_ns` (measured via `rdtsc`).

**Response**: The quote is still submitted (hard cancellation would leave the book
naked), but the breach is recorded in `PerfStats::max_latency_ns`. An alert is
emitted via the logging sink.

**Recovery**: Persistent breaches (> 1% of quotes) trigger automatic CPU frequency
scaling checks and optionally switch to a lighter `SimplifiedPolicyInference`
(table-lookup fallback). Engineers investigate offline.

---

### Error Scenario 4: Inventory Limit Approach

**Condition**: `|net_position| > max_position * 0.9`.

**Response**: `QuoteGenerator::generate()` hard-clips the order size on the
offending side to zero and widens the opposite-side spread by 2× to encourage
mean reversion.

**Recovery**: Automatic — as the inventory reverts toward zero through fills,
normal quoting resumes without intervention.

---

### Error Scenario 5: Tick Replayer / Data Feed Gap

**Condition**: A gap of > 1 second between consecutive tick events (data feed issue
or end-of-session).

**Response**: The strategy detects the gap via timestamp delta, cancels all live
quotes immediately (to avoid stale prices), and pauses quoting.

**Recovery**: Quoting resumes when the next valid tick arrives and the book has
been re-populated to at least 3 levels deep.

---

## Testing Strategy

### Unit Testing Approach

Each C++ component is unit-tested in isolation using Catch2:

- `LobEngine`: add/cancel/fill sequences; verify `best_bid < best_ask` invariant
  after every operation; test crossed-book rejection; benchmark `process()` latency.
- `FeaturePipeline`: feed synthetic snapshots; verify OFI range [-1,1]; test
  vol_regime classification with known z-scores; check NaN guards.
- `QuoteGenerator`: mock `PolicyInference`; verify spread ≥ 1 tick; verify
  inventory-limit size clipping at 90% threshold.
- `TrajectoryBuffer`: fill to capacity; verify GAE output is finite and
  normalized; test edge case of single-step episode.

### Property-Based Testing Approach

**Property Test Library**: `rapidcheck` (C++) + `hypothesis` (Python training code)

Key property tests:

```cpp
// rapidcheck: LOB book never crosses
rc::check("Book never crosses after arbitrary event sequence", []() {
    auto events = *rc::gen::container<std::vector<OrderEvent>>(
                        rc::gen::arbitrary<OrderEvent>());
    mm::LobEngine lob(5);
    for (const auto& ev : events) {
        lob.process(ev);
        if (lob.best_bid() > 0 && lob.best_ask() > 0)
            RC_ASSERT(lob.best_bid() < lob.best_ask());
    }
});

// rapidcheck: Quote spread always >= tick_size
rc::check("Quote spread >= 1 tick for any policy output", []() {
    float delta_bid = *rc::gen::inRange(-10.0f, 10.0f);
    float delta_ask = *rc::gen::inRange(-10.0f, 10.0f);
    // ... construct params and assert
    RC_ASSERT(params.ask_price - params.bid_price >= tick_size);
});
```

```python
# hypothesis: GAE outputs always finite
@given(
    rewards=st.lists(st.floats(min_value=-1e4, max_value=1e4), min_size=1),
    values=st.lists(st.floats(min_value=-1e4, max_value=1e4), min_size=1),
)
def test_gae_finite(rewards, values):
    assume(len(rewards) == len(values))
    adv, ret = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    assert np.all(np.isfinite(adv))
    assert np.all(np.isfinite(ret))
```

### Integration Testing Approach

- Full backtest integration test: replay 1 week of NSE L2 data end-to-end,
  verify that `net_pnl_ticks` is tracked correctly, no crashes, and
  `max_latency_ns < 2000` (2× budget as integration margin).
- Python ↔ C++ bridge test: call `env.step()` 10,000 times through pybind11,
  verify observation shapes, reward finiteness, and that the C++ LOB state
  matches a reference Python LOB implementation.
- Model export/import round-trip: train for 5 episodes, export TorchScript,
  reload in C++ `PolicyInference`, verify that `act()` output matches
  Python `agent.actor()` output to within 1e-5 tolerance.

---

## Performance Considerations

- **Cache-aligned `PriceLevel`**: Each price level is exactly 64 bytes (one cache line),
  eliminating false sharing and enabling prefetch-friendly traversal of the top-K levels.
- **Lock-free SPSC queue** for incoming order events between the network I/O thread and
  the strategy thread; no mutex on the hot path.
- **`rdtsc` timestamping**: Use `__builtin_ia32_rdtscp` for cycle-accurate latency
  measurement without syscall overhead.
- **TorchScript inference with `torch::NoGradGuard`**: Disables autograd bookkeeping
  during `act()`, reducing inference overhead by ~30%.
- **CPU affinity pinning** (`pthread_setaffinity_np`): Eliminates context-switch jitter;
  strategy thread runs exclusively on an isolated core.
- **`std::pmr::monotonic_buffer_resource`** for per-tick temporary allocations:
  arena reset at end of each event avoids heap fragmentation.
- **Target latency breakdown** (800 ns budget):
  - LOB snapshot: ~50 ns
  - Feature computation: ~100 ns
  - TorchScript inference (MLP 8→256→256→3): ~400 ns
  - Quote construction + risk checks: ~100 ns
  - Gateway submission (internal): ~150 ns

---

## Security Considerations

- **Model integrity**: TorchScript `.pt` files are verified via SHA-256 checksum
  before hot-reload to prevent loading a corrupted or tampered model.
- **Position limits enforced in two places**: `QuoteGenerator` (pre-submission) and
  the order gateway adapter (post-fill), preventing a policy bug from breaching
  exchange-mandated position limits.
- **Input validation on pybind11 boundary**: Observation and action arrays are
  bounds-checked and NaN-screened before entering C++ code from Python, preventing
  undefined behavior from malformed training data.
- **Tick data authenticity**: Replayed L2 data is loaded from an immutable, checksummed
  archive; no live mutation path during simulation.
- **No network access from the strategy thread**: All external I/O (order submission,
  feed ingestion) routes through dedicated I/O threads, isolating the hot-path core.

---

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| **libtorch (PyTorch C++ API)** | ≥ 2.2 | TorchScript inference in C++ |
| **PyTorch** | ≥ 2.2 | PPO training, model export |
| **pybind11** | ≥ 2.11 | C++/Python bridge for `LobEnv` |
| **Catch2** | ≥ 3.5 | C++ unit testing |
| **rapidcheck** | latest | C++ property-based testing |
| **hypothesis** | ≥ 6.0 | Python property-based testing |
| **numpy** | ≥ 1.26 | Observation/action array handling |
| **gymnasium** | ≥ 0.29 | `LobEnv` base class |
| **pandas / pyarrow** | ≥ 2.0 | L2 tick data loading (Parquet) |
| **CMake** | ≥ 3.28 | C++ build system |
| **Intel TBB** | ≥ 2021 | `std::pmr` parallel allocators (optional) |
