# Requirements Document

## Introduction

This document specifies the functional and non-functional requirements for the RL-Driven Adaptive
Market Making Engine. The system pairs a high-performance C++ limit-order-book (LOB) simulator
with a Python/PyTorch PPO reinforcement-learning agent to generate adaptive bid-ask quotes
conditioned on real-time inventory risk, order-flow imbalance, and volatility regime.

The engine targets sub-microsecond LOB event processing in simulation and an end-to-end quote
generation latency ≤ 800 ns on a single CPU-affinity-pinned core. Backtested targets are a
Sharpe ratio ≥ 2.7 and ≥ 34% PnL drawdown reduction versus a fixed-spread baseline, evaluated
across NSE liquid futures contracts.

---

## Glossary

- **LOB_Engine**: The C++ component that maintains a price-time priority limit order book for a
  single instrument and processes add/cancel/modify/fill events.
- **Feature_Pipeline**: The C++ component that converts raw `BookSnapshot` data and recent trade
  history into a fixed-length observation vector for the PPO agent.
- **PPO_Agent**: The Python/PyTorch Proximal Policy Optimization agent that trains a stochastic
  actor-critic policy mapping market observations to quote-skew actions.
- **Policy_Inference**: The C++ wrapper around a frozen TorchScript model that performs
  synchronous inference to produce quote actions during live operation.
- **Quote_Generator**: The C++ component that translates policy actions into concrete bid/ask
  limit orders, applying risk checks and size scaling.
- **Strategy_Framework**: The C++ orchestrator that drives the full event loop from tick replay
  through LOB update, feature extraction, policy inference, quote generation, and order submission.
- **LobEnv**: The Gym-compatible Python wrapper around the C++ LOB_Engine and Feature_Pipeline,
  used during PPO training.
- **Trajectory_Buffer**: The data structure that stores PPO rollout transitions and computes
  GAE-λ advantages for training.
- **TorchScript_Model**: A frozen, serialized PyTorch model exported via `torch.jit.script`,
  loaded by `Policy_Inference` for C++ inference.
- **Alpha_Signal**: A hot-swappable plugin implementing `IAlphaSignal` that provides additive
  bid/ask spread adjustments from an external alpha source.
- **Order_Gateway**: The component responsible for submitting limit orders to the exchange and
  receiving fill/cancel events.
- **OFI**: Order-Flow Imbalance — the normalized difference between bid and ask queue depth.
- **GAE**: Generalized Advantage Estimation — the λ-weighted advantage estimator used in PPO.
- **Tick_Replayer**: The component that reads historical L2 tick data and feeds it into the
  LOB_Engine in chronological order.
- **BookSnapshot**: A point-in-time capture of the top-10 bid and ask price levels produced
  by the LOB_Engine after each state-changing event.
- **PriceLevel**: A 64-byte cache-aligned struct holding the total quantity, order count, and
  price for one side of the book at a given price tier.

---

## Requirements

### Requirement 1: Limit Order Book Management

**User Story:** As a quantitative developer, I want a high-performance limit order book engine,
so that I can simulate realistic market microstructure events at sub-microsecond latency.

#### Acceptance Criteria

1. WHEN an `Add` order event is received, THE LOB_Engine SHALL insert the order at the correct
   price level maintaining price-time priority.
2. WHEN a `Cancel` order event is received for an existing order, THE LOB_Engine SHALL remove
   that order from its price level and update the total quantity atomically.
3. WHEN a `Fill` or `PartialFill` event is received, THE LOB_Engine SHALL reduce the quantity
   at the affected price level by the filled amount.
4. WHEN a `Modify` event is received for an existing order, THE LOB_Engine SHALL update the
   order's price or quantity while preserving time priority relative to remaining orders at
   the same price.
5. AFTER processing any state-changing event, THE LOB_Engine SHALL emit a `BookSnapshot`
   containing the current top-10 bid and ask price levels.
6. THE LOB_Engine SHALL maintain the invariant that `best_bid < best_ask` at all times
   (no crossed book).
7. IF a received event would produce a crossed book state, THEN THE LOB_Engine SHALL reject
   the event and return `false` without modifying the book state.
8. IF a `Cancel` or `Modify` event references an `order_id` that does not exist in the active
   order map, THEN THE LOB_Engine SHALL reject the event and return `false`.
9. IF an `Add` or `Fill` event has `qty = 0`, THEN THE LOB_Engine SHALL reject the event and
   return `false`.
10. THE LOB_Engine SHALL process each order event with no heap allocation on the hot path.
11. THE LOB_Engine SHALL process each order event in O(log N) time, where N is the number of
    distinct price levels currently active.
12. THE LOB_Engine SHALL expose `best_bid()`, `best_ask()`, `mid_price()`, and `spread()`
    accessors that return correct values in O(1) time.
13. THE LOB_Engine SHALL expose a `reset()` operation that clears all book state to empty.

---

### Requirement 2: Market Feature Extraction

**User Story:** As a machine learning engineer, I want a feature pipeline that converts raw book
data into normalized observation vectors, so that the PPO agent receives stable, informative
inputs for training and inference.

#### Acceptance Criteria

1. WHEN `update()` is called with a `BookSnapshot` and current net position, THE
   Feature_Pipeline SHALL compute and store the 8-dimensional `MarketFeatures` observation
   vector.
2. THE Feature_Pipeline SHALL compute OFI at the level-1 (`ofi_l1`) and top-5 (`ofi_l5`)
   depths as `(bid_qty - ask_qty) / (bid_qty + ask_qty)` over the respective top-K levels.
3. THE Feature_Pipeline SHALL compute `inventory_skew` as the signed net position normalized
   to `[-1, 1]` by dividing by `max_position`.
4. THE Feature_Pipeline SHALL compute `inventory_urgency` as `|net_position| / max_position`,
   constrained to `[0, 1]`.
5. THE Feature_Pipeline SHALL compute `vol_regime` as the z-scored rolling realized volatility
   using a configurable window (default 60 seconds).
6. THE Feature_Pipeline SHALL compute `micro_price_offset` as
   `(ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)` relative to mid-price.
7. THE Feature_Pipeline SHALL compute `spread_ticks` as the current bid-ask spread measured
   in integer ticks.
8. THE Feature_Pipeline SHALL compute `session_progress` as the fraction of elapsed trading
   session time, in `[0, 1]`.
9. THE Feature_Pipeline SHALL ensure all feature values are finite (NaN-guarded) after every
   `update()` call.
10. THE Feature_Pipeline SHALL normalize all feature values to approximately `[-1, 1]` for
    stable neural network training, applying a 5-sigma guard (`|obs[i]| ≤ 5.0`).
11. THE Feature_Pipeline SHALL provide a zero-copy `observe()` method that writes the current
    `MarketFeatures` into a caller-supplied buffer without dynamic allocation.
12. THE Feature_Pipeline SHALL maintain rolling OFI and realized-volatility windows using O(1)
    mean/variance accumulators (running sum and sum-of-squares).
13. THE Feature_Pipeline SHALL expose a `reset()` operation that clears all window state.

---

### Requirement 3: PPO Reinforcement Learning Agent

**User Story:** As a quantitative researcher, I want a PPO agent that learns to adaptively
skew bid-ask spreads based on market conditions, so that the strategy earns spread capture
while controlling inventory risk.

#### Acceptance Criteria

1. THE PPO_Agent SHALL implement a 3-layer MLP actor network with a Gaussian head that outputs
   mean and log-standard-deviation for each of the 3 action dimensions
   (`delta_bid`, `delta_ask`, `size_frac`).
2. THE PPO_Agent SHALL implement a critic network that shares the trunk with the actor and
   outputs a scalar state-value estimate `V(s)`.
3. WHEN computing a policy update, THE PPO_Agent SHALL apply the clipped surrogate objective:
   `L_CLIP = E[min(r·A, clip(r, 1−ε, 1+ε)·A)]` where `ε ∈ (0, 0.5)`.
4. THE PPO_Agent SHALL apply an entropy bonus weighted by `entropy_coef` to encourage
   exploration during training.
5. THE PPO_Agent SHALL use GAE-λ advantage estimation with configurable `gamma ∈ (0,1)` and
   `lam ∈ (0,1)`.
6. WHEN updating policy parameters, THE PPO_Agent SHALL clip gradient norms to a maximum of
   `0.5` to bound parameter updates.
7. THE PPO_Agent SHALL train using the Adam optimizer with a configurable learning rate.
8. THE PPO_Agent SHALL accept an observation space of shape `(8,)` in `float32` and an action
   space of shape `(3,)` corresponding to `(delta_bid, delta_ask, size_frac)`.
9. WHEN a training episode completes, THE PPO_Agent SHALL clear the trajectory buffer and
   begin collecting a new rollout.
10. AFTER training, THE PPO_Agent SHALL export the actor network as a frozen TorchScript model
    via `torch.jit.script` for use in C++ inference.
11. THE PPO_Agent training SHALL achieve a Sharpe ratio ≥ 2.7 and ≥ 34% PnL drawdown
    reduction versus a fixed-spread baseline when evaluated across NSE liquid futures
    backtests.

---

### Requirement 4: Trajectory Buffer and GAE Computation

**User Story:** As a machine learning engineer, I want a correct and numerically stable
trajectory buffer, so that the PPO agent trains on well-formed advantage estimates without
NaN or overflow.

#### Acceptance Criteria

1. THE Trajectory_Buffer SHALL store transitions containing observation, action, log-probability,
   state-value, reward, and done flag in a contiguous, pre-allocated memory region of
   configurable capacity.
2. THE Trajectory_Buffer capacity SHALL be a power of two to support vectorized GAE computation.
3. WHEN `compute_advantages()` is called, THE Trajectory_Buffer SHALL compute GAE-λ advantages
   by iterating transitions in reverse chronological order.
4. WHEN computing GAE-λ, IF a transition has `done = true`, THEN THE Trajectory_Buffer SHALL
   use `next_value = 0` as the bootstrap value for that step.
5. WHEN `compute_advantages()` is called and the final step is non-terminal, THE
   Trajectory_Buffer SHALL bootstrap using the provided `last_value` from the critic.
6. AFTER `compute_advantages()` completes, THE Trajectory_Buffer SHALL normalize advantages
   to have mean ≈ 0 and standard deviation ≈ 1, using an epsilon of `1e-8` to prevent
   division by zero.
7. THE Trajectory_Buffer SHALL guarantee that all advantage and return values are finite after
   `compute_advantages()` completes.
8. THE Trajectory_Buffer SHALL expose zero-copy tensor views (`obs_tensor`, `action_tensor`,
   `advantage_tensor`, `return_tensor`) backed by contiguous internal buffers for efficient
   transfer to Python/PyTorch.
9. THE Trajectory_Buffer SHALL expose a `clear()` operation that resets the buffer for the
   next rollout without releasing allocated memory.
10. IF a pushed transition contains a non-finite `log_prob`, THEN THE Trajectory_Buffer SHALL
    reject the push and log an error to prevent log(0) propagation.

---

### Requirement 5: Quote Generation and Risk Controls

**User Story:** As a risk manager, I want the quote generator to enforce hard risk limits on
every generated quote, so that the strategy never breaches inventory or spread constraints
regardless of policy output.

#### Acceptance Criteria

1. WHEN `generate()` is called, THE Quote_Generator SHALL invoke `Policy_Inference::act()` and
   translate the continuous action `(delta_bid, delta_ask)` into absolute integer tick prices
   relative to mid-price.
2. THE Quote_Generator SHALL enforce a minimum spread of 1 tick: `ask_price − bid_price ≥ tick_size`.
3. THE Quote_Generator SHALL compute bid and ask prices as:
   `bid_price = mid_price − round(delta_bid × tick_size)` and
   `ask_price = mid_price + round(delta_ask × tick_size)`.
4. THE Quote_Generator SHALL scale `size_frac` to produce `bid_qty` and `ask_qty` in the range
   `[1, max_order_size]`.
5. WHEN `|net_position| ≥ max_inventory × 0.9`, THE Quote_Generator SHALL set `bid_qty = 0`
   on the long side and/or `ask_qty = 0` on the short side to halt further inventory
   accumulation on the offending side.
6. THE Quote_Generator SHALL aggregate adjustments from all attached Alpha_Signals by summing
   their `(delta_bid_adj, delta_ask_adj)` outputs atomically using a lock-free signal list.
7. WHEN `generate()` is called, THE Quote_Generator SHALL complete within the configured
   `latency_budget_ns` (default 800 ns) as measured by `rdtsc`.
8. THE Quote_Generator SHALL support attaching and detaching Alpha_Signals at runtime without
   interrupting the event loop or requiring gateway reconnection.
9. THE Quote_Generator SHALL assign a TTL to each emitted quote via `valid_until_ns` so that
   stale quotes can be cancelled by the Order_Gateway.

---

### Requirement 6: C++/Python Bridge and TorchScript Integration

**User Story:** As a systems engineer, I want seamless interoperability between the C++ LOB
engine and the Python training harness, so that the policy trained in Python can be deployed
in C++ with no functional difference.

#### Acceptance Criteria

1. THE LobEnv SHALL expose a Gym-compatible interface (`reset()`, `step()`, `render()`) backed
   by the C++ LOB_Engine and Feature_Pipeline via pybind11 bindings.
2. WHEN `step()` is called with an action array, THE LobEnv SHALL apply the action to the
   LOB_Engine, advance the simulation by one tick event, and return
   `(observation, reward, done, info)`.
3. THE LobEnv observation space SHALL be `gym.spaces.Box` with shape `(8,)` and dtype
   `float32`, matching `MarketFeatures::DIM`.
4. THE LobEnv action space SHALL be `gym.spaces.Box` with shape `(3,)` representing
   `(delta_bid, delta_ask, size_frac)`.
5. THE Policy_Inference SHALL load a TorchScript `.pt` file and perform synchronous inference
   via `libtorch` without requiring a Python interpreter at runtime.
6. WHEN `act()` is called on Policy_Inference, THE Policy_Inference SHALL produce a
   `QuoteAction` output that matches the Python `agent.actor()` output to within `1e-5`
   absolute tolerance for identical observations.
7. WHEN `reload()` is called on Policy_Inference with a valid new model path, THE
   Policy_Inference SHALL atomically swap the active TorchScript model using
   `std::atomic<std::shared_ptr<torch::jit::Module>>` so that no in-flight `act()` call
   is interrupted.
8. IF `reload()` fails (invalid path, malformed model, or checksum mismatch), THEN THE
   Policy_Inference SHALL leave the currently active model unchanged and return `false`.
9. BEFORE loading a TorchScript model, THE Policy_Inference SHALL verify the file's SHA-256
   checksum against a stored reference to prevent loading corrupted or tampered models.
10. THE pybind11 boundary SHALL validate that all observation and action arrays are within
    expected bounds and contain no NaN values before passing data from Python into C++ code.

---

### Requirement 7: Strategy Event Loop and Orchestration

**User Story:** As a trading system operator, I want a strategy framework that orchestrates
the full market-making pipeline in a deterministic, low-latency event loop, so that every
LOB state change produces exactly one consistent quote update within budget.

#### Acceptance Criteria

1. THE Strategy_Framework SHALL execute a blocking event loop on a single CPU-affinity-pinned
   thread that processes incoming tick events in order.
2. WHEN an order event changes LOB state, THE Strategy_Framework SHALL trigger exactly one
   quote generation cycle: snapshot → feature update → policy inference → quote submission.
3. WHEN an order event does not change LOB state, THE Strategy_Framework SHALL skip quote
   generation and proceed to the next event.
4. THE Strategy_Framework SHALL cancel stale quotes via the Order_Gateway before submitting
   each new quote to avoid duplicate positions.
5. WHILE the event loop is running, THE Strategy_Framework SHALL maintain
   `|gateway.net_position()| ≤ max_position` as a hard invariant enforced on every fill event.
6. THE Strategy_Framework SHALL expose `reload_model()` and `attach_signal()` / `detach_signal()`
   operations callable from a separate management thread without halting or disrupting the
   event loop.
7. THE Strategy_Framework SHALL record per-cycle latency samples via `rdtsc` and maintain
   lock-free `PerfStats` counters for `events_processed`, `quotes_generated`, `net_pnl_ticks`,
   `max_latency_ns`, and `p99_latency_ns`.
8. THE Strategy_Framework event loop SHALL contain no heap allocations; all per-tick temporary
   memory SHALL use a `std::pmr::monotonic_buffer_resource` arena reset after each event.
9. WHEN a hot-reload of the policy model is pending, THE Strategy_Framework SHALL complete the
   atomic model swap between quote generation cycles without inserting latency into the hot path.
10. THE Strategy_Framework SHALL accept a `StrategyConfig` struct specifying instrument ID,
    model path, max position, max order size, tick size, latency budget, and inventory penalty.

---

### Requirement 8: Reward Signal and Training Environment

**User Story:** As a quantitative researcher, I want a well-defined reward signal that
incentivises spread capture while penalising inventory risk, so that the PPO agent converges
to a profitable, risk-controlled market-making policy.

#### Acceptance Criteria

1. WHEN a fill event is processed during training, THE LobEnv SHALL compute the reward as
   `realized_pnl + inventory_penalty + spread_capture`.
2. THE LobEnv reward function SHALL set `spread_capture ≥ 0` (only earned on completed
   round-trip fills) and `inventory_penalty = −γ × |net_position| / max_position ≤ 0` at
   every step.
3. THE LobEnv SHALL pass `gamma` (inventory penalty coefficient γ) as a configurable
   hyperparameter matching `StrategyConfig.inventory_penalty`.
4. THE LobEnv SHALL emit a `done = true` signal at end-of-session or when position limits are
   breached beyond recovery.
5. THE LobEnv SHALL reproduce the same trajectory deterministically when re-seeded with the
   same random seed, enabling reproducible training experiments.
6. THE LobEnv SHALL load NSE L2 tick data from a Parquet file specified by `data_path` using
   `pandas`/`pyarrow`, supporting at least 6 months of tick history per instrument.
7. THE LobEnv SHALL support `render(mode="human")` to display a human-readable summary of
   the current book state, position, and cumulative PnL for debugging.

---

### Requirement 9: Error Handling and Fault Tolerance

**User Story:** As a trading system operator, I want the engine to degrade gracefully under
fault conditions, so that the strategy continues quoting or safely halts without corrupting
book state or breaching risk limits.

#### Acceptance Criteria

1. WHEN the LOB_Engine detects that processing an event would produce a crossed book,
   THE LOB_Engine SHALL reject the event, log a warning, and maintain the last valid
   uncrossed book state.
2. WHEN `Policy_Inference::act()` encounters a runtime exception (e.g. shape mismatch, NaN
   input), THE Policy_Inference SHALL catch the exception internally, log the offending
   observation, and return a safe fallback action of
   `{delta_bid=2, delta_ask=2, size_frac=0.1}`.
3. WHEN the quote generation latency exceeds `latency_budget_ns`, THE Strategy_Framework SHALL
   still submit the quote, record the breach in `PerfStats::max_latency_ns`, and emit an alert
   via the logging sink.
4. WHEN persistent latency breaches exceed 1% of generated quotes, THE Strategy_Framework SHALL
   automatically switch to a `SimplifiedPolicyInference` fallback and emit an operational alert.
5. WHEN `|net_position|` approaches `max_position × 0.9`, THE Quote_Generator SHALL reduce
   the offending-side quote quantity to 0 and widen the opposite-side spread by 2× to
   encourage inventory mean reversion.
6. WHEN a tick data gap exceeds 1 second between consecutive events, THE Strategy_Framework
   SHALL cancel all live quotes immediately and suspend quoting until the next valid tick
   arrives with at least 3 book levels populated.
7. IF a `reload()` call on Policy_Inference fails checksum verification or model loading,
   THEN THE Policy_Inference SHALL retain the currently active model unchanged and log the
   failure details.
8. WHEN the Strategy_Framework is shut down, THE Strategy_Framework SHALL cancel all open
   orders via the Order_Gateway before terminating the event loop.

---

### Requirement 10: Latency and Performance

**User Story:** As a high-frequency trading engineer, I want the engine to meet strict latency
and throughput targets, so that the system is viable for production market-making on NSE
futures.

#### Acceptance Criteria

1. THE Quote_Generator SHALL produce a complete `QuoteParams` output within 800 ns
   end-to-end (policy inference + risk checks + price construction) as measured by `rdtsc`
   on a CPU-affinity-pinned core.
2. THE LOB_Engine SHALL process each `OrderEvent` in sub-microsecond time (< 1000 ns) on the
   hot path.
3. THE Strategy_Framework event loop SHALL exhibit a P99 quote-generation latency of
   ≤ 800 ns over any 10,000-event window.
4. THE Feature_Pipeline `update()` call SHALL complete in ≤ 100 ns per event using O(1)
   rolling accumulators.
5. THE Policy_Inference `act()` call with a 3-layer MLP (8→256→256→3) and `torch::NoGradGuard`
   SHALL complete in ≤ 400 ns per inference call.
6. THE LOB_Engine `PriceLevel` struct SHALL be exactly 64 bytes and cache-aligned to eliminate
   false sharing.
7. THE Strategy_Framework event loop SHALL contain no system calls or mutex acquisitions on
   the hot path; all inter-thread coordination SHALL use lock-free data structures.
8. THE Tick_Replayer SHALL deliver events to the LOB_Engine via a lock-free SPSC queue with
   no blocking on the strategy thread.

---

### Requirement 11: Data Integrity and Validation

**User Story:** As a quantitative developer, I want all data flowing through the system to be
validated at component boundaries, so that malformed inputs are caught early and never
silently corrupt system state.

#### Acceptance Criteria

1. THE LOB_Engine SHALL reject any `OrderEvent` where `price ≤ 0`.
2. THE LOB_Engine SHALL reject any `Add` or `Fill` `OrderEvent` where `qty = 0`.
3. THE LOB_Engine SHALL reject any `OrderEvent` whose `timestamp_ns` is less than the
   timestamp of the previously processed event (monotonicity violation).
4. THE Feature_Pipeline SHALL guard all normalization computations against division by zero
   and replace any resulting NaN or Inf values with 0.
5. THE Trajectory_Buffer SHALL reject any pushed `Transition` where `log_prob` is non-finite.
6. THE pybind11 boundary SHALL screen all observation and action arrays for NaN and out-of-range
   values before forwarding them to C++ components.
7. THE Tick_Replayer SHALL load tick data exclusively from immutable, checksummed Parquet
   archives and reject any file whose checksum does not match the recorded value.
8. THE TorchScript model file SHALL be verified against a SHA-256 checksum before loading;
   THE Policy_Inference SHALL reject any file that fails verification.

---

### Requirement 12: Hot-Swappable Alpha Signals

**User Story:** As a quantitative researcher, I want to attach, detach, and replace alpha
signals at runtime, so that I can experiment with signal combinations without gateway
reconnection or strategy downtime.

#### Acceptance Criteria

1. THE Quote_Generator SHALL maintain a lock-free list of attached Alpha_Signals and iterate
   it on every `generate()` call to accumulate additive spread adjustments.
2. WHEN `attach_signal()` is called, THE Quote_Generator SHALL add the new Alpha_Signal to the
   active list atomically so that the next `generate()` call uses the updated list.
3. WHEN `detach_signal()` is called with a valid signal ID, THE Quote_Generator SHALL remove
   the matching Alpha_Signal atomically; subsequent `generate()` calls SHALL NOT apply that
   signal's adjustments.
4. IF `detach_signal()` is called with an ID that does not match any attached signal, THEN THE
   Quote_Generator SHALL take no action and return without error.
5. THE Alpha_Signal interface SHALL expose an `id()` method returning a unique `uint32_t`
   identifier and an `adjust()` method returning a pair of `(delta_bid_adj, delta_ask_adj)`
   float adjustments in ticks.
6. WHEN the Strategy_Framework calls `attach_signal()` or `detach_signal()` from a management
   thread while the event loop is running, THE Quote_Generator SHALL complete the atomic
   update without blocking or stalling the strategy thread.

---

### Requirement 13: Observability and Performance Monitoring

**User Story:** As a trading system operator, I want real-time performance counters and
structured logging, so that I can monitor system health and diagnose issues without
interrupting the event loop.

#### Acceptance Criteria

1. THE Strategy_Framework SHALL maintain lock-free atomic counters for `events_processed`,
   `quotes_generated`, `net_pnl_ticks`, `max_latency_ns`, and `p99_latency_ns` that are
   readable from any thread without blocking the event loop.
2. THE Strategy_Framework SHALL record `rdtsc`-based latency for every quote generation cycle
   and update `max_latency_ns` and `p99_latency_ns` atomically.
3. WHEN a crossed-book event is rejected, THE LOB_Engine SHALL emit a structured log entry
   containing the timestamp, offending `OrderEvent`, and the current best bid and ask.
4. WHEN a latency breach occurs, THE Strategy_Framework SHALL emit a structured log entry
   containing the cycle timestamp, measured latency, and budget threshold.
5. WHEN `Policy_Inference::act()` falls back to the safe default action, THE Policy_Inference
   SHALL log the offending observation vector and the exception message.
6. THE LobEnv `render()` method SHALL output a human-readable summary of current book state,
   net position, and cumulative PnL to standard output.
