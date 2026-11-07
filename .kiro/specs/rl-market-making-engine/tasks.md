# Implementation Plan: RL-Driven Adaptive Market Making Engine

## Overview

Implementation is split across three tiers:
1. **C++ Core** — LOB engine, feature pipeline, quote generator, strategy framework (CMake project)
2. **Python Training Harness** — PPO agent, trajectory buffer, reward shaping, Gym environment
3. **C++/Python Bridge** — pybind11 bindings, TorchScript export/import, policy inference wrapper

Tasks progress from foundational data structures and interfaces upward through integration,
following the dependency chain: data models → LOB engine → feature pipeline → PPO training →
TorchScript bridge → quote generation → strategy orchestration → observability.

--- 

- [x] 1. Project scaffold and build system
  - Create CMake project structure: `src/cpp/`, `src/python/`, `tests/`, `models/`, `data/`
  - Write top-level `CMakeLists.txt` linking libtorch (≥ 2.2), pybind11 (≥ 2.11), Catch2 (≥ 3.5), rapidcheck
  - Add `pyproject.toml` / `setup.py` for the Python package with gymnasium (≥ 0.29), torch (≥ 2.2), hypothesis (≥ 6.0), pandas/pyarrow (≥ 2.0)
  - Define shared header `include/mm/types.hpp` with `Side`, `VolRegime`, and compile-time constants
  - _Requirements: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10_


- [x] 2. Core data models
  - [x] 2.1 Implement `OrderEvent`, `PriceLevel`, `BookSnapshot` structs in `include/mm/lob_types.hpp`
    - `PriceLevel` must be exactly 64 bytes and `__attribute__((aligned(64)))` — add `static_assert(sizeof(PriceLevel) == 64)`
    - `OrderEvent` must be exactly 32 bytes — add `static_assert(sizeof(OrderEvent) == 32)`
    - Include `PartialFill` in `OrderEvent::Type` enum
    - _Requirements: 1.10, 10.6, 11.1_

  - [x] 2.2 Implement `MarketFeatures`, `QuoteAction`, `QuoteParams`, `RewardSignal`, `VolatilityState` structs in `include/mm/data_models.hpp`
    - `MarketFeatures::DIM = 8` as `static constexpr int`
    - Implement `MarketFeatures::to_array(float* buf)` writing DIM floats zero-copy
    - Implement `RewardSignal::total()` and validate sign constraints as inline assertions
    - _Requirements: 2.1, 3.1, 5.1, 8.1, 8.2_

  - [x] 2.3 Write unit tests for data model layout and invariants (Catch2)
    - Assert `sizeof(PriceLevel) == 64`, `sizeof(OrderEvent) == 32`, alignof checks
    - Test `RewardSignal::total()` with known component values
    - _Requirements: 1.10, 10.6_

 
- [x] 3. LOB Engine implementation
  - [x] 3.1 Implement `LobEngine` class in `src/cpp/lob_engine.cpp` / `include/mm/lob_engine.hpp`
    - Use `std::map<int64_t, PriceLevel>` for bid (reverse) and ask (forward) sides
    - Implement `process()` dispatching Add / Cancel / Modify / Fill / PartialFill with O(log N) price-level ops
    - Maintain internal `std::unordered_map<uint64_t, OrderEntry>` for O(1) order_id lookup
    - Reject events: `price ≤ 0`, `qty = 0` for Add/Fill, timestamp monotonicity violations, unknown order_id for Cancel/Modify
    - Detect and reject crossed-book attempts without mutating state
    - Implement `snapshot()`, `best_bid()`, `best_ask()`, `mid_price()`, `spread()`, `reset()` accessors
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 9.1, 11.1, 11.2, 11.3_

  - [x] 3.2 Write property test for no-crossed-book invariant (rapidcheck)
    - **Property 1: No Crossed Book** — for any sequence of arbitrary `OrderEvent`s applied to a fresh `LobEngine`, after each event `best_bid() < best_ask()` whenever both sides are non-empty
    - **Validates: Requirements 1.6, 1.7, 9.1**

  - [x] 3.3 Write property test for book mutation consistency (rapidcheck)
    - **Property 2: Book Mutation Consistency** — for any `Add` event, `total_qty` at the price level increases by `qty`; for any `Cancel`/`Fill`, it decreases by removed/filled qty, never below zero
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 3.4 Write property test for accessor consistency (rapidcheck)
    - **Property 3: Accessor Consistency** — `best_bid()` = highest bid key, `best_ask()` = lowest ask key, `mid_price()` = `(best_bid + best_ask) / 2`, `spread()` = `best_ask - best_bid`
    - **Validates: Requirements 1.12**

  - [x] 3.5 Write property test for reset behavior (rapidcheck)
    - **Property 4: LOB Reset Clears State** — after `reset()`, both accessors return sentinel empty values and `snapshot()` has all qty fields = 0
    - **Validates: Requirements 1.13**

  - [x] 3.6 Write property test for invalid event rejection (rapidcheck)
    - **Property 5: Invalid Event Rejection** — events with `price ≤ 0`, `qty = 0` on Add/Fill, or backward timestamp return `false` and leave book unchanged
    - **Validates: Requirements 1.8, 1.9, 11.1, 11.2, 11.3**

  - [x] 3.7 Write property test for monotonic timestamp enforcement (rapidcheck)
    - **Property 25: Monotonic Timestamp Enforcement** — if `e_{i+1}.timestamp_ns < e_i.timestamp_ns`, `process()` returns `false` and leaves state unchanged
    - **Validates: Requirements 11.3**

  - [x] 3.8 Write unit tests for LOB engine edge cases (Catch2)
    - Test Add, Cancel, Fill, Modify sequences; verify best bid/ask after each; latency benchmark for `process()` under 1000 ns
    - _Requirements: 1.1–1.13, 10.2_


- [x] 4. Feature Pipeline implementation
  - [x] 4.1 Implement `FeaturePipeline` class in `src/cpp/feature_pipeline.cpp` / `include/mm/feature_pipeline.hpp`
    - Maintain O(1) rolling accumulators (running sum and sum-of-squares) for OFI and realized volatility windows
    - Implement `update()`: compute `ofi_l1`, `ofi_l5`, `inventory_skew`, `inventory_urgency`, `vol_regime`, `micro_price_offset`, `spread_ticks`, `session_progress`
    - Apply NaN guards and 5-sigma clamp (`|obs[i]| ≤ 5.0`) in normalization
    - Implement zero-copy `observe(MarketFeatures& out)` writing into caller-supplied buffer
    - Implement `reset()` clearing all window state
    - _Requirements: 2.1–2.13, 10.4, 11.4_

  - [x] 4.2 Write property test for feature finiteness and bounds (rapidcheck)
    - **Property 6: Feature Finiteness and Bounds** — for any valid `BookSnapshot` and net position in `[-max_position, max_position]`, all 8 `MarketFeatures` values are finite and `|obs[i]| ≤ 5.0`
    - **Validates: Requirements 2.1, 2.9, 2.10**

  - [x] 4.3 Write property test for OFI formula correctness (rapidcheck)
    - **Property 7: OFI Formula Correctness** — for any snapshot with non-zero bid/ask quantities, `ofi_l1` and `ofi_l5` equal `(bid_qty_sum - ask_qty_sum) / (bid_qty_sum + ask_qty_sum)` and lie in `[-1, 1]`
    - **Validates: Requirements 2.2**

  - [x] 4.4 Write property test for inventory normalization invariants (rapidcheck)
    - **Property 8: Inventory Normalization Invariants** — for any `|p| ≤ max_position`, `inventory_skew = p / max_position ∈ [-1,1]` and `inventory_urgency = |p| / max_position ∈ [0,1]`
    - **Validates: Requirements 2.3, 2.4**

  - [x] 4.5 Write unit tests for feature pipeline (Catch2)
    - Feed synthetic snapshots with known values; verify OFI range, vol_regime z-score, NaN guard behavior; benchmark `update()` under 100 ns
    - _Requirements: 2.1–2.13, 10.4_


- [x] 5. Checkpoint — core C++ components
  - Ensure all Catch2 and rapidcheck tests pass for tasks 2–4; benchmark `LobEngine::process()` < 1000 ns and `FeaturePipeline::update()` < 100 ns; ask the user if any questions arise.

- [x] 6. Trajectory Buffer and GAE computation
  - [x] 6.1 Implement `TrajectoryBuffer` class in `src/python/trajectory_buffer.py`
    - Pre-allocate contiguous NumPy arrays of configurable `capacity` (must be power-of-two)
    - Implement `push(transition)` rejecting non-finite `log_prob` values with error log
    - Implement `compute_advantages(gamma, lam)` iterating in reverse chronological order for GAE-λ
    - Bootstrap `next_value = 0` on `done = True` steps; use `last_value` for non-terminal final step
    - Normalize advantages: `(adv - adv.mean()) / (adv.std() + 1e-8)` and assert all values finite
    - Expose zero-copy tensor views `obs_tensor()`, `action_tensor()`, `advantage_tensor()`, `return_tensor()`
    - Implement `clear()` resetting index without releasing memory
    - _Requirements: 4.1–4.10, 11.5_

  - [x] 6.2 Write property test for GAE finiteness and normalization (hypothesis)
    - **Property 12: GAE Finiteness and Normalization** — for any sequence of finite rewards and value estimates, `compute_advantages()` produces finite advantages with mean ≈ 0 and std ≈ 1
    - **Validates: Requirements 4.3, 4.6, 4.7**

  - [x] 6.3 Write property test for non-finite log-prob rejection (hypothesis)
    - **Property 13: Non-Finite Log-Prob Rejection** — any `Transition` with `log_prob` = NaN / ±Inf is rejected and leaves buffer state unchanged
    - **Validates: Requirements 4.10, 11.5**

  - [x] 6.4 Write unit tests for trajectory buffer (pytest)
    - Fill to capacity; verify GAE output is finite and normalized; test single-step episode; test `clear()` preserves capacity
    - _Requirements: 4.1–4.10_


- [x] 7. PPO Agent (Python/PyTorch)
  - [x] 7.1 Implement `ActorCritic` network in `src/python/ppo_agent.py`
    - 3-layer MLP shared trunk (8 → 256 → 256) with `nn.Tanh` activations
    - Actor head: linear → `mean` (shape 3) + `log_std` parameter (shape 3, clamped to `[-4, 2]`)
    - Critic head: linear → scalar `V(s)`
    - Implement `forward(obs) → (mean, log_std, value)`
    - Implement `act(obs) → (action, log_prob)` sampling from `Normal(mean, exp(log_std))` and computing log-prob
    - _Requirements: 3.1, 3.2, 3.8_

  - [x] 7.2 Implement PPO update logic in `PPOAgent.train_step()` in `src/python/ppo_agent.py`
    - Clipped surrogate: `L_CLIP = mean(min(r·A, clip(r, 1−ε, 1+ε)·A))`
    - Value loss: `0.5 × MSE(V(s), returns)`
    - Entropy bonus: `−entropy_coef × mean(entropy)` to encourage exploration
    - Gradient clipping: `clip_grad_norm_(params, max_norm=0.5)`
    - Adam optimizer with configurable `lr` and `eps`
    - Assert `loss.is_finite()` before backprop
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 7.3 Implement TorchScript export in `PPOAgent.export()` in `src/python/ppo_agent.py`
    - Export actor network only via `torch.jit.script(agent.actor)`
    - Save to configured model path; compute and persist SHA-256 checksum alongside the `.pt` file
    - _Requirements: 3.10, 6.9, 11.8_

  - [x] 7.4 Write property test for PPO clipped surrogate correctness (hypothesis)
    - **Property 9: PPO Clipped Surrogate Correctness** — for any ratio `r` and advantage `A`, the clipped surrogate = `min(r·A, clip(r, 1−ε, 1+ε)·A)` and clipped ratio lies in `[1−ε, 1+ε]`
    - **Validates: Requirements 3.3**

  - [x] 7.5 Write property test for gradient norm bounded after clipping (hypothesis)
    - **Property 10: Gradient Norm Bounded After Clipping** — after each PPO update with `max_norm=0.5`, L2 norm of all gradients ≤ 0.5
    - **Validates: Requirements 3.6**

  - [x] 7.6 Write unit tests for actor-critic forward pass (pytest)
    - Verify output shapes `(3,)` for action/log_std, scalar for value; test log_prob finiteness; test entropy > 0 with random obs
    - _Requirements: 3.1, 3.2, 3.8_


- [x] 8. Reward signal and LobEnv training environment
  - [x] 8.1 Implement `RewardShaper` in `src/python/reward_shaper.py`
    - Compute `realized_pnl` (mark-to-market delta in ticks on fill events)
    - Compute `inventory_penalty = −γ × |net_position| / max_position` (always ≤ 0)
    - Compute `spread_capture` (half-spread earned on completed round-trip fills, always ≥ 0)
    - Return `RewardSignal` dataclass and `total()` sum
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 8.2 Implement `LobEnv` Gym wrapper in `src/python/mm_env.py` using pybind11 bindings
    - `observation_space = gym.spaces.Box(shape=(8,), dtype=np.float32)`
    - `action_space = gym.spaces.Box(shape=(3,), dtype=np.float32)` for `(delta_bid, delta_ask, size_frac)`
    - `reset(seed=None)` — re-seed RNG, reset C++ LOB and feature pipeline, reload tick data slice
    - `step(action)` — apply action via `QuoteGenerator`, advance one tick event, return `(obs, reward, done, info)`
    - `done = True` on end-of-session or position-limit breach
    - Load NSE L2 tick data from Parquet via pandas/pyarrow; support ≥ 6 months of tick history
    - `render(mode="human")` — print book state, net position, cumulative PnL to stdout
    - Support deterministic replay via RNG seeding
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.1, 8.3, 8.4, 8.5, 8.6, 8.7, 13.6_

  - [x] 8.3 Write property test for reward component signs (hypothesis)
    - **Property 21: Reward Component Signs** — for any training step, `spread_capture ≥ 0` and `inventory_penalty ≤ 0`
    - **Validates: Requirements 8.2**

  - [x] 8.4 Write property test for reward formula consistency (hypothesis)
    - **Property 22: Reward Formula Consistency** — `total_reward = realized_pnl + inventory_penalty + spread_capture` where `inventory_penalty = −γ × |net_position| / max_position`
    - **Validates: Requirements 8.1, 8.3**

  - [x] 8.5 Write property test for deterministic replay (hypothesis)
    - **Property 23: Deterministic Replay** — two `LobEnv.reset(seed=s)` calls followed by identical actions produce byte-identical obs/reward/done sequences
    - **Validates: Requirements 8.5**

  - [x] 8.6 Write unit tests for LobEnv (pytest)
    - Test `observation_space` shape/dtype, `action_space` bounds, `done` signal on position breach, `render()` output
    - _Requirements: 6.1–6.4, 8.4, 8.7_


- [x] 9. pybind11 bridge and input validation
  - [x] 9.1 Implement pybind11 bindings in `src/cpp/mm_bindings.cpp`
    - Expose `LobEngine` and `FeaturePipeline` to Python as `mm.LobEngine` and `mm.FeaturePipeline`
    - Implement `step()` handler calling `lob.process()`, `features.update()`, returning observation numpy array
    - Implement `reset()` calling `lob.reset()` and `features.reset()`
    - Add boundary validation layer: screen all incoming numpy arrays for NaN, shape mismatches, and out-of-range values; raise `ValueError` on violation
    - _Requirements: 6.1, 6.2, 6.10, 11.6_

  - [x] 9.2 Write property test for pybind11 boundary NaN rejection (hypothesis)
    - **Property 24: pybind11 Boundary NaN Rejection** — any observation or action array containing NaN or out-of-range value raises an exception and is not forwarded to C++
    - **Validates: Requirements 6.10, 11.6**

  - [x] 9.3 Write integration tests for pybind11 bridge (pytest)
    - Call `env.step()` 10,000 times via pybind11; verify observation shapes, reward finiteness, C++ LOB state consistency against a reference Python LOB
    - _Requirements: 6.1, 6.2, 6.3, 6.4_


- [x] 10. Checkpoint — Python training harness
  - Ensure all pytest and hypothesis tests pass for tasks 6–9; verify `LobEnv` resets and steps cleanly for 10,000 steps without assertion errors; ask the user if any questions arise.

- [x] 11. PolicyInference C++ wrapper (TorchScript)
  - [x] 11.1 Implement `PolicyInference` class in `src/cpp/policy_inference.cpp` / `include/mm/policy_inference.hpp`
    - Load TorchScript `.pt` file via `torch::jit::load()`; verify SHA-256 checksum against stored reference before loading
    - Implement `act(const float* obs)` with `torch::NoGradGuard`; wrap entire body in try-catch returning safe fallback `{delta_bid=2.0f, delta_ask=2.0f, size_frac=0.1f}` on any exception; log offending observation and exception message
    - Implement `reload(const std::string& new_model_path)` with atomic swap via `std::atomic<std::shared_ptr<torch::jit::Module>>`; verify checksum before swap; return `false` and retain old model on failure
    - Benchmark `act()` with 8→256→256→3 MLP: assert ≤ 400 ns per call with `torch::NoGradGuard`
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 6.9, 9.2, 9.7, 11.8, 13.5_

  - [x] 11.2 Write property test for TorchScript round-trip consistency (hypothesis + Catch2)
    - **Property 11: TorchScript Round-Trip Consistency** — for any obs `(8,)` float32, C++ `PolicyInference::act(obs)` output matches Python `agent.actor(obs)` to within 1e-5 absolute tolerance on every element
    - **Validates: Requirements 3.10, 6.6**

  - [x] 11.3 Write property test for safe fallback on inference error (rapidcheck)
    - **Property 18: Safe Fallback on Inference Error** — any observation with NaN or causing libtorch exception returns `{delta_bid=2.0f, delta_ask=2.0f, size_frac=0.1f}` without propagating exception
    - **Validates: Requirements 9.2**

  - [x] 11.4 Write unit tests for PolicyInference (Catch2)
    - Test valid model load and `act()` output shapes; test checksum mismatch rejection; test `reload()` atomic swap; test fallback on NaN input; benchmark `act()` latency
    - _Requirements: 6.5–6.9, 10.5_


- [x] 12. Alpha signal interface and Quote Generator
  - [x] 12.1 Implement `IAlphaSignal` interface and `QuoteGenerator` class in `src/cpp/quote_generator.cpp` / `include/mm/quote_generator.hpp`
    - `IAlphaSignal`: pure virtual `id() → uint32_t` and `adjust(const BookSnapshot&) → pair<float,float>`
    - `QuoteGenerator::generate()`: call `policy.act(obs)`, translate `(delta_bid, delta_ask)` to integer tick prices via `mid_price ± round(delta × tick_size)`, enforce spread ≥ 1 tick
    - Scale `size_frac` to `bid_qty`/`ask_qty` in `[1, max_order_size]`
    - Apply inventory halt rule: zero out offending-side qty and widen opposite spread by 2× when `|net_position| ≥ max_inventory × 0.9`
    - Aggregate alpha signal adjustments atomically using a lock-free signal list (`std::atomic` + hazard pointers or `std::shared_ptr` swap)
    - Assign `valid_until_ns` TTL to each `QuoteParams`
    - Implement `attach_signal()` and `detach_signal()` for runtime hot-swap; `detach_signal` with unknown ID is a no-op
    - Measure wall-clock via `rdtsc`; assert completion within `latency_budget_ns`
    - _Requirements: 5.1–5.9, 9.5, 10.1, 12.1–12.6_

  - [x] 12.2 Write property test for minimum spread enforcement (rapidcheck)
    - **Property 14: Minimum Spread Enforcement** — for any policy output `(delta_bid, delta_ask)` and any `mid_price > 0`, `QuoteParams.ask_price − bid_price ≥ tick_size`
    - **Validates: Requirements 5.2, 5.3**

  - [x] 12.3 Write property test for inventory halt rule (rapidcheck)
    - **Property 15: Inventory Halt Rule** — for any `|p| ≥ max_inventory × 0.9`, `bid_qty = 0` when `p > 0` and `ask_qty = 0` when `p < 0`
    - **Validates: Requirements 5.5, 9.5**

  - [x] 12.4 Write property test for quote quantity bounds (rapidcheck)
    - **Property 16: Quote Quantity Bounds** — for any `size_frac ∈ (0,1]` and `|p| < max_inventory × 0.9`, `bid_qty` and `ask_qty` lie in `[1, max_order_size]`
    - **Validates: Requirements 5.4**

  - [x] 12.5 Write property test for alpha signal aggregation (rapidcheck)
    - **Property 17: Alpha Signal Aggregation** — the generated `(bid_price, ask_price)` reflects the sum of all attached signal adjustments; after `detach_signal(id)`, that signal's adjustments no longer appear
    - **Validates: Requirements 5.6, 5.8, 12.2, 12.3**

  - [x] 12.6 Write unit tests for QuoteGenerator (Catch2)
    - Mock `PolicyInference`; test spread enforcement, inventory clipping, alpha signal attach/detach, TTL assignment, latency under 800 ns
    - _Requirements: 5.1–5.9, 10.1, 12.1–12.6_


- [x] 13. Tick Replayer and SPSC queue
  - [x] 13.1 Implement `TickReplayer` in `src/cpp/tick_replayer.cpp` / `include/mm/tick_replayer.hpp`
    - Load NSE L2 tick data from checksummed Parquet archive; verify file checksum before loading; reject mismatched files
    - Deliver events in chronological order via a lock-free SPSC queue (single-producer single-consumer ring buffer) with no blocking on the strategy thread
    - Detect gaps > 1 second between consecutive events and emit a gap notification to the strategy
    - _Requirements: 8.6, 9.6, 10.8, 11.7_

  - [x] 13.2 Write unit tests for TickReplayer (Catch2)
    - Test chronological ordering, checksum rejection, gap detection, lock-free queue non-blocking delivery
    - _Requirements: 10.8, 11.7_


- [x] 14. Strategy Framework orchestration
  - [x] 14.1 Implement `StrategyFramework` class in `src/cpp/strategy_framework.cpp` / `include/mm/strategy_framework.hpp`
    - Accept `StrategyConfig` at construction; initialize all sub-components from config
    - `run()`: blocking event loop on a single CPU-affinity-pinned thread using `std::pmr::monotonic_buffer_resource` arena (reset after each event — no heap alloc on hot path)
    - Event loop: pull from SPSC queue → `lob.process()` → if changed: `features.update()` → `features.observe()` → `quoter.generate()` → `gateway.cancel_stale()` → `gateway.submit()`; skip quote if LOB unchanged
    - Cancel all live orders via gateway on data gap > 1 second; resume only when book has ≥ 3 levels
    - Enforce `|gateway.net_position()| ≤ max_position` hard invariant on every fill event
    - Apply hot-reload model swap between quote-generation cycles (no latency insertion on hot path)
    - `reload_model()`, `attach_signal()`, `detach_signal()` callable from management thread without blocking event loop
    - Shutdown: cancel all open orders via gateway before terminating event loop
    - _Requirements: 7.1–7.10, 9.3, 9.4, 9.6, 9.8, 10.7_

  - [x] 14.2 Implement `PerfStats` lock-free performance counters and structured logging
    - `PerfStats` with `std::atomic` fields: `events_processed`, `quotes_generated`, `net_pnl_ticks`, `max_latency_ns`, `p99_latency_ns`
    - Record `rdtsc`-based latency for every quote generation cycle; update `max_latency_ns` and `p99_latency_ns` atomically
    - Emit structured log entries: crossed-book rejections (timestamp + event + best bid/ask), latency breaches (cycle ts + measured + budget), fallback inference activations
    - Auto-switch to `SimplifiedPolicyInference` fallback when persistent breaches exceed 1% of quotes; emit operational alert
    - _Requirements: 7.7, 9.3, 9.4, 13.1–13.5_

  - [x] 14.3 Write property test for one quote per state-changing event (rapidcheck)
    - **Property 19: One Quote Per State-Changing Event** — total quote generation cycles equals the number of events for which `LobEngine::process()` returned `true`
    - **Validates: Requirements 7.2, 7.3**

  - [x] 14.4 Write property test for inventory bound invariant (rapidcheck)
    - **Property 20: Inventory Bound Invariant** — for any sequence of fill events, `|gateway.net_position()| ≤ max_position` after every fill
    - **Validates: Requirements 7.5**

  - [x] 14.5 Write unit tests for StrategyFramework (Catch2)
    - Test event loop with synthetic tick stream; verify quote skipping on no-change events; verify stale quote cancellation; test `reload_model()` from secondary thread; benchmark P99 latency over 10,000 events ≤ 800 ns
    - _Requirements: 7.1–7.10, 10.3_


- [x] 15. Checkpoint — full C++ core integration
  - Ensure all Catch2 and rapidcheck tests pass for tasks 11–14; run P99 latency benchmark over 10,000 events and confirm ≤ 800 ns; confirm no heap allocations on hot path via heap profiler; ask the user if any questions arise.

- [x] 16. Full training pipeline wiring
  - [x] 16.1 Implement `mm_train.py` training entry point in `src/python/mm_train.py`
    - Instantiate `LobEnv`, `PPOAgent`, `TrajectoryBuffer`; run training loop for configurable `num_episodes` and `rollout_steps`
    - Collect rollout, call `buffer.compute_advantages()`, run `agent.train_step()` for K epochs with mini-batches
    - Export trained actor to TorchScript via `agent.export(model_path)` after training completes
    - Log per-episode Sharpe ratio, PnL drawdown vs. fixed-spread baseline to stdout/file
    - _Requirements: 3.1–3.11, 8.1–8.7_

  - [x] 16.2 Write integration tests for end-to-end training pipeline (pytest)
    - Train for 5 episodes on a synthetic data fixture; verify trajectory buffer fills and clears correctly; export TorchScript; reload in `PolicyInference`; verify `act()` output matches Python actor to within 1e-5
    - _Requirements: 3.10, 6.5, 6.6_

- [x] 17. End-to-end backtest integration
  - [x] 17.1 Implement `mm_backtest.cpp` backtest runner in `src/cpp/mm_backtest.cpp`
    - Wire `StrategyFramework` with real Parquet tick data (1-week slice for integration test)
    - Assert `net_pnl_ticks` is tracked correctly; no crashes over full replay; `max_latency_ns < 2000 ns` (2× budget margin)
    - Print `PerfStats` summary at end of run
    - _Requirements: 7.1–7.10, 10.1–10.8_

  - [x] 17.2 Write integration test for full backtest (Catch2)
    - Replay 1 week of NSE L2 data; assert non-crash completion, `max_latency_ns < 2000`, `net_pnl_ticks` finite and tracked, no inventory violations
    - _Requirements: 7.1–7.10, 10.1, 10.3_

- [x] 18. Final checkpoint — ensure all tests pass
  - Run full test suite (Catch2, rapidcheck, pytest, hypothesis); confirm all 25 correctness properties have associated passing tests; confirm backtest runs clean end-to-end; ask the user if any questions arise.


## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; all 25 correctness properties have corresponding property test sub-tasks
- C++ property tests use **rapidcheck**; Python property tests use **hypothesis**
- C++ unit tests use **Catch2 v3**; Python unit/integration tests use **pytest**
- Checkpoints are placed after foundational layers are complete to validate before building higher layers
- The design uses pseudocode for algorithm descriptions but all C++ interfaces are already specified — C++ for the core, Python/PyTorch for training
- All C++ hot-path code must avoid heap allocation; use `std::pmr::monotonic_buffer_resource` for any temporary per-tick memory
- Property test sub-tasks are numbered to match the corresponding property in the design document's "Correctness Properties" section
- The `SimplifiedPolicyInference` fallback (requirement 9.4) should be implemented as a simple fixed-spread lookup table wired into the `PolicyInference` fallback path


## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "2.2"] },
    { "id": 1, "tasks": ["2.3", "3.1", "6.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "4.1", "6.2", "6.3", "6.4"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "4.5", "7.1"] },
    { "id": 4, "tasks": ["7.2", "8.1", "9.1"] },
    { "id": 5, "tasks": ["7.3", "7.4", "7.5", "7.6", "8.2"] },
    { "id": 6, "tasks": ["8.3", "8.4", "8.5", "8.6", "9.2", "9.3", "11.1", "13.1"] },
    { "id": 7, "tasks": ["11.2", "11.3", "11.4", "12.1", "13.2"] },
    { "id": 8, "tasks": ["12.2", "12.3", "12.4", "12.5", "12.6", "14.1"] },
    { "id": 9, "tasks": ["14.2", "14.3", "14.4", "14.5"] },
    { "id": 10, "tasks": ["16.1"] },
    { "id": 11, "tasks": ["16.2", "17.1"] },
    { "id": 12, "tasks": ["17.2"] }
  ]
}
```
