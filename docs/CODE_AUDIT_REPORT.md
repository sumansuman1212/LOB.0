# LOB.0 — Baseline Code Audit & Engineering Hardening Report

**Project:** LOB.0 — RL-Driven Adaptive Market Making Engine  
**Audit Stage:** Phase 0 — Baseline Architecture & Security Audit  
**Date:** 2026-11-07  
**Commit Baseline:** `26feda4` on branch `main`  
**Auditor:** Senior C++ Systems Engineer & Quantitative Software Engineer  

---

## 1. Executive Summary & Baseline Metrics

The **LOB.0** codebase has been comprehensively inspected across its C++17 execution engine, Python/PyTorch reinforcement learning harness, PyBind11 bindings, and automated test suites.

### Baseline Status Matrix

| Component | Test Suite / Benchmark | Result | Notes |
| :--- | :--- | :---: | :--- |
| **Python RL Suite** | `python -m pytest tests/ -v` | **37 / 37 PASS** | 7.41s execution time; 0 failures |
| **C++ Core Test Suite** | `./build/tests/mm_tests` | **22 / 22 PASS** | 1,761 assertions executed in Linux Codespaces |
| **Native Neural Inference** | `./build/src/cpp/verify_native_inference` | **PASS** | Validates `models/ppo_actor.pt` + SHA-256 integrity |
| **Git Working Tree** | `git status` | **CLEAN** | All files tracked cleanly on branch `main` |
| **Model Checkpoint** | `Get-FileHash models/ppo_actor.pt` | **VERIFIED** | SHA-256: `88bad3b41954124ad1da7f1b7f19e6f49e908cd6a8deac221f3f7b5d846d939a` |

---

## 2. Identified Engineering Weaknesses & Risk Classifications

Our static analysis and manual audit have revealed several genuine engineering weaknesses and edge-case vulnerabilities categorized by severity:

### 2.1 Critical / High Severity Issues

#### [CRIT-01] Signed Integer Overflow & Sentinel Contamination in `FeaturePipeline::update()`
- **Source File:** `src/cpp/feature_pipeline.cpp` (Lines 237–298)
- **Vulnerability:**
  `LobEngine` uses `kEmptyAsk = std::numeric_limits<int64_t>::max()` when the ask side of the book is empty.
  In `FeaturePipeline::update()`:
  ```cpp
  int64_t mid_price_ticks = snap.best_bid > 0 && snap.best_ask > 0
                              ? (snap.best_bid + snap.best_ask) / 2
                              : 0;
  ```
  When the ask side is empty, `snap.best_ask` is `INT64_MAX` (which is `> 0`). When `snap.best_bid > 0`, the expression `(snap.best_bid + snap.best_ask)` triggers **signed integer overflow**, which is **Undefined Behavior (UB)** in C++.
  Furthermore, `spread_ticks` calculation:
  ```cpp
  if (snap.best_ask > 0 && snap.best_bid > 0) {
      int64_t raw_spread = snap.best_ask - snap.best_bid;
      if (raw_spread > 0) {
          spread_ticks = static_cast<float>(std::log1p(static_cast<double>(raw_spread) - 1.0));
      }
  }
  ```
  If the ask side is empty, `raw_spread` becomes `INT64_MAX - best_bid` ($\approx 9.22 \times 10^{18}$ ticks), computing an erroneous astronomical spread feature that pollutes the neural network observation.
- **Remediation:** Introduce explicit validity checks for empty/one-sided books:
  `bool has_bid = snap.best_bid > 0 && snap.best_bid != std::numeric_limits<int64_t>::min();`
  `bool has_ask = snap.best_ask > 0 && snap.best_ask != std::numeric_limits<int64_t>::max();`
  Only compute mid-price, micro-price, and spread when both sides are valid (`has_bid && has_ask`).

#### [CRIT-02] SPSC Queue `size()` Wraparound Arithmetic Bug
- **Source File:** `include/mm/tick_replayer.hpp` (Lines 70–74)
- **Vulnerability:**
  ```cpp
  size_t size() const noexcept {
      const size_t h = head_.load(std::memory_order_relaxed);
      const size_t t = tail_.load(std::memory_order_relaxed);
      return (t >= h) ? (t - h) : 0;
  }
  ```
  When the unsigned `tail_` index wraps around `SIZE_MAX`, `t < h` temporarily. The ternary condition `(t >= h) ? (t - h) : 0` evaluates to `false` and returns `0`, falsely reporting the queue as empty even when items are buffered.
- **Remediation:** In unsigned modular arithmetic, `return (t - h);` without the conditional is mathematically correct across index wraparound boundaries.

#### [CRIT-03] Single-Producer Invariant Violation in `TickReplayer::next_event()`
- **Source File:** `src/cpp/tick_replayer.cpp` (Lines 58–70)
- **Vulnerability:**
  ```cpp
  bool TickReplayer::next_event(OrderEvent& out_event) noexcept {
      if (queue_.pop(out_event)) {
          return true;
      }
      if (cursor_ < events_.size()) {
          pump_to_queue(256);
          return queue_.pop(out_event);
      }
      return false;
  }
  ```
  If an ingestion thread calls `pump_to_queue()` while a strategy consumer thread calls `next_event()`, both threads invoke `queue_.push()` concurrently, violating the **Single-Producer** invariant of `SPSCQueue`. Furthermore, `cursor_` and `last_timestamp_ns_` are non-atomic, resulting in a data race.
- **Remediation:** Clearly segregate consumer reading (`queue_.pop()`) from producer ingestion (`pump_to_queue()`), or provide an explicit synchronous mode for single-threaded backtesting.

---

### 2.2 Moderate Severity Issues

#### [MED-01] Negative Unsigned Integer Wrap in `dict_to_order_event` (`mm_bindings.cpp`)
- **Source File:** `src/cpp/mm_bindings.cpp` (Line 115)
- **Vulnerability:**
  `ev.qty = d["qty"].cast<uint32_t>();`
  If a Python dictionary passes a negative quantity (e.g., `qty: -5`), casting directly to `uint32_t` silently wraps around to `4294967291` without triggering a validation error, corrupting book level aggregations.
- **Remediation:** Cast to `int64_t` first, validate `qty > 0`, then assign to `uint32_t`.

#### [MED-02] Financial Accounting Terminology Discrepancy in `OrderGateway`
- **Source File:** `src/cpp/strategy_framework.cpp` (Lines 36–43)
- **Vulnerability:**
  `OrderGateway::realized_pnl_` accumulates cash flow:
  `realized_pnl_ -= (price * qty)` on buy, `realized_pnl_ += (price * qty)` on sell.
  This member variable tracks cumulative cash flow, not closed-position realized PnL. While holding an open position, it appears negative due to inventory purchase cost.
- **Remediation:** Rename/clarify to `cash_balance_` and surface true Mark-to-Market Total PnL (`Cash + NetPosition * MidPrice`).

---

## 3. Proposed Remediation Plan

We propose executing the improvements across 4 reviewable stages:

```
Stage 1: Numerical Safety & Sentinel Hardening (CRIT-01, MED-01)
         - Fix FeaturePipeline empty-book sentinel guards.
         - Fix dict_to_order_event negative qty integer wrap.
         - Add unit tests for empty and one-sided books.

Stage 2: Concurrency & Lock-Free SPSC Hardening (CRIT-02, CRIT-03)
         - Fix SPSCQueue size() unsigned modular difference.
         - Separate single-threaded synchronous replay from threaded SPSC ingestion.
         - Add SPSC queue capacity and wraparound stress tests.

Stage 3: Financial Accounting & Metric Precision (MED-02)
         - Align OrderGateway cash ledger and MTM equity calculation.
         - Add multi-fill position accounting assertions.

Stage 4: Edge-Case Test Expansion (Phase 5 Requirements)
         - Add 20 comprehensive edge-case tests in C++ Catch2 test suite.
         - Verify all tests pass cleanly in both Python and C++.
```

---

## 4. Codespaces Action Items

**Do you need to do anything in Codespaces right now?**
- **NO.** The Codespace can remain suspended/offline while we review the audit and prepare the code fixes locally.
- Once the code hardening is implemented and committed to GitHub, you will only need to open Codespaces to run the final verification commands:
  ```bash
  git pull origin main
  cmake --build build --target mm_tests mm_backtest verify_native_inference
  ./build/tests/mm_tests
  ./build/src/cpp/verify_native_inference
  ```
