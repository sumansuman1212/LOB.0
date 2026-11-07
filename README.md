# LOB.0 — Reinforcement Learning-Driven Adaptive Market Making Engine

[![LOB.0 CI](https://github.com/sumansuman1212/LOB.0/actions/workflows/ci.yml/badge.svg)](https://github.com/sumansuman1212/LOB.0/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](include/mm/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](src/python/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](src/python/ppo_agent.py)

> **High-performance C++17 limit order book execution core coupled with a PyTorch PPO reinforcement learning harness for adaptive market-making, inventory risk control, and low-latency quote generation.**

---

## 1. Project Overview

### What is Market Making?
A market maker acts as a continuous liquidity provider in financial markets, simultaneously quoting bid (buy) and ask (sell) prices. By capturing the bid-ask spread, the market maker generates revenue while absorbing two fundamental risks:
1. **Inventory Risk:** Accumulating an unbalanced long or short inventory that exposes capital to unfavorable price drift.
2. **Adverse Selection:** Trading against informed market participants who possess superior short-term information, leading to toxic fills immediately prior to adverse price movements.

### The Problem Explored
Classical analytical market-making frameworks—most notably the Avellaneda-Stoikov model and Guéant-Tapia-Manziadi extensions—derive optimal reservation prices and spread half-widths under restrictive closed-form assumptions: continuous Brownian motion for mid-price dynamics and constant, stationary Poisson arrival intensities.

In modern electronic limit order books (LOBs), price formation is non-linear and non-stationary. High-frequency signals such as **order flow imbalance (OFI)**, micro-price deviations, depth imbalances, and short-term volatility bursts contain critical short-horizon alpha. **LOB.0** investigates how an **Actor-Critic Reinforcement Learning agent (Proximal Policy Optimization)** can learn continuous quote-skewing policies directly from real-time microstructure features while strictly satisfying hard exchange rules, tick discretization, minimum spread invariants, and inventory limits.

### Why Modern C++?
Ultra-low-latency electronic trading demands sub-microsecond determinism:
- **Zero Heap Allocations on Hot Paths:** Pre-allocated memory, circular buffers, and in-place state mutation ensure predictable tick-to-trade latency.
- **Hardware Cache Alignment:** `PriceLevel` structs are aligned to 64-byte boundaries (`alignas(64)`) to prevent cacheline straddling. `OrderEvent` structs are packed to exactly 32 bytes (`sizeof(OrderEvent) == 32`).
- **Lock-Free Concurrency:** Inter-thread communication between data replay/ingestion and strategy execution is mediated by an atomic, lock-free Single-Producer Single-Consumer (`SPSCQueue`) ring buffer.

### Why Reinforcement Learning (PPO)?
Market making is a constrained, multi-objective Markov Decision Process (MDP): balancing spread capture against inventory penalization and execution probability. PPO provides:
- **Continuous Action Space Exploration:** Outputting continuous Gaussian distributions over bid spreads, ask spreads, and lot size scalers.
- **Clipped Surrogate Objective:** Preventing catastrophically destabilizing policy parameter updates during training.
- **Generalized Advantage Estimation (GAE-$\lambda$):** Managing the bias-variance trade-off across multi-step trading horizons.

### How C++ and Python Communicate
The system operates as a hybrid architecture designed for both high-throughput research and low-latency execution:
1. **Offline Training (Python/PyTorch):** High-throughput vectorized simulation in Gymnasium (`LobEnv`) feeds batched rollouts to `PPOAgent`, computing GAE advantages and updating neural policy weights.
2. **Model Freezing & Serialization:** Trained actor networks are exported to optimized TorchScript computation graphs (`models/ppo_actor.pt`) accompanied by a cryptographic SHA-256 checksum (`models/ppo_actor.pt.sha256`).
3. **Online Native Inference (C++17 / LibTorch):** The C++ strategy framework loads the TorchScript model using LibTorch, cryptographically verifies its hash at startup, and performs zero-copy native tensor inference within microsecond latency budgets.
4. **Analytical Fallback Circuit Breaker:** If LibTorch is absent, an observation is non-finite, or inference latency breaches the configured budget, execution automatically transitions without interruption to an inventory-aware heuristic policy (`SimplifiedPolicyInference`).
5. **Bidirectional Bindings (PyBind11):** High-fidelity C++ engine types (`LobEngine`, `FeaturePipeline`, `OrderEvent`, `BookSnapshot`) are exposed directly to Python for zero-copy NumPy feature extraction and parity testing.

---

## 2. Architecture Overview

```
                                  LOB.0 END-TO-END PIPELINE
                                  
  Market Events (Tick Data / Replayer)
       │
       ▼
  C++ Limit Order Book Engine (LobEngine)
       │  • Price-time priority matching (bids descending, asks ascending)
       │  • O(1) order hash-map lookup; 64-byte aligned PriceLevels
       │  • Invariants: crossed-book rejection, monotonic sequencing
       ▼
  Market Feature Pipeline (FeaturePipeline)
       │  • 8-dimensional normalized observation vector x ∈ ℝ⁸
       │  • OFI L1 & L5, Inventory Skew & Urgency, Realized Vol, Micro-Price
       │  • Zero-heap hot path; empty-book sentinel protection; [-5, 5] clamping
       ▼
  Python / C++ Interface (PyBind11 / TorchScript Bridge)
       │  • Offline: Gym environment (LobEnv) & PPO training harness
       │  • Online: Serialized actor graph (models/ppo_actor.pt) + SHA-256 verify
       ▼
  PPO Policy Inference (PolicyInference / Analytical Fallback)
       │  • Native LibTorch forward pass: obs(8) ──► action(3)
       │  • Zero-overhead fallback heuristic on latency breach or NaN detection
       ▼
  Quote Generator (QuoteGenerator)
       │  • Discretizes continuous action offsets to integer exchange ticks
       │  • Enforces minimum spread invariant (P_ask - P_bid ≥ tick_size)
       │  • Dynamic quote sizing: clamp(⌊size_frac · max_size⌉, 1, max_size)
       ▼
  Risk Management & Execution Gateway (OrderGateway)
       │  • 90% Inventory Halt soft limit: halts quote on accumulating side
       │  • Hard inventory limits (|q| ≤ q_max)
       │  • Quote TTL expiration (100 ms) and stale order cancellation
       ▼
  Order Execution & Simulated Matching
       │  • FIFO passive queue execution & aggressor fill simulation
       │  • Fill callbacks update cash flow, volume, and net inventory
       ▼
  PnL Accounting & Performance Analytics (Financial Ledger)
          • Cash Balance:  Cash_{t+1} = Cash_t - P_fill · Δq
          • Equity Value:  Equity_t = Cash_t + Inventory_t · P_mid,t
          • Total PnL:     PnL_t = Realized PnL_t + Unrealized PnL_t
          • Separation:    Surrogate RL reward shaping ≠ Financial accounting
```

---

## 3. Technical Implementation

### 3.1 C++17 Limit Order Book (`src/cpp/lob_engine.cpp`)
- **Price-Time Priority:** Dual red-black trees (`std::map<int64_t, PriceLevel, std::greater<int64_t>>` for bids; `std::map<int64_t, PriceLevel, std::less<int64_t>>` for asks).
- **Fast Order Cancellation & Modification:** Hash map (`std::unordered_map<uint64_t, OrderEntry> orders_`) provides $O(1)$ order lookup.
- **Memory Footprint & Alignment:** 
  - `PriceLevel` is padded to 64 bytes (`alignas(64)`) to prevent cross-cacheline invalidations during level updates.
  - `OrderEvent` is packed into 32 bytes (`sizeof(OrderEvent) == 32`).
  - `BookSnapshot` provides a fixed-size `std::array<PriceLevel, 10>` for bids and asks, enabling zero-allocation cache-coherent snapshot transfers.
- **Transaction Safety & Invariants:**
  - Rejects duplicate order IDs on Add.
  - Rejects non-monotonic backwards-jumping timestamps.
  - Rejects non-positive prices and zero-quantity events.
  - Rejects crossed-book events with structured stderr alerts without altering book state.

### 3.2 Microstructure Feature Pipeline (`src/cpp/feature_pipeline.cpp`)
Extracts an 8-dimensional normalized observation vector $\mathbf{x} \in \mathbb{R}^8$ with zero dynamic memory allocation on the hot path:

| Index | Feature Name | Mathematical Formulation | Nominal Range | Description |
| :---: | :--- | :--- | :---: | :--- |
| `0` | **$\text{OFI}_{L1}$** | $\frac{Q_{\text{bid},0} - Q_{\text{ask},0}}{Q_{\text{bid},0} + Q_{\text{ask},0}}$ | $[-1, 1]$ | Immediate Level-1 order flow imbalance |
| `1` | **$\text{OFI}_{L5}$** | $\frac{\sum_{i=0}^4 Q_{\text{bid},i} - \sum_{i=0}^4 Q_{\text{ask},i}}{\sum_{i=0}^4 Q_{\text{bid},i} + \sum_{i=0}^4 Q_{\text{ask},i}}$ | $[-1, 1]$ | Aggregate Top-5 depth pressure imbalance |
| `2` | **Inventory Skew** | $\frac{q}{q_{\max}}$ | $[-1, 1]$ | Signed current net position ratio |
| `3` | **Inventory Urgency** | $\frac{\|q\|}{q_{\max}}$ | $[0, 1]$ | Absolute inventory exposure magnitude |
| `4` | **Volatility Regime** | $z = \frac{\text{RMS} - \mu_\sigma}{\sigma_\sigma}$ | $[-5, 5]$ | $z$-scored rolling realized volatility from circular return buffer |
| `5` | **Micro-Price Offset** | $\frac{P_{\text{micro}} - P_{\text{mid}}}{\text{half\_spread}}$ | $[-5, 5]$ | Normalized distance between volume-weighted micro-price and mid |
| `6` | **Spread in Ticks** | $\log(1 + (\text{best\_ask} - \text{best\_bid} - 1))$ | $[0, 5]$ | Log-compressed current market spread in ticks |
| `7` | **Session Progress** | $\frac{t - t_0}{T_{\text{session}}}$ | $[0, 1]$ | Normalized elapsed fraction of trading session |

*Sentinel & Overflow Protection:* Empty-book sentinels (`INT64_MIN` for bids, `INT64_MAX` for asks) are explicitly guarded to prevent signed integer overflow and undefined behavior in mid-price, spread, and micro-price calculations. All features pass through a 5-$\sigma$ clamp and NaN filter.

### 3.3 Continuous Policy Action Space & Quoting Engine (`src/cpp/quote_generator.cpp`)
The PPO policy outputs a 3-dimensional continuous Gaussian action $\mathbf{a} = (\Delta_{\text{bid}}, \Delta_{\text{ask}}, \text{size\_frac}) \in \mathbb{R}^3$:
- $\Delta_{\text{bid}} \in [0.0, 20.0]$: Desired bid offset from mid-price in tick units.
- $\Delta_{\text{ask}} \in [0.0, 20.0]$: Desired ask offset from mid-price in tick units.
- $\text{size\_frac} \in [0.01, 1.0]$: Fraction of maximum order size to quote on each side.

**Discretization & Safety Invariants:**
- Prices are rounded to discrete exchange ticks:
  $$P_{\text{bid}} = P_{\text{mid}} - \lfloor \Delta_{\text{bid}} \cdot \text{tick\_size} \rceil, \quad P_{\text{ask}} = P_{\text{mid}} + \lfloor \Delta_{\text{ask}} \cdot \text{tick\_size} \rceil$$
- **Minimum Spread Invariant:** $P_{\text{ask}} - P_{\text{bid}} \ge \text{tick\_size}$ is enforced unconditionally.
- **Quote Sizing:** $Q_{\text{bid}} = Q_{\text{ask}} = \text{clamp}(\lfloor \text{size\_frac} \cdot \text{max\_order\_size} \rceil, 1, \text{max\_order\_size})$.
- **TTL Stamping:** Quotes are tagged with microsecond-resolution expiration timestamps ($100\text{ ms}$).

### 3.4 Reinforcement Learning Agent (`src/python/ppo_agent.py`)
- **Actor-Critic Architecture:** Shared feature representation feeding separate Actor (mean $\mu(\mathbf{x})$, learnable $\log \sigma$) and Critic ($V(\mathbf{x})$) MLP heads.
- **Clipped Surrogate Loss:** Clips probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ to $[1-\epsilon, 1+\epsilon]$ with $\epsilon = 0.2$.
- **Advantage Estimation:** Generalized Advantage Estimation (GAE-$\lambda$) with $\gamma = 0.99$ and $\lambda = 0.95$. Advantages are normalized per batch.
- **Gradient Clipping:** Maximum gradient norm clipped to $0.5$ for numerical stability.

### 3.5 TorchScript Runtime & Analytical Fallback (`src/cpp/policy_inference.cpp`)
- **TorchScript Graph Execution:** The C++ engine deserializes `models/ppo_actor.pt` via the LibTorch C++ API.
- **Cryptographic Verification:** At model initialization, an internal zero-dependency C++ SHA-256 implementation hashes the model binary and compares it against `models/ppo_actor.pt.sha256`. Any mismatch aborts model loading and routes immediately to fallback.
- **Analytical Circuit Breaker (`SimplifiedPolicyInference`):** If LibTorch is not compiled, model loading fails, or latency budget is exceeded, an analytical inventory-skew heuristic generates quotes:
  $$\Delta_{\text{bid}} = 1.0 + 1.5 \max(0, \text{skew}) - 0.5 \max(0, -\text{skew}) - 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$
  $$\Delta_{\text{ask}} = 1.0 + 1.5 \max(0, -\text{skew}) - 0.5 \max(0, \text{skew}) + 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$

### 3.6 Risk Management & Execution Gateway (`src/cpp/strategy_framework.cpp`)
- **90% Inventory Halt Rule:** When absolute inventory reaches or exceeds 90% of `max_position`:
  - If excessively long ($q \ge 0.9 \cdot q_{\max}$): Buying is immediately halted ($Q_{\text{bid}} = 0$) and the ask spread is doubled to incentivize inventory liquidation.
  - If excessively short ($q \le -0.9 \cdot q_{\max}$): Selling is immediately halted ($Q_{\text{ask}} = 0$) and the bid spread is doubled.
- **Hard Clamping:** Hard position breaches are strictly clamped to `max_position`.
- **Latency Budget Enforcement:** Strategy loop tracks execution latency per event; breaches beyond `latency_budget_ns` (default 800 ns) increment lock-free performance counters.

### 3.7 Financial Accounting vs. RL Reward Shaping
LOB.0 enforces a strict architectural separation between **real-world financial accounting** and **reinforcement learning surrogate reward shaping**:

1. **Financial Ledger (Realized PnL, Cash & Equity):**
   - $\text{Cash}_{t+1} = \text{Cash}_t - P_{\text{fill}} \cdot \Delta q - \text{Fees}$
   - $\text{Equity}_t = \text{Cash}_t + \text{Inventory}_t \cdot P_{\text{mid}, t}$
   - $\text{Total PnL}_t = \text{Equity}_t - \text{InitialEquity} = \text{Realized PnL}_t + \text{Unrealized PnL}_t$
   - *Realized PnL* is credited only upon position closure: $\sum (\text{ExitPrice} - \text{EntryPrice}) \cdot \text{ClosedQty}$.
   - *Unrealized PnL* is floating mark-to-market: $\text{Inventory}_t \cdot (P_{\text{mid}, t} - \text{AvgCost})$.
2. **Reinforcement Learning Reward Shaping (Surrogate Objective):**
   $$R_t = \Delta \text{MtM}_t - \gamma \left(\frac{|\text{Inventory}_t|}{q_{\max}}\right) + \text{SpreadCapture}_t$$
   - $-\gamma \frac{|q|}{q_{\max}}$ is an artificial holding penalty to discourage toxic inventory buildup. **It is not cash flow.**
   - $\text{SpreadCapture}_t$ is proxy credit awarded exclusively on the closed portion of resting inventory.

---

## 4. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Execution Core** | **C++17** (GCC 11+, Clang 14+, MSVC 2022) | Low-latency order matching, feature pipeline, and quote generation |
| **Build System** | **CMake 3.22+** / **Ninja** | Cross-platform build orchestration (Linux, Windows MSVC, macOS) |
| **Reinforcement Learning**| **Python 3.11 / 3.12**, **PyTorch 2.0+** | PPO Actor-Critic training, GAE advantage estimation, tensor math |
| **RL Environment** | **Gymnasium**, **NumPy** | Market making MDP environment (`LobEnv`), vector math |
| **Model Serialization** | **TorchScript**, **LibTorch (C++ API)** | Serialized model export, zero-copy native C++ neural forward pass |
| **Interoperability** | **PyBind11** | C++ class bindings, zero-copy NumPy feature buffer sharing |
| **C++ Unit Testing** | **Catch2 (v3 Header Shim)** | Self-contained, zero-dependency C++ unit and property testing |
| **Python Unit Testing** | **pytest**, **Hypothesis** | Unit tests, hypothesis-driven property tests, parity suites |
| **CI / Automation** | **GitHub Actions**, **DevContainers** | Multi-platform CI (Linux GCC, Windows MSVC, LibTorch verification) |

---

## 5. Verification Results

The complete codebase has been subjected to rigorous automated verification across both Python and C++ test harnesses:

| Component | Test Suite / Target | Assertions / Regimes | Status | Verification Details |
| :--- | :--- | :---: | :---: | :--- |
| **Python RL Suite** | `pytest tests/python/` | 37 tests | **37 / 37 PASS** | 100% pass in 4.46s; zero regressions |
| **C++ Core Test Suite** | `mm_tests` | 22 test cases | **22 / 22 PASS** | **1,761 assertions** executed cleanly in Linux Codespaces |
| **Financial Accounting**| `test_pnl_accounting.py` | 10 scenarios | **10 / 10 PASS** | Long/short, partial/full close, reversals, multiple fills, drift |
| **Feature Parity** | `test_feature_parity.py` | 7 market regimes | **7 / 7 PASS** | Exact numerical alignment between C++ and Python feature math |
| **Interface Dimensions**| `test_interface_dimensions.py` | Cross-stack | **5 / 5 PASS** | Observation dim = 8, Action dim = 3 verified across C++ and Py |
| **Model Integrity** | `ppo_actor.pt.sha256` | SHA-256 Digest | **VERIFIED** | Cryptographic hash: `88bad3b4...` verified at runtime |
| **Native C++ Inference**| `verify_native_inference` | End-to-end forward | **PASS** | Native LibTorch tensor forward pass and quote generation verified |

---

## 6. Repository Structure

```
LOB.0/
├── .devcontainer/                         # Turnkey dev environment for GitHub Codespaces
│   ├── devcontainer.json                  # DevContainer specification (Ubuntu 22.04 + GCC + Python)
│   └── setup.sh                           # Setup script (installs LibTorch, Ninja, and Python packages)
├── .github/workflows/ci.yml               # Automated CI for Python, Linux C++, Windows MSVC, & LibTorch
├── CMakeLists.txt                         # Top-level CMake configuration (C++17)
├── pyproject.toml                         # Python package and build specification
├── README.md                              # Main architectural documentation
├── BUILDING.md                            # Detailed compilation and build guide
├── LICENSE                                # MIT License
├── CITATION.cff                           # Academic citation metadata
│
├── include/                               # Public C++ Header Files
│   ├── catch2/
│   │   └── catch_test_macros.hpp          # Self-contained zero-dependency test runner shim
│   └── mm/
│       ├── types.hpp                      # Constants (kObsDim=8, kActionDim=3) and enums (Side, VolRegime)
│       ├── lob_types.hpp                  # Cacheline-aligned PriceLevel (64B) and OrderEvent (32B)
│       ├── data_models.hpp                # MarketFeatures (8D), QuoteParams, and RewardSignal
│       ├── lob_engine.hpp                 # Price-time priority LobEngine interface
│       ├── feature_pipeline.hpp           # 8D feature extraction pipeline interface
│       ├── quote_generator.hpp            # QuoteGenerator and IAlphaSignal interface
│       ├── policy_inference.hpp           # IPolicyInference, PolicyInference, and SimplifiedPolicyInference
│       ├── strategy_framework.hpp         # Strategy event loop, OrderGateway, and PerfStats
│       └── tick_replayer.hpp              # Lock-free SPSCQueue and chronological TickReplayer
│
├── src/
│   ├── cpp/                               # C++17 Core Engine Implementations
│   │   ├── lob_engine.cpp                 # Order matching, level updates, and crossed-book guards
│   │   ├── feature_pipeline.cpp           # O(1) sliding-window variance, OFI, and sentinel safety
│   │   ├── quote_generator.cpp            # Quote rounding, minimum spread invariant, and 90% halt
│   │   ├── policy_inference.cpp           # LibTorch forward pass, SHA-256 digest, and fallback circuit
│   │   ├── strategy_framework.cpp         # Strategy event loop, order gateway, and latency tracking
│   │   ├── tick_replayer.cpp              # Tick replay with 1-second gap detection and SPSC safety
│   │   ├── mm_backtest.cpp                # Standalone backtest engine runner
│   │   ├── mm_bindings.cpp                # PyBind11 bindings with input validation
│   │   └── verify_native_inference.cpp    # End-to-end native C++ inference verification binary
│   │
│   └── python/                            # Python Reinforcement Learning Modules
│       ├── ppo_agent.py                   # ActorCritic network architecture and PPO update logic
│       ├── trajectory_buffer.py           # Pre-allocated rollout buffer with GAE(λ) advantage calculation
│       ├── reward_shaper.py               # Decomposed multi-factor surrogate reward calculator
│       ├── mm_env.py                      # Gymnasium LobEnv market simulator with full financial ledger
│       ├── feature_pipeline.py            # Python reference feature pipeline for parity verification
│       └── mm_train.py                    # Multi-epoch PPO training loop with checkpoint export
│
├── tests/
│   ├── CMakeLists.txt                     # C++ test target definitions
│   ├── cpp/                               # C++ Unit & Property Test Suite (Catch2)
│   │   ├── test_placeholder.cpp           # Catch2 test runner entry point
│   │   ├── test_data_models.cpp           # Struct size (32B/64B) and alignment verification
│   │   ├── test_lob_engine.cpp            # Order matching, partial fills, and crossed-book rejection
│   │   ├── test_feature_pipeline.cpp      # OFI formulas, sentinel guards, and feature bounded bounds
│   │   ├── test_policy_inference.cpp      # SHA-256 checksum and policy skew regressions
│   │   ├── test_quote_generator.cpp       # Discretization, minimum spread, and 90% inventory halt
│   │   ├── test_tick_replayer.cpp         # Lock-free SPSC queue wraparound and threaded safety
│   │   └── test_strategy_framework.cpp    # Event loop, gateway cash ledger, and shutdown cleanup
│   │
│   └── python/                            # Python Unit & Property Test Suite (pytest + Hypothesis)
│       ├── test_train_step.py             # PPOAgent.train_step smoke verification
│       ├── test_feature_parity.py         # Python/C++ numerical parity across 7 distinct regimes
│       ├── test_interface_dimensions.py   # State and action space dimensions (8 obs, 3 actions)
│       ├── test_pnl_accounting.py         # 10 rigorous financial accounting scenarios
│       ├── test_mm_env.py                 # LobEnv step mechanics and deterministic replay
│       ├── test_ppo_agent.py              # PPO surrogate loss, clipping, and TorchScript export
│       └── test_trajectory_buffer.py      # GAE advantage normalization and buffer lifecycle
│
├── models/                                # Serialized Checkpoints
│   ├── ppo_actor.pt                       # Exported TorchScript actor network (SHA-256 verified)
│   └── ppo_actor.pt.sha256                # Cryptographic checksum digest
│
├── docs/                                  # Project Documentation
│   ├── CODE_AUDIT_REPORT.md               # Baseline engineering audit and hardening log
│   ├── ARCHITECTURE.md                    # Detailed architectural breakdown
│   ├── INSTALLATION.md                    # Installation and troubleshooting guide
│   └── BENCHMARKS.md                      # Latency, throughput, and profiling notes
│
└── data/                                  # Market Data Store
    ├── README.md                          # Data format specification and replay guide
    └── .gitkeep
```

---

## 7. Quick Start & Reproduction

### 7.1 One-Click via GitHub Codespaces (Recommended)
You can launch, compile, and run the entire system in your browser without local installation:
1. Open the repository on GitHub and click **Code** -> **Codespaces** -> **Create codespace on main**.
2. Run in the Codespace terminal:
   ```bash
   source .venv/bin/activate
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
   cmake --build build --target mm_tests mm_backtest verify_native_inference
   ./build/tests/mm_tests
   ./build/src/cpp/verify_native_inference
   python -m pytest tests/ -v
   ```

### 7.2 Local Python Setup & Testing

```bash
# Clone repository
git clone https://github.com/sumansuman1212/LOB.0.git
cd LOB.0

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\Activate.ps1

# Install package and test dependencies
pip install -e .
pip install pytest hypothesis

# Run full Python test suite (37 tests)
python -m pytest tests/ -v

# Run PPO training smoke test
python tests/python/test_train_step.py

# Train PPO agent on simulated LOB
python -m src.python.mm_train --episodes 5 --rollout 2048 --batch-size 256 --epochs 10
```

### 7.3 Local C++ Engine Build (CMake)

The C++ build uses CMake 3.22+. The test suite uses a self-contained test runner (`include/catch2/catch_test_macros.hpp`) with **zero mandatory external library requirements**.

```bash
# Configure CMake (Standalone Fallback)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build test runner, backtest simulator, and native inference verification
cmake --build build --target mm_tests mm_backtest verify_native_inference

# Run native C++ test suite
./build/tests/mm_tests                  # Windows: .\build\tests\Release\mm_tests.exe
./build/src/cpp/verify_native_inference # Windows: .\build\src\cpp\Release\verify_native_inference.exe
```

*LibTorch Build:* To enable native LibTorch tensor inference, supply `-DCMAKE_PREFIX_PATH=/path/to/libtorch` or use Python's PyTorch directory:
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
```

---

## 8. Limitations & Scope

To maintain rigorous scientific and engineering integrity, the scope and assumptions of LOB.0 are explicitly stated:
1. **Synthetic Simulation:** In `LobEnv`, order flow is generated using a geometric random walk mid-price combined with Poisson-inspired depth and aggressor fill dynamics. While the C++ `TickReplayer` natively reads chronological tick events, proprietary live exchange feeds (e.g. ITCH/OUCH, Multicast FAST) are not bundled.
2. **Simplified Fill Model:** In production exchange matching engines, fill priority depends on microsecond FIFO position within each discrete price level (Level-3 queue position). `LobEnv` simulates fill probabilities as a function of quote distance from mid and resting volume rather than exact Level-3 queue simulation.
3. **No Direct Exchange Connectivity:** LOB.0 provides the algorithmic matching and risk engine; it does not include physical FIX session engines, kernel-bypass network drivers (Solarflare EF_VI), or FPGA NIC integrations.
4. **Queue & Market Impact:** Quoting actions do not model endogenous market impact—i.e., external market participants do not dynamically alter their quote placement in reaction to the agent's resting orders.
5. **Soft Risk Limit vs Exchange Execution:** The 90% inventory halt rule suppresses quote generation at the strategy layer; in live trading, in-flight orders already acknowledged by an exchange may still execute prior to cancellation confirmation.

---

## 9. License & Citation

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

If you use this project in your academic coursework, research, or quantitative portfolio evaluation, please cite it using [`CITATION.cff`](CITATION.cff):

```bibtex
@software{saurav_lob0_2026,
  author = {Saurav},
  title = {LOB.0: Reinforcement Learning-Driven Adaptive Market Making Engine},
  year = {2026},
  url = {https://github.com/sumansuman1212/LOB.0}
}
```
