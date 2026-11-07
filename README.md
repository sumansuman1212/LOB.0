# LOB.0 — Reinforcement Learning-Driven Adaptive Market Making Engine

[![LOB.0 CI](https://github.com/sumansuman1212/LOB.0/actions/workflows/ci.yml/badge.svg)](https://github.com/sumansuman1212/LOB.0/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](include/mm/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](src/python/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](src/python/ppo_agent.py)

**LOB.0** is an undergraduate quantitative engineering and systems research project investigating how reinforcement learning policies can adaptively position quotes in a limit order book (LOB) while subject to inventory risk constraints, minimum-spread invariants, and low-latency execution budgets.

The project bridges an offline **Python (PyTorch) reinforcement learning harness** (PPO with Generalized Advantage Estimation) with an ultra-low-latency **C++17 execution core** that handles price-time priority order matching, 8-dimensional microstructure feature extraction, TorchScript neural policy inference, and inventory-aware quote generation.

---

## 1. Motivation & Engineering Goals

Standard classical market-making models (e.g., Avellaneda-Stoikov, Guéant-Tapia-Manziadi) derive optimal quoting spreads under closed-form assumptions regarding constant Poisson arrival intensities and Brownian price dynamics. In reality, short-term price discovery is heavily driven by **order-flow imbalance (OFI)**, micro-price deviations, and transient volatility spikes.

The goal of this project is to:
1. Implement a **deterministic, cache-aware limit order book engine** in modern C++17 with strict transaction invariants (no crossed books, monotonic sequencing, 64-byte cacheline alignment).
2. Extract **real-time microstructure features** (OFI, rolling realized volatility, inventory urgency) without heap allocation on the hot path.
3. Train an **Actor-Critic agent using Proximal Policy Optimization (PPO)** in a Gymnasium environment to discover adaptive quote skews.
4. Bridge the two systems using **frozen TorchScript graphs** verified at runtime via standalone SHA-256 checksums, with an analytical fallback circuit breaker.
5. Maintain strict separation between **real-world financial accounting** (cash, equity, realized/unrealized PnL) and **reinforcement learning surrogate reward shaping**.

---

## 2. Architecture Overview

```
                                 OFFLINE TRAINING HARNESS (Python)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│   Synthetic L2 Feed / Parquet ──► LobEnv (Gymnasium) ──► PPOAgent (ActorCritic MLP)     │
│                                           │                      │                     │
│                                           ▼                      ▼                     │
│                                   RewardShaper           TrajectoryBuffer (GAE-λ)      │
│                                           │                      │                     │
│                                           ▼                      ▼                     │
│                                    Export Actor Graph ────► models/ppo_actor.pt        │
│                                                              + SHA-256 Checksum        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Hot-Reload / Checksum Verify
                                            ▼
                                ONLINE EXECUTION CORE (C++17)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│   Tick Feed ──► TickReplayer (SPSC Queue) ──► StrategyFramework                        │
│                                                    │                                   │
│           ┌────────────────────────────────────────┴──────────────────────────┐        │
│           ▼                                                                   ▼        │
│     LobEngine (O(1) Map) ──► FeaturePipeline (8D) ──► PolicyInference ──► Quoter       │
│     (64B PriceLevels)        (OFI, Vol, Skew)         (LibTorch / Fallback)   │        │
│                                                                               │        │
│     OrderGateway (Execution / Net Position) ◄─────────────────────────────────┘        │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implemented Subsystems & Mathematical Formulation

### 3.1 C++17 Limit Order Book (`src/cpp/lob_engine.cpp`)
- **Price-Time Priority:** Dual `std::map` trees (`std::greater` for descending bids, `std::less` for ascending asks). Note that `std::map` and `std::list` perform dynamic node allocations on the heap upon insertion.
- **Order Tracking & Complexity:** Internal hash map (`orders_`) maps `order_id` to its price-level list iterator in $O(1)$ average time for fast cancellation and modifications. If an order cancellation completely empties a price level, removing that level from `std::map` incurs an $O(\log M)$ tree deletion.
- **Memory Alignment & Cache Footprint:** `PriceLevel` structs are aligned to 64 bytes (`alignas(64)`) to prevent single-level structs from straddling a standard x86-64 cacheline boundary. `OrderEvent` structs are packed to 32 bytes (`sizeof(OrderEvent) == 32`). Contiguous cache locality is realized in `BookSnapshot` (`std::array<PriceLevel, 10>`).
- **Invariant Safety:** Rejects duplicate Add IDs, monotonic timestamp backward jumps, non-positive prices, and crossed-book attempts without mutating internal state.

### 3.2 Microstructure Feature Pipeline (`src/cpp/feature_pipeline.cpp`)
Extracts an 8-dimensional observation vector $\mathbf{x} \in \mathbb{R}^8$ with zero dynamic memory allocation on the hot path:

| Index | Feature Name | Mathematical Definition | Nominal Range |
| :---: | :--- | :--- | :---: |
| `0` | **$\text{OFI}_{L1}$** | $\frac{Q_{\text{bid},0} - Q_{\text{ask},0}}{Q_{\text{bid},0} + Q_{\text{ask},0}}$ (Level-1 depth imbalance) | $[-1, 1]$ |
| `1` | **$\text{OFI}_{L5}$** | $\frac{\sum_{i=0}^4 Q_{\text{bid},i} - \sum_{i=0}^4 Q_{\text{ask},i}}{\sum_{i=0}^4 Q_{\text{bid},i} + \sum_{i=0}^4 Q_{\text{ask},i}}$ (Top-5 depth imbalance) | $[-1, 1]$ |
| `2` | **Inventory Skew** | $\frac{q}{q_{\max}}$ (signed net position ratio) | $[-1, 1]$ |
| `3` | **Inventory Urgency** | $\frac{\|q\|}{q_{\max}}$ (absolute exposure magnitude) | $[0, 1]$ |
| `4` | **Volatility Regime** | $z$-scored rolling realized volatility: $\frac{\text{RMS} - \mu_\sigma}{\sigma_\sigma}$ | $[-5, 5]$ |
| `5` | **Micro-Price Offset** | $\frac{P_{\text{micro}} - P_{\text{mid}}}{\text{half\_spread}}$ where $P_{\text{micro}} = \frac{P_{\text{ask}} Q_{\text{bid},0} + P_{\text{bid}} Q_{\text{ask},0}}{Q_{\text{bid},0} + Q_{\text{ask},0}}$ | $[-5, 5]$ |
| `6` | **Spread in Ticks** | $\log(1 + (\text{best\_ask} - \text{best\_bid} - 1))$ | $[0, 5]$ |
| `7` | **Session Progress** | Elapsed fraction of trading day: $\frac{t - t_0}{T_{\text{session}}}$ | $[0, 1]$ |

*Implementation Note on Realized Volatility:* `FeaturePipeline` uses a circular buffer of log mid-price returns with sliding-window sum ($\sum r$) and sum-of-squares ($\sum r^2$) accumulators in double precision. Realized volatility is calculated as the root-mean-square (RMS) return $\sqrt{\frac{1}{W}\sum r_i^2}$. It is an $O(1)$ fixed-window sliding accumulator designed for high-throughput updates.

### 3.3 Continuous Policy Action Space
The PPO policy network outputs a 3-dimensional continuous Gaussian action vector $\mathbf{a} = (\Delta_{\text{bid}}, \Delta_{\text{ask}}, \text{size\_frac}) \in \mathbb{R}^3$:

| Index | Action Parameter | Description | Valid Bounds |
| :---: | :--- | :--- | :---: |
| `0` | $\Delta_{\text{bid}}$ | Desired bid distance from mid-price in tick units | $[0.0, 20.0]$ |
| `1` | $\Delta_{\text{ask}}$ | Desired ask distance from mid-price in tick units | $[0.0, 20.0]$ |
| `2` | $\text{size\_frac}$ | Fraction of `max_order_size` to quote on each side | $[0.01, 1.0]$ |

### 3.4 Quote Generator & Risk Controls (`src/cpp/quote_generator.cpp`)
- **Discrete Tick Rounding:** Converts continuous offsets into exchange-valid integer prices:
  $$P_{\text{bid}} = P_{\text{mid}} - \lfloor \Delta_{\text{bid}} \cdot \text{tick\_size} \rceil, \quad P_{\text{ask}} = P_{\text{mid}} + \lfloor \Delta_{\text{ask}} \cdot \text{tick\_size} \rceil$$
- **Minimum Spread Invariant:** Strictly clamps prices so $P_{\text{ask}} - P_{\text{bid}} \ge \text{tick\_size}$.
- **Lot Sizing:** Scales quote volume: $Q_{\text{bid}} = Q_{\text{ask}} = \text{clamp}(\lfloor \text{size\_frac} \cdot \text{max\_order\_size} \rceil, 1, \text{max\_order\_size})$.
- **90% Inventory Halt Soft Limit:** When $|q| \ge 0.9 \cdot q_{\max}$:
  - If long ($q > 0$): Halts buying ($Q_{\text{bid}} = 0$) and widens ask distance by $2\times$ while preserving alpha signal adjustments.
  - If short ($q < 0$): Halts selling ($Q_{\text{ask}} = 0$) and widens bid distance by $2\times$ while preserving alpha signal adjustments.
- **TTL Stamping:** Quotes carry microsecond TTL expiration timestamps ($100\text{ ms}$).

### 3.5 C++ Neural Inference & Analytical Fallback (`src/cpp/policy_inference.cpp`)
- **TorchScript Runtime:** Loads serialized PyTorch actor networks (`models/ppo_actor.pt`) via LibTorch.
- **Cryptographic Checksum Verification:** Verifies the model's SHA-256 digest at startup against `models/ppo_actor.pt.sha256` using an internal zero-dependency SHA-256 implementation.
- **Analytical Fallback Circuit Breaker (`SimplifiedPolicyInference`):** If LibTorch is disabled, throws a runtime exception, encounters non-finite observations, or breaches latency budgets, execution routes to an inventory-aware quoting heuristic inspired by Avellaneda-Stoikov:
  $$\Delta_{\text{bid}} = 1.0 + 1.5 \max(0, \text{skew}) - 0.5 \max(0, -\text{skew}) - 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$
  $$\Delta_{\text{ask}} = 1.0 + 1.5 \max(0, -\text{skew}) - 0.5 \max(0, \text{skew}) + 0.3 \cdot \text{OFI} + 0.2 \max(0, \text{vol})$$
  *(Note: This heuristic does not claim to be the exact continuous-time Avellaneda-Stoikov model, as it omits the terminal horizon discounting $(T-t)$ and uses linear additive terms for OFI and volatility).*

### 3.6 Financial Accounting vs. Reinforcement Learning Reward
To ensure quantitative and accounting rigor, LOB.0 maintains strict separation between **real-world financial ledgers** and **reinforcement learning surrogate reward shaping**:

1. **Financial Accounting Identities (Ledger):**
   - **Cash Balance:** $\text{Cash}_{t+1} = \text{Cash}_t - P_{\text{fill}} \cdot \Delta q - \text{Fees}$
   - **Net Inventory:** $\text{Inventory}_{t+1} = \text{Inventory}_t + \Delta q$
   - **Portfolio Equity:** $\text{Equity}_t = \text{Cash}_t + \text{Inventory}_t \cdot P_{\text{mid}, t}$
   - **Total Financial PnL:** $\text{PnL}_t = \text{Equity}_t - \text{InitialEquity} = \text{Realized PnL}_t + \text{Unrealized PnL}_t$
     - *Realized PnL:* Closed cash profit: $\sum (\text{ExitPrice} - \text{EntryPrice}) \cdot \text{ClosedQty}$.
     - *Unrealized PnL:* Floating mark-to-market gain: $\text{Inventory}_t \cdot (P_{\text{mid}, t} - \text{AvgCost})$.
2. **Reinforcement Learning Reward Signal (Shaping Heuristic):**
   $$R_t = \Delta \text{MtM}_t - \gamma \left(\frac{|\text{Inventory}_t|}{q_{\max}}\right) + \text{SpreadCapture}_t$$
   - $\Delta \text{MtM}_t = (P_{\text{mid}, t} - P_{\text{mid}, t-1}) \cdot \text{Inventory}_{t-1}$: One-step mark-to-market drift.
   - $-\gamma \frac{|q|}{q_{\max}}$: Artificial holding penalty to penalize inventory accumulation. **This is not cash flow.**
   - $\text{SpreadCapture}_t = \min(|q_{t-1}|, \text{fill\_qty}) \cdot \text{half\_spread}$: Proxy credit awarded exclusively on the closed portion of resting inventory.

---

## 4. Repository Structure

```
LOB.0/
├── .devcontainer/                         # Turnkey Linux dev environment for GitHub Codespaces
│   ├── devcontainer.json                  # DevContainer specification (Ubuntu 22.04 + C++17 + Python)
│   └── setup.sh                           # Codespaces setup script (installs Linux LibTorch & tools)
├── .github/workflows/ci.yml               # Automated CI for Python, Linux C++, Windows MSVC, & LibTorch
├── CMakeLists.txt                         # Top-level CMake configuration (C++17)
├── pyproject.toml                         # Python package specification
├── README.md                              # This file
├── BUILDING.md                            # Quick build reference guide
├── LICENSE                                # MIT License
├── CITATION.cff                           # Academic citation metadata
├── test_train_step.py                     # Standalone PPO training step smoke test
│
├── include/                               # Public C++ Header Files
│   ├── catch2/
│   │   └── catch_test_macros.hpp          # Self-contained zero-dependency test shim
│   └── mm/
│       ├── types.hpp                      # Enums (Side, VolRegime) and engine constants (kObsDim=8, kActionDim=3)
│       ├── lob_types.hpp                  # PriceLevel (64B), OrderEvent (32B), BookSnapshot
│       ├── data_models.hpp                # MarketFeatures (8D), QuoteParams, RewardSignal
│       ├── lob_engine.hpp                 # LobEngine price-time priority interface
│       ├── feature_pipeline.hpp           # 8D FeaturePipeline interface
│       ├── quote_generator.hpp            # QuoteGenerator & IAlphaSignal interface
│       ├── policy_inference.hpp           # IPolicyInference, PolicyInference, SimplifiedPolicyInference
│       ├── strategy_framework.hpp         # Strategy event loop, OrderGateway, PerfStats
│       └── tick_replayer.hpp              # SPSC lock-free tick queue
│
├── src/
│   ├── cpp/                               # C++17 Core Engine Implementations
│   │   ├── lob_engine.cpp                 # Price-time order matching & validation
│   │   ├── feature_pipeline.cpp           # O(1) sliding-window variance & OFI
│   │   ├── quote_generator.cpp            # Quote discretization & inventory halt
│   │   ├── policy_inference.cpp           # LibTorch forward pass, SHA-256 & fallback
│   │   ├── strategy_framework.cpp         # Strategy event loop & latency monitoring
│   │   ├── tick_replayer.cpp              # Tick replay & data gap detection
│   │   ├── mm_backtest.cpp                # Standalone backtest runner
│   │   ├── mm_bindings.cpp                # PyBind11 bindings for Python
│   │   └── verify_native_inference.cpp    # End-to-end native C++ inference verification
│   │
│   └── python/                            # Python Reinforcement Learning Modules
│       ├── ppo_agent.py                   # ActorCritic network & PPO update logic
│       ├── trajectory_buffer.py           # Pre-allocated buffer & GAE computation
│       ├── reward_shaper.py               # Decomposed multi-factor reward function
│       ├── mm_env.py                      # Gymnasium LobEnv market simulator (Cash/Equity ledger)
│       ├── feature_pipeline.py            # Python reference mirror for parity testing
│       └── mm_train.py                    # Multi-epoch PPO training runner
│
├── tests/                                 # Verification Suites
│   ├── test_data_models.cpp               # C++ memory layout & alignment tests
│   ├── test_lob_engine.cpp                # C++ order book mutation & safety tests
│   ├── test_feature_pipeline.cpp          # C++ feature pipeline correctness tests
│   ├── test_policy_inference.cpp          # C++ SHA-256 & policy skew regression tests
│   ├── test_quote_generator.cpp           # C++ quote generation & halt rule tests
│   ├── test_tick_replayer.cpp             # C++ queue throughput & gap tests
│   ├── test_strategy_framework.cpp        # C++ strategy event cycle tests
│   ├── test_feature_parity.py             # Python/C++ feature parity tests across 7 regimes
│   ├── test_interface_dimensions.py       # Cross-stack dimension verification (8 obs, 3 actions)
│   ├── test_pnl_accounting.py             # Financial ledger accounting tests (10 scenarios)
│   ├── test_mm_env.py                     # Python LobEnv & unwinding regression tests
│   ├── test_ppo_agent.py                  # Python PPO loss & TorchScript export tests
│   └── test_trajectory_buffer.py          # Python GAE advantage normalization tests
│
├── models/                                # Serialized Checkpoints
│   ├── ppo_actor.pt                       # Exported TorchScript actor network (285 KB)
│   └── ppo_actor.pt.sha256                # Cryptographic SHA-256 digest
│
├── docs/                                  # Extended Technical Documentation
│   ├── ARCHITECTURE.md                    # Detailed architectural breakdown
│   ├── INSTALLATION.md                    # Environment setup & build troubleshooting
│   └── BENCHMARKS.md                      # Latency, throughput, and profiling notes
│
└── data/                                  # Market Data Store (placeholder for .parquet)
    └── .gitkeep
```

---

## 5. Quick Start & Reproduction

### 5.1 One-Click via GitHub Codespaces (Recommended)
You can launch, compile, and run the complete system directly in your browser:
1. Click **Code** -> **Codespaces** -> **Create codespace on main**.
2. Run in the Codespace terminal:
   ```bash
   source .venv/bin/activate
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
   cmake --build build --target mm_tests mm_backtest verify_native_inference
   ./build/tests/mm_tests
   ./build/src/cpp/verify_native_inference
   pytest -v tests/
   ```

### 5.2 Python Environment Setup (Local)

```bash
# 1. Clone repository
git clone https://github.com/sumansuman1212/LOB.0.git
cd LOB.0

# 2. Create and activate virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install dependencies in editable mode
pip install -e .
pip install pytest hypothesis
```

### 5.3 Running the Python Test Suite

```bash
# Run all 37 unit, accounting, dimension, and parity regression tests
pytest -v tests/
```

### 5.4 Training the PPO Market Maker

```bash
# Run a training session on the Gymnasium LOB simulator
python -m src.python.mm_train --episodes 5 --rollout 2048 --batch-size 256 --epochs 10
```

### 5.5 Building the C++ Engine (CMake)

The C++ build uses CMake 3.22+. The test suite uses a self-contained test runner (`include/catch2/catch_test_macros.hpp`) with **zero mandatory external library requirements**.

```bash
# 1. Configure CMake (Standalone Fallback)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build test runner, backtest simulator, and native inference verification
cmake --build build --target mm_tests mm_backtest verify_native_inference

# 3. Run native C++ test suite
# Linux / macOS:
./build/tests/mm_tests
./build/src/cpp/verify_native_inference

# Windows MSVC:
.\build\tests\Release\mm_tests.exe
.\build\src\cpp\Release\verify_native_inference.exe
```

*Note on LibTorch (Optional):* To build with native LibTorch neural inference, point CMake to your LibTorch distribution:
`-DCMAKE_PREFIX_PATH=/path/to/libtorch` (or use Python's PyTorch installation via `$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')`). If omitted, the engine automatically compiles with its built-in SHA-256 engine and analytical fallback policy.

---

## 6. Honest Scope & Limitations

As an undergraduate research and engineering project, transparency regarding design scope and assumptions is paramount:
1. **Simulation vs. Real Market Data:** By default, `LobEnv` generates order flow via a geometric random-walk mid-price model with simulated fill probabilities. While historical Parquet tick files can be loaded, no proprietary live exchange tick archives are bundled with this repository.
2. **Fill Modeling:** In real matching engines, fills depend on microscopic queue priority (orders ahead in the FIFO queue). The simulated environment estimates fill probabilities as a function of distance from mid and resting depth rather than full Level-3 message simulation.
3. **Execution Environment on Windows:** Windows 11 ASLR requires 64-bit binaries built with dynamic base (`/DYNAMICBASE` on MSVC or modern 64-bit MinGW-w64). Legacy 32-bit MinGW toolchains produce `0xC0000142` (Illegal System DLL Relocation) errors. Use MSVC 2022, modern MSYS2 UCRT64, or GitHub Codespaces.
4. **Soft Risk Limit (90% Inventory Halt):** Halting quote generation on the accumulating side is a soft application-level control; in-flight orders already acknowledged by an exchange can still execute.

---

## 7. License & Citation

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

If you find this codebase helpful in your academic coursework or research, please cite it using [`CITATION.cff`](CITATION.cff):
```bibtex
@software{saurav_lob0_2026,
  author = {Saurav},
  title = {LOB.0: Reinforcement Learning-Driven Adaptive Market Making Engine},
  year = {2026},
  url = {https://github.com/sumansuman1212/LOB.0}
}
```
