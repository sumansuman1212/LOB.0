# Complete Project File Map & Inventory
**Project:** RL-Driven Adaptive Market Making Engine (`mm_engine` / `LOB.0`)  
**Cross-References:** [PROJECT_COMPLETE_UNDERSTANDING.md](file:///c:/Users/saura/Coding/SSProjects/LOB.0/PROJECT_COMPLETE_UNDERSTANDING.md) | [PROJECT_ARCHITECTURE.md](file:///c:/Users/saura/Coding/SSProjects/LOB.0/PROJECT_ARCHITECTURE.md)

---

## 1. Directory Tree & Inventory

```
c:/Users/saura/Coding/SSProjects/LOB.0/
├── CMakeLists.txt                      # Top-level CMake configuration (C++17, LibTorch, PyBind11, Catch2)
├── pyproject.toml                      # Python package build configuration (PEP 517/621, dependencies)
├── README.md                           # Repository high-level README
├── test_train_step.py                  # Standalone PPOAgent.train_step smoke test
│
├── include/                            # Public C++ Header Files
│   ├── mm/                             # Core Market Making Header Definitions
│   │   ├── types.hpp                   # Side, VolRegime enums & compile-time dimension constants
│   │   ├── lob_types.hpp               # PriceLevel (64B), OrderEvent (32B), BookSnapshot
│   │   ├── data_models.hpp             # MarketFeatures (8D), QuoteAction, QuoteParams, RewardSignal
│   │   ├── lob_engine.hpp              # LobEngine class definition (price-time priority order book)
│   │   ├── feature_pipeline.hpp        # FeaturePipeline class (rolling accumulators & 8D feature extraction)
│   │   ├── quote_generator.hpp         # QuoteGenerator & IAlphaSignal interface
│   │   ├── policy_inference.hpp        # IPolicyInference, PolicyInference (TorchScript), SimplifiedPolicyInference
│   │   ├── strategy_framework.hpp      # StrategyFramework event loop, OrderGateway, PerfStats
│   │   └── tick_replayer.hpp           # TickReplayer & SPSCQueue lock-free ring buffer
│   └── catch2/                         # Vendored Catch2 test framework headers
│
├── src/                                # Source Code Root
│   ├── __init__.py                     # Python package root marker
│   ├── cpp/                            # C++ Implementations
│   │   ├── CMakeLists.txt              # CMake build definitions for mm_core, mm_backtest, and mm (Python module)
│   │   ├── lob_engine.cpp              # LobEngine implementation (O(log N) book mutations, validation)
│   │   ├── feature_pipeline.cpp        # FeaturePipeline implementation (O(1) Welford rolling vol, OFI)
│   │   ├── quote_generator.cpp         # QuoteGenerator implementation (tick rounding, inventory halt, TTL)
│   │   ├── policy_inference.cpp        # PolicyInference, standalone SHA-256 & SimplifiedPolicyInference fallback
│   │   ├── strategy_framework.cpp      # StrategyFramework & OrderGateway implementation
│   │   ├── tick_replayer.cpp           # TickReplayer implementation & gap detection (>1s)
│   │   ├── mm_backtest.cpp             # Standalone C++ backtest driver & synthetic L2 generator
│   │   └── mm_bindings.cpp             # PyBind11 bindings exporting C++ engine to Python `mm` module
│   │
│   └── python/                         # Python Reinforcement Learning Harness
│       ├── __init__.py                 # Export public Python symbols (PPOAgent, LobEnv, TrajectoryBuffer, etc.)
│       ├── trajectory_buffer.py        # Pre-allocated TrajectoryBuffer with GAE-lambda advantage computation
│       ├── reward_shaper.py            # RewardShaper & decomposed RewardSignal (MtM + inventory penalty + spread)
│       ├── ppo_agent.py                # ActorCritic PyTorch network (MLP trunk + Gaussian head) & PPOAgent
│       ├── mm_env.py                   # Gymnasium-compatible LobEnv environment for PPO training
│       └── mm_train.py                 # Full PPO training script, metric logging & TorchScript export
│
├── tests/                              # Test Suites (C++ & Python)
│   ├── CMakeLists.txt                  # CMake definitions for mm_tests executable
│   ├── test_placeholder.cpp            # Test harness entry point
│   ├── test_data_models.cpp            # Catch2 unit tests for memory layout & MarketFeatures/RewardSignal
│   ├── test_lob_engine.cpp             # Catch2 unit tests for LOB mutations, duplicate checks, crossed book
│   ├── test_feature_pipeline.cpp       # Catch2 unit tests for OFI, rolling vol, inventory features, reset
│   ├── test_policy_inference.cpp       # Catch2 unit tests for SimplifiedPolicyInference, SHA-256, fallback
│   ├── test_quote_generator.cpp        # Catch2 unit tests for spread enforcement, inventory halt, alpha signals
│   ├── test_strategy_framework.cpp     # Catch2 integration tests for end-to-end event cycle, inventory invariants
│   ├── test_tick_replayer.cpp          # Catch2 unit tests for SPSCQueue, chronological sorting, gap detection
│   ├── test_mm_env.py                  # Pytest/Hypothesis property tests for LobEnv & RewardShaper
│   ├── test_ppo_agent.py               # Pytest/Hypothesis property tests for PPO loss & TorchScript export
│   └── test_trajectory_buffer.py       # Pytest/Hypothesis property tests for GAE-lambda normalization & memory reuse
│
├── models/                             # Serialized Models & Artifacts
│   ├── .gitkeep                        # Git directory anchor
│   ├── ppo_actor.pt                    # Serialized TorchScript actor model for C++ inference
│   └── ppo_actor.pt.sha256             # SHA-256 checksum for model integrity validation
│
├── data/                               # Market Data Directory
│   └── .gitkeep                        # Anchor for NSE L2 tick archives (.parquet)
│
└── Veil/                               # Auxiliary Flutter Application (Scaffolded UI module)
    ├── pubspec.yaml                    # Flutter project configuration
    ├── analysis_options.yaml           # Dart analysis configuration
    ├── lib/                            # Flutter Dart sources
    └── ...
```

---

## 2. Component Responsibility Matrix

| File Path | Role | Layer | Key Exports | Criticality |
| :--- | :--- | :--- | :--- | :--- |
| [`include/mm/types.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/types.hpp) | Base Types & Constants | Core Types | `Side`, `VolRegime`, `kMaxDepth`, `kObsDim`, `kActionDim` | **CRITICAL** |
| [`include/mm/lob_types.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/lob_types.hpp) | LOB Data Structures | LOB Core | `PriceLevel` (64B), `OrderEvent` (32B), `BookSnapshot` | **CRITICAL** |
| [`include/mm/data_models.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/data_models.hpp) | Market Features & Signals | RL/Feature | `MarketFeatures`, `QuoteAction`, `QuoteParams`, `RewardSignal` | **CRITICAL** |
| [`include/mm/lob_engine.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/lob_engine.hpp) | Limit Order Book API | LOB Core | `LobEngine`, `OrderEntry` | **CRITICAL** |
| [`src/cpp/lob_engine.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/lob_engine.cpp) | LOB Engine Implementation | LOB Core | Price-time priority matching, snapshot, validation | **CRITICAL** |
| [`include/mm/feature_pipeline.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/feature_pipeline.hpp) | Feature Pipeline API | Feature Extraction | `FeaturePipeline` | **CRITICAL** |
| [`src/cpp/feature_pipeline.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/feature_pipeline.cpp) | Feature Calculation Logic | Feature Extraction | OFI l1/l5, Welford rolling realized vol, sanitization | **CRITICAL** |
| [`include/mm/quote_generator.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/quote_generator.hpp) | Quote Generator API | Execution & Quoting | `QuoteGenerator`, `IAlphaSignal` | **CRITICAL** |
| [`src/cpp/quote_generator.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/quote_generator.cpp) | Quote Generation Logic | Execution & Quoting | Continuous action translation, inventory halt, TTL | **CRITICAL** |
| [`include/mm/policy_inference.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/policy_inference.hpp) | Policy Inference API | Neural Inference | `IPolicyInference`, `PolicyInference`, `SimplifiedPolicyInference` | **CRITICAL** |
| [`src/cpp/policy_inference.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/policy_inference.cpp) | Inference & SHA-256 | Neural Inference | LibTorch forward pass, atomic reload, SHA-256 | **CRITICAL** |
| [`include/mm/strategy_framework.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/strategy_framework.hpp) | Strategy Framework API | Orchestration | `StrategyFramework`, `OrderGateway`, `PerfStats`, `StrategyConfig` | **CRITICAL** |
| [`src/cpp/strategy_framework.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/strategy_framework.cpp) | Strategy Event Loop | Orchestration | `step()`, `run()`, latency tracking, fallback circuit breaker | **CRITICAL** |
| [`include/mm/tick_replayer.hpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/include/mm/tick_replayer.hpp) | Ingestion & Queue API | Data Ingestion | `TickReplayer`, `SPSCQueue` (lock-free) | **IMPORTANT** |
| [`src/cpp/tick_replayer.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/tick_replayer.cpp) | Tick Replayer Logic | Data Ingestion | Chronological sorting, ring buffer pump, gap detection | **IMPORTANT** |
| [`src/cpp/mm_bindings.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/mm_bindings.cpp) | PyBind11 Bridge | C++/Python Binding | Python module `mm` (`mm.LobEngine`, `mm.FeaturePipeline`, `mm.step`) | **CRITICAL** |
| [`src/cpp/mm_backtest.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/cpp/mm_backtest.cpp) | Standalone C++ Driver | Backtesting | `main()` driver with synthetic market generator | **SUPPORTING** |
| [`src/python/ppo_agent.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/python/ppo_agent.py) | PyTorch Actor-Critic | RL Algorithm | `ActorCritic`, `PPOAgent`, `ActorInferenceModule`, TorchScript export | **CRITICAL** |
| [`src/python/trajectory_buffer.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/python/trajectory_buffer.py) | Rollout Storage & GAE | RL Algorithm | `TrajectoryBuffer`, `Transition`, reverse GAE-lambda | **CRITICAL** |
| [`src/python/reward_shaper.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/python/reward_shaper.py) | Multi-Objective Reward | RL Environment | `RewardShaper`, `RewardSignal` (decomposed rewards) | **CRITICAL** |
| [`src/python/mm_env.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/python/mm_env.py) | Gymnasium RL Env | RL Environment | `LobEnv` (8D observation space, 3D action space) | **CRITICAL** |
| [`src/python/mm_train.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/src/python/mm_train.py) | RL Training Harness | Training Pipeline | `train()`, `calculate_sharpe_and_drawdown()`, CLI entrypoint | **CRITICAL** |
| [`test_train_step.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/test_train_step.py) | Python Smoke Test | Verification | Quick single-step training test | **SUPPORTING** |
| [`tests/test_lob_engine.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_lob_engine.cpp) | C++ LOB Unit Tests | Verification | Price-time priority, duplicate order, transaction rollback | **IMPORTANT** |
| [`tests/test_feature_pipeline.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_feature_pipeline.cpp) | C++ Feature Tests | Verification | Boundedness, OFI formula checks, inventory features | **IMPORTANT** |
| [`tests/test_policy_inference.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_policy_inference.cpp) | C++ Inference Tests | Verification | SimplifiedPolicyInference, SHA-256 file hashing | **IMPORTANT** |
| [`tests/test_quote_generator.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_quote_generator.cpp) | C++ Quoter Tests | Verification | Spread bounds, inventory halt rule, alpha plugin swapping | **IMPORTANT** |
| [`tests/test_strategy_framework.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_strategy_framework.cpp) | C++ Strategy Tests | Verification | Full step lifecycle, gateway fills, shutdown order cancel | **IMPORTANT** |
| [`tests/test_tick_replayer.cpp`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_tick_replayer.cpp) | C++ Ingestion Tests | Verification | SPSC lock-free queue, chronological sorting, gap detection | **IMPORTANT** |
| [`tests/test_mm_env.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_mm_env.py) | Python Env Tests | Verification | Gym spaces, step dynamics, Hypothesis property tests | **IMPORTANT** |
| [`tests/test_ppo_agent.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_ppo_agent.py) | Python PPO Tests | Verification | ActorCritic forward pass, clipped surrogate loss properties | **IMPORTANT** |
| [`tests/test_trajectory_buffer.py`](file:///c:/Users/saura/Coding/SSProjects/LOB.0/tests/test_trajectory_buffer.py) | Python Buffer Tests | Verification | GAE-lambda reverse calculation, memory reuse without leakage | **IMPORTANT** |
