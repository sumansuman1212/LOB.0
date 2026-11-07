# Installation & Build Guide

This document provides step-by-step instructions for setting up the development environment, compiling the C++ execution engine, running Python reinforcement learning workflows, running inside GitHub Codespaces, and troubleshooting common platform issues.

---

## 1. Quick Start with GitHub Codespaces (Recommended)

LOB.0 provides a turnkey **GitHub Codespaces** development environment (`.devcontainer/devcontainer.json`). This runs an Ubuntu 22.04 container pre-configured with:
- GCC 11+ (C++17 standard)
- CMake 3.22+ and Ninja build system
- Python 3.11 with virtual environment and dependencies
- Pre-downloaded Linux CPU LibTorch (cxx11 ABI) in `/opt/libtorch`

To launch:
1. Open the repository on GitHub.
2. Click the green **Code** button -> **Codespaces** tab -> **Create codespace on main**.
3. Once the Codespace boots, open a terminal:
   ```bash
   source .venv/bin/activate
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
   cmake --build build --target mm_tests mm_backtest verify_native_inference
   ./build/tests/mm_tests
   ./build/src/cpp/verify_native_inference
   pytest -v tests/
   ```

---

## 2. Local Prerequisites

If developing on your local workstation:
- **Git** (version 2.25+)
- **Python** (version 3.10, 3.11, 3.12, or 3.13)
- **CMake** (version 3.22+)
- **A Modern 64-bit C++17 Compiler:**
  - **Linux:** GCC 11+ or Clang 14+ (`sudo apt-get install build-essential cmake ninja-build g++`)
  - **Windows:** Microsoft Visual C++ (MSVC 2022 v143+ with C++ Desktop Workload) or MSYS2 MinGW-w64 (UCRT64)
  - **macOS:** Apple Clang (Xcode Command Line Tools 14+)

---

## 3. Python Environment Setup

We recommend using an isolated virtual environment (`.venv`):

```bash
# 1. Clone repository
git clone https://github.com/sumansuman1212/LOB.0.git
cd LOB.0

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Linux / macOS / Codespaces:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd.exe:
.venv\Scripts\activate.bat

# 4. Install dependencies in editable mode
pip install --upgrade pip
pip install -e .
pip install pytest hypothesis
```

Verify Python tests:
```bash
pytest -v tests/
```

---

## 4. C++ Build Instructions (CMake)

The C++ build system is configured via CMake. It features **zero mandatory external dependencies** for compiling and running the core test suite (`mm_tests`), thanks to the self-contained test harness header in `include/catch2/catch_test_macros.hpp`.

### 4.1 Standalone Build (Analytical Fallback)

If LibTorch is not present, CMake automatically builds with standalone analytical fallback:

#### On Linux / macOS
```bash
# Configure Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile all targets
cmake --build build --target mm_tests mm_backtest verify_native_inference

# Run test suite
./build/tests/mm_tests

# Run native inference verification (analytical fallback mode)
./build/src/cpp/verify_native_inference
```

#### On Windows with MSVC (Visual Studio 2022)
Open "x64 Native Tools Command Prompt for VS 2022" or PowerShell:
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target mm_tests mm_backtest verify_native_inference
.\build\tests\Release\mm_tests.exe
.\build\src\cpp\Release\verify_native_inference.exe
```

---

## 5. Building with LibTorch (Native Neural Inference)

To enable native C++ forward pass of the trained TorchScript model (`models/ppo_actor.pt`):

### Option A: Using Standalone LibTorch C++ Download (Linux / Windows)
1. Download **LibTorch C++ (CPU)** from [pytorch.org](https://pytorch.org/).
   - Linux: `libtorch-cxx11-abi-shared-with-deps-2.2.*.zip`
   - Windows: `libtorch-win-shared-with-deps-2.2.*.zip`
2. Extract to a directory (e.g., `/opt/libtorch` or `C:\libtorch`).
3. Pass `-DCMAKE_PREFIX_PATH`:
   ```bash
   cmake -B build -DCMAKE_PREFIX_PATH="/opt/libtorch" -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target mm_tests mm_backtest verify_native_inference
   ```

### Option B: Using Python's PyTorch Installation
If you already installed PyTorch in your Python virtual environment, you can point CMake to PyTorch's bundled CMake config:
```bash
# Linux / macOS:
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')

# Windows PowerShell:
$torch_path = python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$torch_path
```

When LibTorch is found, CMake prints:
```
-- Torch found: <version>
```
and defines `USE_TORCH`, linking `${TORCH_LIBRARIES}`.

---

## 6. Troubleshooting: Windows 11 DLL Relocation (0xC0000142)

If you encounter:
`Program 'mm_tests.exe' failed to run: {Illegal System DLL Relocation} ... The application will not run properly.`

- **Diagnosis:** Windows 11 enforces Address Space Layout Randomization (ASLR). Obsolete 32-bit MinGW toolchains (e.g. from `C:\MinGW\bin`) emit fixed-base binaries without `/DYNAMICBASE`, causing DLL collisions at runtime.
- **Solution:** Use a 64-bit compiler:
  - Visual Studio 2022 Build Tools (MSVC `cl.exe` x64).
  - Modern MSYS2 UCRT64 (`C:\msys64\ucrt64\bin\g++.exe`).
  - Or use **GitHub Codespaces** to develop in a pristine Linux environment.
