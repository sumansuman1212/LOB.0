# Building & Running LOB.0

This quick reference outlines how to compile and run **LOB.0**. For comprehensive details and troubleshooting, see [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

---

## 1. Quick Start: GitHub Codespaces (Browser)

LOB.0 provides pre-configured devcontainers for GitHub Codespaces:
1. Open this repository on GitHub.
2. Click **Code** -> **Codespaces** -> **Create codespace on main**.
3. In the Codespace terminal:
   ```bash
   source .venv/bin/activate
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch
   cmake --build build --target mm_tests mm_backtest verify_native_inference
   ./build/tests/mm_tests
   ./build/src/cpp/verify_native_inference
   pytest -v tests/
   ```

---

## 2. Local C++ Build (Linux / macOS)

```bash
# Configure build (standalone mode with zero external dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile all targets
cmake --build build --target mm_tests mm_backtest verify_native_inference

# Run test suite and native inference verification
./build/tests/mm_tests
./build/src/cpp/verify_native_inference
```

*Note on LibTorch:* To enable native neural inference, pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch` or point to Python's PyTorch installation via `$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')`.

---

## 3. Local C++ Build (Windows MSVC 2022)

From an "x64 Native Tools Command Prompt for VS 2022" or PowerShell:
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target mm_tests mm_backtest verify_native_inference
.\build\tests\Release\mm_tests.exe
.\build\src\cpp\Release\verify_native_inference.exe
```

---

## 4. Python RL Test Suite

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and test dependencies
pip install -e .
pip install pytest hypothesis

# Run tests
pytest -v tests/
```

