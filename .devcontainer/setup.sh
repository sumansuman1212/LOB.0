#!/usr/bin/env bash
# =============================================================================
# LOB.0 — Codespaces Post-Create Setup Script
# Configures Linux C++17 build tools, Linux LibTorch, and Python RL harness.
# =============================================================================
set -e

echo "=== [1/4] Installing system dependencies ==="
sudo apt-get update -y
sudo apt-get install -y cmake ninja-build g++ wget unzip git

echo "=== [2/4] Setting up Python virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest hypothesis

echo "=== [3/4] Installing Linux CPU LibTorch (cxx11 ABI) ==="
LIBTORCH_TARGET="/opt/libtorch"
if [ ! -d "$LIBTORCH_TARGET" ]; then
    echo "Downloading LibTorch C++ distribution..."
    sudo mkdir -p /opt
    sudo wget -q --show-progress -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcpu.zip
    sudo unzip -q /tmp/libtorch.zip -d /opt
    sudo rm -f /tmp/libtorch.zip
    sudo chmod -R a+rX /opt/libtorch
fi

# Export LibTorch cmake prefix path for bash sessions
if ! grep -q "CMAKE_PREFIX_PATH=/opt/libtorch" ~/.bashrc; then
    echo 'export CMAKE_PREFIX_PATH="/opt/libtorch:${CMAKE_PREFIX_PATH}"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"' >> ~/.bashrc
fi

export CMAKE_PREFIX_PATH="/opt/libtorch:${CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}"

echo "=== [4/4] Environment Ready! ==="
echo ""
echo "To build and verify the C++ execution engine and LibTorch inference:"
echo "  source .venv/bin/activate"
echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch"
echo "  cmake --build build --target mm_tests mm_backtest verify_native_inference"
echo "  ./build/tests/mm_tests"
echo "  ./build/src/cpp/verify_native_inference"
echo ""
echo "To run the Python test suite:"
echo "  pytest -v tests/"
echo ""

