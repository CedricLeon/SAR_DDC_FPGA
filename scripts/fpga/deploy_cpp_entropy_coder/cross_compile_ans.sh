#!/bin/bash
set -e

# Ensure we are at the project root
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." >/dev/null 2>&1 && pwd )"
cd "$ROOT_DIR"

# 1. Prepare the package directory using the existing script
echo "[Host] bundling C++ sources..."
bash scripts/fpga/deploy_cpp_entropy_coder/setup_fpga_cpp.sh

# The setup script creates fpga_cpp_pkg in the root
PKG_DIR="fpga_cpp_pkg"
if [ ! -d "$PKG_DIR" ]; then
    echo "Error: Directory $PKG_DIR not found. Check setup_fpga_cpp.sh."
    exit 1
fi
ABS_PKG_DIR="$ROOT_DIR/$PKG_DIR"

echo "[Host] Starting Cross-Compilation Container (ARM64 Python 3.11)..."
echo "       This may take a few minutes if the image needs to be downloaded."

# 2. Run Docker container to compile
# We use python:3.11-slim-bullseye (Debian 11) instead of latest (Debian 12+)
# to link against an older GLIBCXX/CXXABI, ensuring compatibility with older target systems.
# If this is still too new, we might need to go even older (e.g. Ubuntu 20.04 + deadsnakes PPA).
docker run --rm \
    --platform linux/arm64 \
    -v "$ABS_PKG_DIR":/workspace \
    -w /workspace \
    python:3.11-slim-bullseye \
    /bin/bash -c "
        echo '[Container] Installing dependencies...' && \
        apt-get update >/dev/null && \
        apt-get install -y --no-install-recommends build-essential >/dev/null && \
        pip install pybind11 >/dev/null && \
        echo '[Container] Compiling ans.so...' && \
        make
    "

echo "[Host] Compilation finished."
echo "       Artifact: $PKG_DIR/ans.*.so"
