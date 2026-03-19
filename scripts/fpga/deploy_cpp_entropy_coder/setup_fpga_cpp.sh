#!/bin/bash
set -e

# Root_dir must be DDC_FPGA/ but sometimes we run from the parent directory, so we ensure we're in the right place
ROOT_DIR="DDC_FPGA"
if [ ! -d "$ROOT_DIR" ]; then
    ROOT_DIR="."
fi
cd $ROOT_DIR

# Configuration
PACKAGE_DIR="fpga_cpp_pkg"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR/include

echo "Packaging FPGA C++ environment..."

# 1. Copy C++ Source Files
cp CompressAI/compressai/cpp_exts/rans/rans_interface.cpp $PACKAGE_DIR/
cp CompressAI/compressai/cpp_exts/rans/rans_interface.hpp $PACKAGE_DIR/
cp CompressAI/third_party/ryg_rans/rans64.h $PACKAGE_DIR/

# 2. Copy Build Files
cp scripts/fpga/deploy_cpp_entropy_coder/Makefile_ans $PACKAGE_DIR/Makefile

#2.5 Copy verification script
cp scripts/fpga/deploy_cpp_entropy_coder/verify_ans_on_target.py $PACKAGE_DIR/

# 3. Try to bundle PyBind11 headers (Robustness for Target)
echo "Attempting to bundle PyBind11 headers..."
PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())" 2>/dev/null || echo "")

if [ -n "$PYBIND_INCLUDE" ]; then
    echo "Found PyBind11 at $PYBIND_INCLUDE. Copying..."
    # Copy the contents (usually 'pybind11' folder) to our include dir
    # get_include() returns '.../site-packages/pybind11/include' which contains 'pybind11' folder
    cp -r $PYBIND_INCLUDE/* $PACKAGE_DIR/include/
    echo "PyBind11 bundled successfully."
else
    echo "WARNING: PyBind11 not found in current Python environment."
    echo "You must ensure 'pybind11' is installed on the Target or copy the headers manually."
    echo "Target command: pip3 install pybind11"
fi

# 4. Create Tarball
echo "Package created: $PACKAGE_DIR"
echo ""
echo "Setup Instructions (Run Once on FPGA):"
echo "1. Copy '$PACKAGE_DIR' to the FPGA (e.g., scp -r $PACKAGE_DIR root@10.0.0.1:~/SAR_DDC/)."
echo "2. On FPGA: cd $PACKAGE_DIR && make"
echo "3. On FPGA: Move 'ans*.so' to a shared library folder (e.g. ~/SAR_DDC/)"
echo "   OR ensure your inference script adds this folder to sys.path."
echo ""
