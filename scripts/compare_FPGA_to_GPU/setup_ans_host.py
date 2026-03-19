import os

import pybind11
from setuptools import Extension, setup

sources = [
    # Adjust relative path from scripts/compare_FPGA_to_GPU/ to CompressAI/compressai/cpp_exts/rans/rans_interface.cpp
    "../../CompressAI/compressai/cpp_exts/rans/rans_interface.cpp"
]

include_dirs = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
    "../../CompressAI/compressai/cpp_exts/rans",
    "../../CompressAI/third_party/ryg_rans",
]

ext_modules = [
    Extension(
        "ans",
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="ans",
    version="0.1",
    ext_modules=ext_modules,
)
