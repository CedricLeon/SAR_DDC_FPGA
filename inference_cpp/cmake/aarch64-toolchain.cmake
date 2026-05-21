# Cross-compilation toolchain for Xilinx ZCU102 (aarch64 PetaLinux, GCC 11)
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
#         -DSYSROOT=/path/to/petalinux/sysroot \
#         ..
#
# The sysroot can be extracted from the Vitis-AI Docker image or from the
# PetaLinux SDK. It must contain the board's /usr/include and /usr/lib.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Toolchain prefix — available inside the Vitis-AI Docker
set(CROSS_COMPILE aarch64-linux-gnu-)
set(CMAKE_C_COMPILER   ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++)
set(CMAKE_AR           ${CROSS_COMPILE}ar)
set(CMAKE_RANLIB       ${CROSS_COMPILE}ranlib)
set(CMAKE_STRIP        ${CROSS_COMPILE}strip)

# Sysroot — set via -DSYSROOT= or defaults to environment variable
if(NOT DEFINED SYSROOT AND DEFINED ENV{PETALINUX_SYSROOT})
    set(SYSROOT $ENV{PETALINUX_SYSROOT})
endif()

if(DEFINED SYSROOT)
    set(CMAKE_SYSROOT ${SYSROOT})
    set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()
