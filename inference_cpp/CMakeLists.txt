cmake_minimum_required(VERSION 3.16)
project(inference_hybrid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(HAVE_DPU "Build with Vitis-AI VART/XIR DPU support (requires board libs)" ON)

# Opt-in per-patch entropy-encode sub-stage profiler (symbuild / lookup / flush,
# split EB vs GC). Off by default: zero overhead, byte-identical bitstream. See
# src/rans/rans_profile.hpp and docs/onboard_pipeline.md (N6).
option(DDC_RANS_PROFILE "Instrument the rANS encode path with sub-stage timers" OFF)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-O3 -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# OpenCV — required on the ZCU102 (4.5.2 with opencv_quality for SSIM).
# On a dev host without OpenCV, build with WITHOUT_OPENCV=ON to stub out SSIM.
# ---------------------------------------------------------------------------
option(WITHOUT_OPENCV "Build without OpenCV (stubs out SSIM; for syntax checking only)" OFF)
if(NOT WITHOUT_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc quality)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "WITHOUT_OPENCV=ON — SSIM will return 0.0 (stub)")
endif()

# ---------------------------------------------------------------------------
# nlohmann/json — system-wide on ZCU102 (/usr/include/nlohmann/json.hpp).
# Required only for HAVE_DPU=ON (dpu_runners.cpp + inference_runner.cpp).
# On a dev host without it, install: sudo apt install nlohmann-json3-dev
# ---------------------------------------------------------------------------
if(HAVE_DPU)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include
        REQUIRED)
    message(STATUS "nlohmann/json found at: ${NLOHMANN_JSON_INCLUDE_DIR}")
else()
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include)
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "nlohmann/json found at: ${NLOHMANN_JSON_INCLUDE_DIR}")
    else()
        set(NLOHMANN_JSON_INCLUDE_DIR "")
        message(STATUS "nlohmann/json not found — inference_runner not built in this config")
    endif()
endif()

# ---------------------------------------------------------------------------
# rANS static library (pybind11-free fork)
# ---------------------------------------------------------------------------
add_library(rans_cxx STATIC
    src/rans/rans_interface_cxx.cpp
    src/rans/rans_profile.cpp
)
target_include_directories(rans_cxx PUBLIC src/rans)
# rans_interface_cxx has no external dependencies beyond the bundled rans64.h

# When enabled, define DDC_RANS_PROFILE PUBLIC so it reaches every consumer that
# uses the profiler macros (entropy_models.cpp via inference_lib links rans_cxx).
if(DDC_RANS_PROFILE)
    target_compile_definitions(rans_cxx PUBLIC DDC_RANS_PROFILE)
    message(STATUS "DDC_RANS_PROFILE=ON — rANS encode sub-stage timers compiled in")
endif()

# ---------------------------------------------------------------------------
# Core inference library
# Sources that do NOT require DPU are always included.
# inference_runner.cpp and dpu_runners.cpp need nlohmann/json (board only).
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    src/npy_io.cpp
    src/metrics.cpp
    src/entropy_models.cpp
)

if(NLOHMANN_JSON_INCLUDE_DIR)
    list(APPEND CORE_SOURCES src/inference_runner.cpp)
endif()

if(HAVE_DPU)
    list(APPEND CORE_SOURCES src/dpu_runners.cpp)
endif()

add_library(inference_lib STATIC ${CORE_SOURCES})
target_include_directories(inference_lib PUBLIC src include)
if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(inference_lib PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(NOT WITHOUT_OPENCV)
    target_link_libraries(inference_lib PUBLIC rans_cxx ${OpenCV_LIBS})
else()
    target_compile_definitions(inference_lib PUBLIC WITHOUT_OPENCV)
    target_link_libraries(inference_lib PUBLIC rans_cxx)
endif()

if(HAVE_DPU)
    target_compile_definitions(inference_lib PUBLIC HAVE_DPU)
    # VART/XIR are installed system-wide on the ZCU102 (PetaLinux)
    target_link_libraries(inference_lib PUBLIC vart-runner xir)
endif()

# ---------------------------------------------------------------------------
# Main executable — only built when nlohmann/json is available
# ---------------------------------------------------------------------------
if(NLOHMANN_JSON_INCLUDE_DIR)
    add_executable(inference_hybrid src/main.cpp)
    target_include_directories(inference_hybrid PRIVATE src include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(inference_hybrid PRIVATE inference_lib)
else()
    message(STATUS "Skipping inference_hybrid target (nlohmann/json not found)")
endif()

# ---------------------------------------------------------------------------
# benchmark_hardware — per-stage latency and throughput profiler
# Requires nlohmann/json (same as inference_hybrid).
# ---------------------------------------------------------------------------
if(NLOHMANN_JSON_INCLUDE_DIR)
    set(BENCHMARK_SOURCES
        src/benchmark/bench_pipeline.cpp
        src/benchmark/bench_configs.cpp
        src/benchmark/power_sampler.cpp
        src/benchmark/main_benchmark.cpp
    )
    add_executable(benchmark_hardware ${BENCHMARK_SOURCES})
    target_include_directories(benchmark_hardware PRIVATE
        src src/benchmark include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(benchmark_hardware PRIVATE inference_lib)
else()
    message(STATUS "Skipping benchmark_hardware target (nlohmann/json not found)")
endif()

# ---------------------------------------------------------------------------
# Unit tests (rANS roundtrip — no DPU required)
# ---------------------------------------------------------------------------
add_executable(test_rans tests/test_rans_roundtrip.cpp)
target_include_directories(test_rans PRIVATE src include)
target_link_libraries(test_rans PRIVATE rans_cxx)

# .ddc container codec (header-only, no DPU / no deps) — byte-compat with src/utils/ddc_format.py
add_executable(test_ddc_io tests/test_ddc_io.cpp)
target_include_directories(test_ddc_io PRIVATE src)

# tile_source windowed SD reader (header-only, no DPU / no deps)
add_executable(test_tile_source tests/test_tile_source.cpp)
target_include_directories(test_tile_source PRIVATE src)

# ---------------------------------------------------------------------------
# stream_pipeline — sequential streaming compressor (tile -> .ddc). New binary,
# reuses BenchPipeline stages (bench_pipeline.cpp) + inference_lib. Needs nlohmann/json.
# ---------------------------------------------------------------------------
if(NLOHMANN_JSON_INCLUDE_DIR)
    add_executable(stream_pipeline
        src/stream/stream_pipeline.cpp
        src/stream/main_stream.cpp
        src/benchmark/bench_pipeline.cpp
        src/benchmark/power_sampler.cpp)
    target_include_directories(stream_pipeline PRIVATE
        src src/benchmark include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(stream_pipeline PRIVATE inference_lib)
else()
    message(STATUS "Skipping stream_pipeline target (nlohmann/json not found)")
endif()

# ---------------------------------------------------------------------------
# dump_latents — one-off debug tool: pre-entropy-coding y/z latents for named patches, for the
# INT8-vs-FP32 bpp-gap investigation (docs/tmp_jetson_orin_overnight.md). Not part of any pipeline;
# reuses BenchPipeline like stream_pipeline does, same build guard.
# ---------------------------------------------------------------------------
if(NLOHMANN_JSON_INCLUDE_DIR)
    add_executable(dump_latents
        src/tools/dump_latents.cpp
        src/benchmark/bench_pipeline.cpp)
    target_include_directories(dump_latents PRIVATE
        src src/benchmark include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(dump_latents PRIVATE inference_lib)
else()
    message(STATUS "Skipping dump_latents target (nlohmann/json not found)")
endif()
