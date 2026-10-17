# packetshield-agentic — CMakeLists.txt
#
# Builds the PacketShield-Agentic high-throughput GPU deep packet inspection
# engine (data plane: C++20 + CUDA).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#   ./build/packetshield_bench
#
# If a CUDA toolkit is present, the GPU pipeline (src/gpu_matcher.cu) is
# compiled with nvcc and HAS_CUDA is defined. Otherwise the project builds
# CPU-only so the benchmark and tests still run on machines without a GPU.
#
# The agentic control plane lives in agent/ (Python 3.11+, FastAPI,
# LangGraph) and is deployed independently: pip install -r agent/requirements.txt

cmake_minimum_required(VERSION 3.24)
project(PacketShieldAgentic VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Optimization flags ----------------------------------------------------
add_compile_options(-O3 -Wall -Wextra -Wpedantic)

# ---- Optional CUDA ----------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  # CUDA language level: C++17 device code keeps compatibility with older
  # toolkits while host code uses C++20.
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wall>)
  set(HAVE_CUDA TRUE)
  message(STATUS "CUDA compiler found: building with GPU support")
else()
  set(HAVE_CUDA FALSE)
  message(STATUS "No CUDA compiler found: building CPU-only")
endif()

# ---- Library ----------------------------------------------------------------
add_library(packetshield STATIC
  src/cpu_matcher.cpp
  $<$<BOOL:${HAVE_CUDA}>:src/gpu_matcher.cu>
)
target_include_directories(packetshield PUBLIC include)
target_compile_definitions(packetshield PUBLIC
  $<$<BOOL:${HAVE_CUDA}>:HAS_CUDA>
)
set_target_properties(packetshield PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

# ---- Benchmark harness -------------------------------------------------------
add_executable(packetshield_bench src/main.cpp)
target_link_libraries(packetshield_bench PRIVATE packetshield)

# ---- Tests -------------------------------------------------------------------
enable_testing()
add_executable(test_matcher tests/test_matcher.cpp)
target_link_libraries(test_matcher PRIVATE packetshield)
add_test(NAME matcher_regression COMMAND test_matcher)
set_tests_properties(matcher_regression PROPERTIES
  TIMEOUT 300
  LABELS "regression;correctness;memory-safety;agentic-contract"
)

# ---- Summary ------------------------------------------------------------------
message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard      : ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA support      : ${HAVE_CUDA}")
