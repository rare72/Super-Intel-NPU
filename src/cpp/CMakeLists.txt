cmake_minimum_required(VERSION 3.10)
project(NewOfferingExecutive)

set(CMAKE_CXX_STANDARD 17)

# 1. FIND OPENVINO
# Handles the high-level NPU and iGPU orchestration
find_package(OpenVINO REQUIRED)

# 2. FIND LEVEL ZERO (ze_loader)
# This library is the "silicon bridge" for your Core Ultra NPU
find_library(LEVEL_ZERO_LIB NAMES ze_loader)
# Standard location for Xubuntu/Ubuntu headers
find_path(LEVEL_ZERO_INCLUDE NAMES ze_api.h PATH_SUFFIXES level_zero)

if (NOT LEVEL_ZERO_LIB)
    message(WARNING "Level Zero library not found. NPU discovery might fail if linking is required.")
endif()

# 3. CONFIGURE REALTIME (SHM) SUPPORT
# This links the -lrt flag required for POSIX shared memory (shm_open)
find_library(RT_LIB rt)

# 4. DEFINE EXECUTABLE
add_executable(executive_shard executive_shard.cpp)

# 5. INCLUDE PATHS
target_include_directories(executive_shard PRIVATE
    ${LEVEL_ZERO_INCLUDE}
    ${OpenVINO_INCLUDE_DIRS}
)

# 6. LINK LIBRARIES
target_link_libraries(executive_shard PRIVATE
    openvino::runtime
    ${LEVEL_ZERO_LIB}
    ${RT_LIB}      # Link -lrt for shared memory
    pthread        # Required for multi-threaded inference sync
)

# 7. CORE ULTRA OPTIMIZATIONS
# Native architecture flags to ensure the compiler uses AVX-VNNI / Intel-specific instructions
# Only check if compiler supports it to avoid errors in sandbox
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(executive_shard PRIVATE -march=native -O3)
endif()
