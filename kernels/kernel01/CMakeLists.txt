cmake_minimum_required(VERSION 3.10)
project(ConvBenchmark)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Google Benchmark as a dependency
find_package(benchmark REQUIRED PATHS "/home/csalvado/git/benchmark-install")

# Find BLIS library and headers
find_path(BLIS_INCLUDE_DIR blis.h PATHS "/home/csalvado/tensor-layout/yaconv-new/blis-install/include")
find_library(BLIS_LIBRARY NAMES blis PATHS "/home/csalvado/tensor-layout/yaconv-new/blis-install/lib")

if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY)
    message(FATAL_ERROR "BLIS library not found. Please ensure it is installed.")
endif()

# Add sources
set(SOURCE_FILES
    driver.cpp
    kernel_conv_naive.c
    kernel_conv_im2col.c
)

# Create executables
add_executable(benchmark_conv ${SOURCE_FILES})

# Link Google Benchmark and pthread library
target_link_libraries(benchmark_conv benchmark::benchmark pthread ${BLIS_LIBRARY})

# Include BLIS headers
target_include_directories(benchmark_conv PRIVATE ${BLIS_INCLUDE_DIR})

# Add optimization flags
set_target_properties(benchmark_conv PROPERTIES COMPILE_FLAGS "-O2")
