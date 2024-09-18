cmake_minimum_required(VERSION 3.10)
project(ConvBenchmark)

# Create compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Google Benchmark as a dependency
find_package(benchmark REQUIRED PATHS "/home/csalvado/git/benchmark-install")

# Find BLIS library and headers
find_path(BLIS_INCLUDE_DIR blis PATHS "/home/csalvado/tensor-layout/blis-install/include")
find_library(BLIS_LIBRARY NAMES blis PATHS "/home/csalvado/tensor-layout/blis-install/lib")

if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY)
    message(FATAL_ERROR "BLIS library not found. Please ensure it is installed.")
endif()

add_executable(benchmark_naive
    src/driver_conv_naive.cpp
    src/kernel_conv_naive.c
)
target_include_directories(benchmark_naive PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_naive benchmark::benchmark pthread)
set_target_properties(benchmark_naive PROPERTIES COMPILE_FLAGS "-O3 -march=native")

add_executable(benchmark_im2col
    src/driver_conv_im2col.cpp
    src/kernel_conv_im2col.c
)
target_include_directories(benchmark_im2col PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(benchmark_im2col PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(benchmark_im2col benchmark::benchmark pthread ${BLIS_LIBRARY})
set_target_properties(benchmark_im2col PROPERTIES COMPILE_FLAGS "-O3 -march=native")

add_executable(benchmark_yaconv
    src/driver_conv_yaconv.cpp
    src/kernel_conv_yaconv.c
)
target_include_directories(benchmark_yaconv PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(benchmark_yaconv PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(benchmark_yaconv benchmark::benchmark pthread ${BLIS_LIBRARY})
set_target_properties(benchmark_yaconv PROPERTIES COMPILE_FLAGS "-O3 -march=native -g -gdwarf-4")

add_executable(correctness
    src/verify_correctness.cpp
    src/kernel_conv_im2col.c
    src/kernel_conv_naive.c
)
target_include_directories(correctness PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(correctness PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(correctness benchmark::benchmark pthread ${BLIS_LIBRARY})
set_target_properties(correctness PROPERTIES COMPILE_FLAGS "-O3 -march=native")
