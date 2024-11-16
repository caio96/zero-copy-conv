cmake_minimum_required(VERSION 3.10)
project(ConvBenchmark)

# Create compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native") # -g -gdwarf-4"
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native") # -g -gdwarf-4"

# Set Install Paths
set(BENCHMARK_INSTALL
    ""
    CACHE PATH "Path to Google Benchmark installation")
set(TORCH_INSTALL
    ""
    CACHE PATH "Path to LibTorch installation")
set(BLIS_INSTALL
    ""
    CACHE PATH "Path to Blis installation")

# Check if BLIS exists, the other two may be installed globally
if(NOT EXISTS "${BLIS_INSTALL}")
  message(FATAL_ERROR "BLIS_INSTALL is not a valid path: ${BLIS_INSTALL}")
endif()

# Add dependencies
find_package(benchmark REQUIRED PATHS ${BENCHMARK_INSTALL})
find_package(Torch REQUIRED PATHS ${TORCH_INSTALL})
find_package(OpenMP REQUIRED)
find_package(DNNL REQUIRED)

# Find BLIS library and headers, there is no automatic package
find_path(BLIS_INCLUDE_DIR blis PATHS "${BLIS_INSTALL}/include")
find_library(
  BLIS_LIBRARY_DIR
  NAMES blis
  PATHS "${BLIS_INSTALL}/lib")
if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY_DIR)
  message(
    FATAL_ERROR
      "BLIS not found. Please specify the Blis installation path using -DBLIS_INSTALL=<path>. This path should contain 'include' and 'lib' directories."
  )
endif()

# Naive convolution
add_executable(benchmark_naive src/driver.cpp src/kernel_conv_naive.c
                               src/utils.cpp)
target_include_directories(benchmark_naive
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_naive benchmark::benchmark)
set_target_properties(benchmark_naive PROPERTIES COMPILE_FLAGS "-DNAIVE")

# Im2col convolution
add_executable(benchmark_im2col src/driver.cpp src/kernel_conv_im2col.c
                                src/utils.cpp)
target_include_directories(
  benchmark_im2col
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(benchmark_im2col benchmark::benchmark ${BLIS_LIBRARY_DIR})
set_target_properties(benchmark_im2col PROPERTIES COMPILE_FLAGS "-DIM2COL")

# Yaconv convolution
add_executable(benchmark_yaconv src/driver.cpp src/kernel_conv_yaconv.c
                                src/utils.cpp)
target_include_directories(
  benchmark_yaconv
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(benchmark_yaconv benchmark::benchmark ${BLIS_LIBRARY_DIR})
set_target_properties(benchmark_yaconv PROPERTIES COMPILE_FLAGS "-DYACONV")

# Zero-copy convolution
add_executable(benchmark_zero_copy src/driver.cpp src/kernel_conv_zero_copy.c
                                   src/utils.cpp)
target_include_directories(
  benchmark_zero_copy
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(benchmark_zero_copy benchmark::benchmark
                      ${BLIS_LIBRARY_DIR} OpenMP::OpenMP_C)
set_target_properties(benchmark_zero_copy PROPERTIES COMPILE_FLAGS
                                                     "-DZERO_COPY")

# OneDNN convolution
add_executable(benchmark_onednn src/driver_onednn.cpp src/utils.cpp)
target_include_directories(benchmark_onednn
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_onednn benchmark::benchmark DNNL::dnnl)

# Libtorch convolution using channel last layout
add_executable(benchmark_libtorch src/driver_libtorch.cpp src/utils.cpp)
target_include_directories(benchmark_libtorch
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_libtorch benchmark::benchmark
                      "${TORCH_LIBRARIES}")
set_target_properties(benchmark_libtorch PROPERTIES COMPILE_FLAGS
                                                    "${TORCH_CXX_FLAGS}")

# Correctness verification
add_executable(
  correctness
  src/verify_correctness.cpp src/kernel_conv_im2col.c src/kernel_conv_yaconv.c
  src/kernel_conv_zero_copy.c src/utils.cpp)
target_include_directories(
  correctness
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${BLIS_INCLUDE_DIR})
target_link_libraries(correctness benchmark::benchmark ${BLIS_LIBRARY_DIR}
                      "${TORCH_LIBRARIES}" OpenMP::OpenMP_C)
set_target_properties(correctness PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS}")
