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

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
set(BLIS_YACONV_INSTALL
    ""
    CACHE PATH "Path to Blis installation that contains the yaconv addon")

# Check if BLIS exists
if(NOT EXISTS "${BLIS_INSTALL}")
  message(FATAL_ERROR "BLIS_INSTALL is not a valid path: ${BLIS_INSTALL}")
endif()

# Check if BLIS Yaconv exists
if(NOT EXISTS "${BLIS_YACONV_INSTALL}")
  message(
    FATAL_ERROR
      "BLIS_YACONV_INSTALL is not a valid path: ${BLIS_YACONV_INSTALL}")
endif()

# Add dependencies
find_package(benchmark REQUIRED PATHS ${BENCHMARK_INSTALL})
find_package(Torch REQUIRED PATHS ${TORCH_INSTALL})
find_package(OpenMP REQUIRED)
find_package(DNNL REQUIRED)

option(USE_MKL "Use Intel MKL as BLAS. Default is to use Blis" OFF)
option(BUILD_ZCONV_BLIS "If MKL is being used as blas, enable this option to build an additional zconv binary that uses BLIS" OFF)
option(USE_MKL_JIT "Use MKL to jit the main gemm colled by Zero Copy Conv." OFF)
option(USE_FIXED_ITERATIONS "Run each benchmark a fixed number of iterations" OFF)

if(NOT USE_MKL AND USE_MKL_JIT)
  message(FATAL_ERROR "USE_MKL must be set to ON when USE_MKL_JIT is ON.")
endif()

if(NOT USE_MKL AND BUILD_ZCONV_BLIS)
  message(FATAL_ERROR "USE_MKL must be set to ON when BUILD_ZCONV_BLIS is ON.")
endif()

set(blas_includes "")
set(blas_libraries "")
set(blas_define "")

if(USE_MKL)
  # Find MKL package
  find_package(MKL REQUIRED)
  set(blas_libraries "MKL::MKL")
  set(blas_define "-DUSE_MKL")
  message("Using MKL for BLAS")
  if(USE_MKL_JIT)
    set(blas_define "${blas_define} -DUSE_MKL_JIT")
    message("Using MKL jit")
  endif()
else()
  # Find BLIS library and headers
  find_path(BLIS_INCLUDE_DIR blis PATHS "${BLIS_INSTALL}/include")
  find_library(
    BLIS_LIBRARY
    NAMES blis
    PATHS "${BLIS_INSTALL}/lib")

  if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY)
    message(
      FATAL_ERROR
        "BLIS not found. Please specify the BLIS installation path using -DBLIS_INSTALL=<path>. This path should contain 'include' and 'lib' directories."
    )
  endif()

  set(blas_includes "${BLIS_INCLUDE_DIR}")
  set(blas_libraries "${BLIS_LIBRARY}")
  message("Using BLIS for BLAS")
endif()

set(run_mode_define "")
if(USE_FIXED_ITERATIONS)
  set(run_mode_define "-DFIXED_ITERATIONS")
  message("Using a fixed number of iterations")
endif()

# Find BLIS library and headers, there is no automatic package
find_path(BLIS_YACONV_INCLUDE_DIR blis PATHS "${BLIS_YACONV_INSTALL}/include")
find_library(
  BLIS_YACONV_LIBRARY_DIR
  NAMES blis
  PATHS "${BLIS_YACONV_INSTALL}/lib")
if(NOT BLIS_YACONV_INCLUDE_DIR OR NOT BLIS_YACONV_LIBRARY_DIR)
  message(
    FATAL_ERROR
      "BLIS not found. Please specify the Blis installation path using -DBLIS_YACONV_INSTALL=<path>. This path should contain 'include' and 'lib' directories."
  )
endif()

# Naive convolution ------------------------------------------------------------
add_executable(benchmark_naive src/driver.cpp src/kernel_conv_naive.c
                               src/utils.cpp)
target_include_directories(benchmark_naive
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_naive benchmark::benchmark)
set_target_properties(benchmark_naive PROPERTIES COMPILE_FLAGS "-DNAIVE ${run_mode_define}")
install(TARGETS benchmark_naive DESTINATION bin)

# Im2col convolution -----------------------------------------------------------
add_executable(benchmark_im2col src/driver.cpp src/kernel_conv_im2col.c
                                src/utils.cpp)
target_include_directories(
  benchmark_im2col PRIVATE ${PROJECT_SOURCE_DIR}/include ${blas_includes})
target_link_libraries(benchmark_im2col benchmark::benchmark ${blas_libraries})
set_target_properties(benchmark_im2col PROPERTIES COMPILE_FLAGS
                                                  "-DIM2COL ${blas_define} ${run_mode_define}")
install(TARGETS benchmark_im2col DESTINATION bin)

# Yaconv convolution -----------------------------------------------------------
add_executable(benchmark_yaconv src/driver.cpp src/kernel_conv_yaconv.c
                                src/utils.cpp)
target_include_directories(
  benchmark_yaconv
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${BLIS_YACONV_INCLUDE_DIR})
target_link_libraries(benchmark_yaconv benchmark::benchmark
                      ${BLIS_YACONV_LIBRARY_DIR})
set_target_properties(benchmark_yaconv PROPERTIES COMPILE_FLAGS "-DYACONV ${run_mode_define}")
install(TARGETS benchmark_yaconv DESTINATION bin)

# Zero-copy convolution --------------------------------------------------------
add_executable(benchmark_zero_copy src/driver.cpp src/kernel_conv_zero_copy.c
                                   src/utils.cpp)
target_include_directories(
  benchmark_zero_copy PRIVATE ${PROJECT_SOURCE_DIR}/include ${blas_includes})
target_link_libraries(benchmark_zero_copy benchmark::benchmark
                      ${blas_libraries} OpenMP::OpenMP_C)
set_target_properties(benchmark_zero_copy
                      PROPERTIES COMPILE_FLAGS "-DZERO_COPY ${blas_define} ${run_mode_define}")
install(TARGETS benchmark_zero_copy DESTINATION bin)

# OneDNN convolution using any layout ------------------------------------------
add_executable(benchmark_onednn_any src/driver_onednn_any.cpp src/utils.cpp)
target_include_directories(benchmark_onednn_any
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_onednn_any benchmark::benchmark DNNL::dnnl)
set_target_properties(benchmark_onednn_any PROPERTIES COMPILE_FLAGS "${run_mode_define}")
install(TARGETS benchmark_onednn_any DESTINATION bin)

# Libtorch convolution ---------------------------------------------------------
add_executable(benchmark_libtorch src/driver_libtorch.cpp src/utils.cpp)
target_include_directories(benchmark_libtorch
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_libtorch benchmark::benchmark
                      "${TORCH_LIBRARIES}")
set_target_properties(benchmark_libtorch PROPERTIES COMPILE_FLAGS
                                                    "${TORCH_CXX_FLAGS} ${run_mode_define}")
install(TARGETS benchmark_libtorch DESTINATION bin)

# Libtorch convolution using ZeroCopy2D ----------------------------------------
add_executable(benchmark_libtorch_zerocopy src/driver_libtorch.cpp
                                           src/utils.cpp)
target_include_directories(benchmark_libtorch_zerocopy
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(benchmark_libtorch_zerocopy benchmark::benchmark
                      "${TORCH_LIBRARIES}")
set_target_properties(benchmark_libtorch_zerocopy
                      PROPERTIES COMPILE_FLAGS "-DZERO_COPY ${TORCH_CXX_FLAGS} ${run_mode_define}")
install(TARGETS benchmark_libtorch_zerocopy DESTINATION bin)

# Correctness verification -----------------------------------------------------
if(NOT USE_MKL)
  # If not using MKL, BLIS yaconv already contains everything needed
  set(blas_includes "")
  set(blas_libraries "")
endif()

add_executable(
  correctness
  src/verify_correctness.cpp src/kernel_conv_im2col.c src/kernel_conv_yaconv.c
  src/kernel_conv_zero_copy.c src/utils.cpp)
target_include_directories(
  correctness PRIVATE ${PROJECT_SOURCE_DIR}/include ${blas_includes}
                      ${BLIS_YACONV_INCLUDE_DIR})

target_link_libraries(
  correctness benchmark::benchmark ${blas_libraries} "${TORCH_LIBRARIES}"
  ${BLIS_YACONV_LIBRARY_DIR} OpenMP::OpenMP_C)
set_target_properties(
  correctness PROPERTIES COMPILE_FLAGS "${TORCH_CXX_FLAGS} ${blas_define}")
install(TARGETS correctness DESTINATION bin)

# Zconv convolution using BLIS ------------------------------------------------
if(BUILD_ZCONV_BLIS)
  # Find BLIS library and headers
  find_path(BLIS_INCLUDE_DIR blis PATHS "${BLIS_INSTALL}/include")
  find_library(
    BLIS_LIBRARY
    NAMES blis
    PATHS "${BLIS_INSTALL}/lib")

  if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY)
    message(
      FATAL_ERROR
        "BLIS not found. Please specify the BLIS installation path using -DBLIS_INSTALL=<path>. This path should contain 'include' and 'lib' directories."
    )
  endif()

  set(blas_includes "${BLIS_INCLUDE_DIR}")
  set(blas_libraries "${BLIS_LIBRARY}")

  add_executable(benchmark_zero_copy_blis src/driver.cpp src/kernel_conv_zero_copy.c
                                     src/utils.cpp)
  target_include_directories(
    benchmark_zero_copy_blis PRIVATE ${PROJECT_SOURCE_DIR}/include ${BLIS_INCLUDE_DIR})
  target_link_libraries(benchmark_zero_copy_blis benchmark::benchmark
                        ${BLIS_LIBRARY} OpenMP::OpenMP_C)
  set_target_properties(benchmark_zero_copy_blis
                        PROPERTIES COMPILE_FLAGS "-DZERO_COPY ${run_mode_define}")
  install(TARGETS benchmark_zero_copy_blis DESTINATION bin)
endif()

