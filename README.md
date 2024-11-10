# Zero-Copy GEMM-Based Fast Convolution :zap:

This repository is designed to test multiple implementations of convolution. They are:

- Naive convolution: For loops and simple multiply accumulate
- Im2col convolution: Transforms the input image with im2col and executes convolution as a single GEMM call using the Blis library
- Yaconv convolution: Implementation from this [paper](https://dl.acm.org/doi/10.1145/3570305) with slight improvements. Defined in [blis-conv](https://github.com/caio96/blis-conv)
- Zero-Copy convolution: New convolution implementation, yet to be published
- LibTorch convolution: Pytorch Conv2D implementation using the C++ API

|           | Feature Layout | Weight Layout | Output Layout | Multithreading     |
| --------- | -------------- | ------------- | ------------- | ------------------ |
| Naive     | NHWC           | OIHW          | NHWC          | :x:                |
| Im2col    | NCHW           | OIHW          | NCHW          | :white_check_mark: |
| Yaconv    | NHWC           | OIHW          | NHWC          | :x:                |
| Zero-Copy | NHWC           | OIHW          | NWHC          | :white_check_mark: |
| LibTorch  | NCHW           | OIHW          | NCHW          | :white_check_mark: |

Note:

- The Im2col version depends on Blis to be multithreaded, which needs be enabled before compiling Blis.
- Yaconv only supports stride == 1, no grouping, and no dilation.

## Repository Structure

- `data`: Contains convolution layer parameters obtained with the script `util/timm_convolution_extraction`. Timm version 1.0.11 was used.
- `include`
- `src`
    - `driver` is the main file that uses Google Benchmark to call a convolution benchmark. The defines passed at compile time control which convolution method.
    - `kernel_conv_[method_name]` files have the implementation of each convolution method
    - `utils` has helper functions
    - `verify_correctness` calls all convolutions methods converting their output if necessary to compare them
- `scripts`:
    - `timm_convolution_extraction` is adapted from [ConvBench](https://github.com/LucasFernando-aes/ConvBench/) to extracts convolution layer parameters from multiple models into a csv
    - `filter_csv` allows filtering the csv containing convolution layer parameters
    - `benchmark_runner` controls running all convolutions in a generated csv to measure correctness or performance
    - `summarize_correctness` generates csv files that summarize correctness results based on the logs from `benchmark_runner`
    - `summarize_performance` generates csv files and graphs that summarize performance results based on the logs from `benchmark_runner`
    - `zero_copy_conv` is a simplified version of the Zero-Copy Convolution implemented in Python and compared against Pytorch

## Dependencies

- [Google Benchmark](https://github.com/google/benchmark)
- [Blis](https://github.com/flame/blis), but to use Yaconv, use the Blis fork in [blis-conv](https://github.com/caio96/blis-conv)
- [LibTorch](https://pytorch.org/cppdocs/installing.html)

### How to build Google Benchmark:

```sh
git clone https://github.com/google/benchmark.git
cd benchmark
git checkout v1.9.0
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
cmake --install build --config Release --prefix /path/to/benchmark-install
```

### How to build Blis:

```sh
git clone git@github.com:caio96/blis-conv.git
cd blis-conv
git checkout yaconv-update
./configure --prefix=/path/to/blis-install --enable-threading=openmp -a yaconv auto
make install -j4
```

The public [Blis repo](https://github.com/flame/blis) could also be used, but then the `benchmark_yaconv` executable would need to be removed from the CMakeLists.txt file.

### How to get LibTorch:

- Go to [link](https://pytorch.org/get-started/locally/)
- Select Package as "LibTorch", Language as "C++/Java", Compute platform as "CPU"
- Download the cxx11 ABI version and unzip it

Or run:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip
unzip libtorch.zip -d .
```

## Building this repo

```sh
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang                         \
      -DCMAKE_CXX_COMPILER=clang++                     \
      -DBENCHMARK_INSTALL="/path/to/benchmark-install" \
      -DTORCH_INSTALL="/path/to/libtorch"              \
      -DBLIS_INSTALL="/path/to/blis-install"           \
      ..
```

## Running Benchmarks

After building, the `this-repo/build/bin/` directory will contain one executable per convolution method with the name `benchmark_[method_name]`.
The executables can be run with `--help` to show the parameters they take. If run with no parameters, a default configuration is run.

### Multithreading

To control the number of threads set the environment variable `OMP_NUM_THREADS`.
`LibTorch` and `Zero-Copy` will automatically parallelize, `Im2col` requires that the environment variable be set to the number of threads.

To set it, run:

```sh
export OMP_NUM_THREADS=4
# or
OMP_NUM_THREADS=4 ./benchmark_[method_name]
```

## Verifying correctness

Also in the `this-repo/build/bin/` directory, the `correctness` executable allows verifying if outputs match.
The output does not say if the results are correct or not, it rather shows the maximum absolute difference between two elements in the output.
The reference outputs are from LibTorch.

## Scripts

Scripts are gather convolutional layer parameters, run benchmarks and summarize results.
All scripts can take the `-h` flag to show usage information.

Workflow:
 1. Run `timm_convolution_extraction` to generate a csv file containing convolution parameters, or use the csv file provided (`data/conv_layers.csv`)
 2. (optional) Run `filter_csv` to filter the csv generated in step 1
 3. Build this repo
 4. Run `benchmark_runner` with the build dir and the csv from step 1 or 2 to test performance or correctness
 5. Run `summarize_correctness` or `summarize_performance` depending on the type of run with the output csv generated by the runner to get final csv and graphs

