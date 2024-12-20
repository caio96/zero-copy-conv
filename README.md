# Zero-Copy GEMM-Based Fast Convolution :zap:

This repository is designed to test the zero-copy gemm-based convolution.
The repository is divided in two:
- [Single convolution testing](#single-convolution-testing)
- [End-to-end model testing](#end-to-end-model-testing)
---

# Conda Environment

Install conda first if necessary:

```sh
mkdir -p ~/.anaconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/.anaconda3/miniconda.sh
bash ~/.anaconda3/miniconda.sh -b -u -p ~/.anaconda3
rm ~/.anaconda3/miniconda.sh
source ~/.anaconda3/bin/activate
conda init --all
```

Then create a new environment and install some common dependencies:

```sh
conda create -y -n eval-zero-copy python==3.12
conda activate eval-zero-copy
conda install cmake ninja
```

Activate this environment when building or running the following experiments.


---

# Single Convolution Testing

Compares the Zero-Copy convolution with the following convolution implementations in over 10000 convolution layers extracted from real models.
Data type used is float32.

- Naive convolution: For loops and simple multiply accumulate
- Im2col convolution: Transforms the input image with im2col and executes convolution as a single GEMM call
- Yaconv convolution: Implementation from this [paper](https://dl.acm.org/doi/10.1145/3570305) with slight improvements. Defined in [blis-conv](https://github.com/caio96/blis-conv)
- Zero-Copy convolution: New convolution implementation, yet to be published
- LibTorch convolution: Pytorch Conv2D implementation using the C++ API
- OneDNN_nhwc convolution: Intel's OneDNN implementation setting layouts to NHWC
- OneDNN_any concolution: Intel's OneDNN implementation allowing the framework to decide the best layouts. Transforming the layout is not included in the timing.

|             | Feature Layout | Weight Layout | Output Layout | Multithreading     |
| ----------- | -------------- | ------------- | ------------- | ------------------ |
| Naive       | NCHW           | OIHW          | NCHW          | :x:                |
| Im2col      | NCHW           | OIHW          | NCHW          | :white_check_mark: |
| Yaconv      | NHWC           | HWIO          | NHWC          | :x:                |
| Zero-Copy   | NHWC           | HWIO          | NWHC          | :white_check_mark: |
| LibTorch    | NHWC           | OHWI          | NHWC          | :white_check_mark: |
| OneDNN_any  | ??             | ??            | ??            | :white_check_mark: |
| OneDNN_nhwc | NHWC           | HWIO          | NHWC          | :white_check_mark: |

Note:

- Yaconv only supports stride == 1, no grouping, and no dilation.
- Libtorch has a slightly different semantic from other methods: it also allocates its output in its convolution call, so that is included in the timing, while other methods only run convolution with a preallocated output.

## Repository Structure

- `data`: Contains convolution layer parameters obtained with the script `util/timm_convolution_extraction`. Timm version 1.0.11 was used.
- `include`
- `src`
  - `driver` is the main file that uses Google Benchmark to call a convolution benchmark. The defines passed at compile time control which convolution method is called
  - `kernel_conv_[method_name]` files have the implementation of each convolution method
  - `driver_[method_name]` is a specialized driver if the method differs too much from the main driver
  - `utils` has helper functions
  - `verify_correctness` calls all convolutions methods converting their output if necessary to verify results. LibTorch's output is used as a reference.
- `scripts`:
  - `timm_convolution_extraction` is adapted from [ConvBench](https://github.com/LucasFernando-aes/ConvBench/) to extracts convolution layer parameters from multiple models into a csv
  - `filter_csv` allows filtering the csv containing convolution layer parameters
  - `benchmark_runner` controls running all convolutions in a generated csv to measure correctness or performance
  - `summarize_correctness` generates csv files that summarize correctness results based on the logs from `benchmark_runner`
  - `summarize_performance` generates csv files and graphs that summarize performance results based on the logs from `benchmark_runner`
  - `zero_copy_conv` is a simplified version of the Zero-Copy Convolution implemented in Python

## Dependencies

- [Google Benchmark](https://github.com/google/benchmark)
- [Blis](https://github.com/flame/blis)
- [Yaconv Blis](https://github.com/caio96/blis-conv), which is a fork of Blis that contains Yaconv
- [LibTorch](https://pytorch.org/cppdocs/installing.html)
- [OneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html)
- [OneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

### How to build Google Benchmark

```sh
git clone https://github.com/google/benchmark.git
cd benchmark
git checkout v1.9.0
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
cmake --install build --config Release --prefix "/path/to/benchmark-install"
```

### How to build Blis

```sh
git clone https://github.com/flame/blis.git
cd blis
git checkout 1.0
./configure --prefix="/path/to/blis-install" \
            --enable-threading=openmp        \
            --enable-cblas                   \
            CC=clang CXX=clang++             \
            auto
make install -j4
```

### How to build Blis Yaconv

```sh
git clone git@github.com:caio96/blis-conv.git
cd blis-conv
git checkout yaconv-update
./configure --prefix="/path/to/blis-conv-install" \
            --enable-threading=openmp             \
            --enable-cblas                        \
            CC=clang CXX=clang++                  \
            -a yaconv                             \
            auto
make install -j4
```

### How to get LibTorch

- Go to [link](https://pytorch.org/get-started/locally/)
- Select Package as "LibTorch", Language as "C++/Java", Compute platform as "CPU"
- Download the cxx11 ABI version and unzip it

Or run:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip
unzip libtorch.zip -d .
```

### How to install OneDNN and OneMKL

```sh
conda install conda-forge::onednn==3.5.3 conda-forge::mkl-devel==2025.0.0
```

## Building Single Conv Benchmarks

```sh
cd single-conv
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang                                \
      -DCMAKE_CXX_COMPILER=clang++                            \
      -DBENCHMARK_INSTALL="/path/to/google-benchmark-install" \
      -DTORCH_INSTALL="/path/to/libtorch"                     \
      -DBLIS_INSTALL="/path/to/blis-install"                  \
      -DBLIS_YACONV_INSTALL="/path/to/blis-conv-install"      \
      -DUSE_MKL="[ON/OFF]"                                    \
      -DUSE_MKL_JIT="[ON/OFF]"                                \
      ..
```

- Instead of using the path to the downloaded LibTorch, TORCH_INSTALL can also use the path to `pytorch/torch`, where `pytorch` is the custom PyTorch built for the end-to-end testing in this [section](#how-to-build-custom-pytorch)
- If USE_MKL is set OFF, Blis is used as a BLAS library.
- If USE_MKL_JIT is set ON, the Zero Copy Convolution will use MKL's jit for its base GEMM configuration, other GEMMs configurations (mostly due to padding) won't use jit.

## Running Benchmarks

After building, the `this-repo/build/bin/` directory will contain one executable per convolution method with the name `benchmark_[method_name]`.
The executables can be run with `--help` to show the parameters they take. If run with no parameters, a default configuration is run.

### Multithreading

To control the number of threads set the environment variable `OMP_NUM_THREADS`.
`LibTorch`, `OneDNN`, and `Zero-Copy` will automatically parallelize, `Im2col` may require the following environment variable be set to the number of threads.
It is also recommended to disable Hyper-Threading or to use numactl to set which cores are used.

To set it, run:

```sh
export OMP_NUM_THREADS=4
# or
OMP_NUM_THREADS=4 ./benchmark_[method_name]
```

### Examples

```sh
# Zero-Copy Conv, default parameters, single thread
OMP_NUM_THREADS=1 ./benchmark_zero_copy
# Zero-Copy Conv, default parameters, 8 threads
OMP_NUM_THREADS=8 ./benchmark_zero_copy
# Zero-Copy Conv, default parameters, 8 threads, using cores 0 to 7
OMP_NUM_THREADS=8 numactl -C 0-7 ./benchmark_zero_copy
# Check how to use custom parameters
./benchmark_zero_copy -h
# Zero-Copy Conv, custom parameters, 8 threads, using cores 0 to 7
OMP_NUM_THREADS=8 numactl -C 0-7 ./benchmark_zero_copy 16 64 64 64 128 3 3 1 1 1 1 1 1 1 1 1 0 0
```

More custom layer configurations are found in `data/conv_layers.csv`

## Verifying correctness

Also in the `this-repo/build/bin/` directory, the `correctness` executable allows verifying if outputs match.
The output does not say if the results are correct or not, it rather shows the maximum absolute difference between two elements in the output.
The reference outputs are from LibTorch.
This executable can also be run with `--help` and it takes the same parameters as the `benchmark_` executables.

## Scripts

Scripts gather convolutional layer parameters, run benchmarks and summarize results.
All scripts can take the `-h` flag to show usage information.

Workflow:

1.  Run `timm_convolution_extraction` to generate a csv file containing convolution parameters, or use the csv file provided (`data/conv_layers.csv`)
2.  (optional) Run `filter_csv` to filter the csv generated in step 1
3.  Build this repo
4.  Run `benchmark_runner` with the build dir and the csv from step 1 or 2 to test performance or correctness
5.  Run `summarize_correctness` or `summarize_performance` depending on the type of run with the output csv generated by the runner to get final csv and graphs

---

# End-to-end Model Testing

Adds Zero-Copy Convolution to PyTorch, enabling end-to-end runs that use this convolution implementation.
Zero-Copy Convolution is integrated to the convolution selector in PyTorch, so it may not always be selected.

## Dependencies

- [Custom PyTorch](https://github.com/caio96/pytorch-zero-copy.git)
- [Pytorch Vision](https://github.com/pytorch/vision/tree/main)

### How to build custom PyTorch

Install MKL:

```sh
conda install conda-forge::mkl-static==2025.0.0 conda-forge::mkl-include==2025.0.0
```

Build and install

```sh
git clone --recursive git@github.com:caio96/pytorch-zero-copy.git pytorch
cd pytorch
git checkout v2.5.1-zero-copy
git submodule sync
git submodule update --init --recursive

pip install -r requirements.txt

# Setup build environment
export USE_CUDA=0 USE_ROCM=0 USE_XPU=0
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"

# Set number of threads and compile, this command will install torch
MAX_JOBS=8 python setup.py develop && python tools/build_libtorch.py
```

### How to build PyTorch vision

```sh
conda install libpng libjpeg-turbo -c pytorch

git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.20.1
MAX_JOBS=8 python setup.py develop
```

## Running Models

Use the script `run_torch_model.py` to run a PyTorch model.
- Zero-Copy Convolution is not enabled by default, use the flag `--enable-zero-copy-conv` to enable it.
- PyTorch models in channel last use a weight layout of OHWI, whereas Zero-Copy Conv expects HWIO. By default Zero-Copy Conv transfoms the weights to produce correct results. Disable this transformation with the flag `--ignore-weight-transform`.
- Zero-Copy Conv transposes the height and width of the output. By default Zero-Copy Conv transfoms the output to produce correct results. Disable this transformation with the flag `--ignore-output-transform`.
- For more options, run `run_torch_model.py -h`
- Multithreading works the same as explained in this [section](#multithreading).

