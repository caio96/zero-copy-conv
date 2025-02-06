# Zero-Copy GEMM-Based Fast Convolution :zap:

This repository is designed to test the Zero-Copy GEMM-based convolution (in CPUs).
The repository is divided in two:
- [Single convolution testing](#single-convolution-testing)
- [End-to-end model testing](#end-to-end-model-testing)

---

# Common Dependecies

## Conda Environment

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

Use this environment when building or running the following experiments.

## Custom PyTorch + TorchVision

Build and install a custom PyTorch that has a Zero-Copy Convolution implementation, and TorchVision.

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

# Set number of threads and compile, this command will install torch and build LibTorch
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

---

# Single Convolution Testing

Compares the following convolution implementations.
Over 9000 convolution layers were extracted from real models to evaluate the different methods.
Data type used is float32.

- Naive: For loops and simple multiply accumulate
- Im2col: Transforms the input image with im2col and executes convolution as a GEMM call
- Yaconv: Implementation from this [paper](https://dl.acm.org/doi/10.1145/3570305) with slight improvements. Defined in [blis-conv](https://github.com/caio96/blis-conv)
- ZeroCopy: Standalone C++ implementation that executes convolution as a sequence of GEMMs without transforming the input
- OneDNN_any: Intel's OneDNN implementation allowing the framework to decide the best layouts. Transforming the layout is not included in the timing.
- LibTorch_ZeroCopy: ZeroCopy Conv2D implemented inside Pytorch, run using its C++ API
- LibTorch: Pytorch Conv2D implementation using the C++ API (the ZeroCopy implementation in Pytorch is disabled)

|                   | Feature Layout | Weight Layout | Output Layout | Multithreading     |
| ----------------- | -------------- | ------------- | ------------- | ------------------ |
| Naive             | NCHW           | OIHW          | NCHW          | :x:                |
| Im2col            | NCHW           | OIHW          | NCHW          | :white_check_mark: |
| Yaconv            | NHWC           | HWIO          | NHWC          | :x:                |
| ZeroCopy          | NHWC           | HWIO          | NWHC          | :white_check_mark: |
| OneDNN_any        | ??             | ??            | ??            | :white_check_mark: |
| LibTorch_ZeroCopy | NHWC           | HWIO          | NHWC          | :white_check_mark: |
| LibTorch          | NHWC           | OHWI          | NHWC          | :white_check_mark: |

Note:

- Yaconv only supports stride == 1, no grouping, and no dilation.
- The LibTorch methods have a slightly different semantic from other methods: they also allocate their output inside the convolution call. Therefore, output allocation is included in the timing, while other methods only run convolution with a preallocated output.

## Files

- `data`: Contains convolution layer parameters obtained with the script `util/convolution_extraction`. Timm 1.0.13 and TorchVision 0.20.1 were used.
- `include`
- `src`
  - `driver` is the main file that uses Google Benchmark to call a convolution benchmark. The defines passed at compile time control which convolution method is called
  - `driver_[method_name]` is a specialized driver if the method differs too much from the main driver
  - `kernel_conv_[method_name]` files have the implementation of each convolution method
  - `utils`
  - `verify_correctness` calls all convolutions methods converting their output if necessary to verify results. LibTorch's output is used as a reference.
- `scripts`:
  - `convolution_extraction` extracts convolution layer parameters from multiple models into a csv (adapted from [ConvBench](https://github.com/LucasFernando-aes/ConvBench/))
  - `filter_csv` allows filtering the csv containing convolution layer parameters and removes parameters that cause errors
  - `benchmark_runner` controls running convolutions from a csv to measure correctness or performance
  - `summarize_correctness` generates csv files that summarize correctness results based on the logs from `benchmark_runner`
  - `summarize_performance` generates a csv file that summarizes performance results based on the logs from `benchmark_runner`
  - `learn_heuristic` uses the speedup csv generated by `summarize_performance` to train a decision tree to decide which method to use based the convolution parameters
  - `zero_copy_conv` is a simplified version of the Zero-Copy Convolution implemented in Python

## Dependencies

- [Google Benchmark](https://github.com/google/benchmark)
- [Blis](https://github.com/flame/blis)
- [Yaconv Blis](https://github.com/caio96/blis-conv), which is a fork of Blis that contains Yaconv
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

### How to install OneDNN and OneMKL

```sh
conda install conda-forge::onednn==3.5.3 conda-forge::mkl-devel==2025.0.0
```

### (Alternative) How to get LibTorch

This method will work for the default LibTorch implementation, but it won't have LibTorch_ZeroCopy.
Some adjustments in the repo may be needed for this to work as the custom PyTorch implementation is expected.

- Go to [link](https://pytorch.org/get-started/locally/)
- Select Package as "LibTorch", Language as "C++/Java", Compute platform as "CPU"
- Download the cxx11 ABI version and unzip it

Or run:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip
unzip libtorch.zip -d .
```

## How to Build Single Convolution Benchmarks

```sh
cd single-conv
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang                                \
      -DCMAKE_CXX_COMPILER=clang++                            \
      -DBENCHMARK_INSTALL="/path/to/google-benchmark-install" \
      -DTORCH_INSTALL="/path/to/custom-pytorch/torch"         \
      -DBLIS_INSTALL="/path/to/blis-install"                  \
      -DBLIS_YACONV_INSTALL="/path/to/blis-conv-install"      \
      -DUSE_MKL="[ON/OFF]"                                    \
      -DUSE_MKL_JIT="[ON/OFF]"                                \
      ..
```

- If USE_MKL is set OFF, Blis is used as a BLAS library.
- If USE_MKL_JIT is set ON, the Zero Copy Convolution may use MKL's jit for some GEMM configurations.

## Running Benchmarks

After building, the `this-repo/build/bin/` directory will contain one executable per convolution method with the name `benchmark_[method_name]`.
The executables can be run with `--help` to show the parameters they take. If run with no parameters, a default configuration is run.

### Multithreading

To control the number of threads, set the environment variable `OMP_NUM_THREADS`.
`LibTorch`, `OneDNN`, and `Zero-Copy` will automatically parallelize, `Im2col` may require the environment variable be set to the number of threads.
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
OMP_NUM_THREADS=8 numactl -C 0-7 ./benchmark_zero_copy 1 128 14 14 32 3 3 1 1 1 1 1 1 1 1 1 0 0
```

More custom layer configurations are found in `data/`

## Verifying correctness

Also in the `this-repo/build/bin/` directory, the `correctness` executable allows verifying if outputs match.
The output does not say if the results are correct or not, it rather shows the maximum absolute difference between two elements in the output.
The reference outputs are from LibTorch (with the ZeroCopy convolution implementation disabled).
This executable can also be run with `--help` and it takes the same parameters as the `benchmark_` executables.

## Workflow

Scripts gather convolutional layer parameters, run benchmarks, and summarize results.
All scripts can take the `-h` flag to show usage information.

1. Run `convolution_extraction` to generate a csv file containing convolution parameters, or use the csv files provided (`data/conv_layers.csv`)
2. Run `filter_csv` to remove parameters that cause errors in the csv generated in step 1 and to optionally filter convolution types
3. Build this repo
4. Run `benchmark_runner` with the build dir and the csv from step 2 to test performance or correctness
    - This script will use the `benchmark_*` executables found in the build dir (except the naive one), remove executables if you do not want to execute them
5. Run `summarize_correctness` or `summarize_performance` depending on the type of run with the output csv generated by the runner to summary CSVs
6. Run `learn_heuristic` on one of the speedup outputs of `summarize_performance` to see decision tree heuristics
7. Modify the heuristic in `summarize_performance` run it on the output of the runner with `--use-heuristic` enabled to see the effects of the heuristic

---

# End-to-end Model Testing

Evaluates the performance of Zero-Copy convolution integrated to PyTorch, enabling end-to-end runs that use this convolution implementation.
The Zero-Copy convolution is split into two implementations `ZeroCopy2d` and `ZeroCopy2d_Ext`. The `Ext` version supports dilated and grouped convolution.
Zero-Copy Convolution is integrated to the convolution selector in PyTorch and it has its own heuristic, so it may not always be selected.

## Files

- `run_torch_model` executes a model multiple times using the `torch.utils.benchmark` API and reports execution time metrics.
- `benchmark_models` executes all models with multiple methods (*e.g.* ZeroCopy2d enabled and disabled) and saves results in a csv
- `summarize_performance_end_to_end` generates a csv file and summarizes performance results based on the logs from `benchmark_models`

## ZeroCopy2d Behavior in PyTorch

The execution of ZeroCopy2d in PyTorch is controlled by the following environment variables.
- <code>ZC_ENABLE=[TRUE/**FALSE**]</code>: Enables ZeroCopy2d
- <code>ZC_TIME=[TRUE/**FALSE**]</code>: Prints the parameters and execution time of the convolution layers run in the model
- <code>ZC_TRANSFORM_OUTPUT=[**TRUE**/FALSE]</code>: If set to FALSE, disables the transposition of `WH` to `HW` of the output.
    - Disabling the transformation is not supported and may generated incorrect results
- <code>ZC_HEURISTIC=[**TRUE**/FALSE]</code>: If set to FALSE, the heuristic that decides whether to use ZeroCopy2d (and ZeroCopy2d_Ext) is ignored. Thus, ZeroCopy2d is always used (and ZeroCopy2d_Ext is never used).
- <code>ZC_WEIGHTS_LAYOUT=[**HWIO**/OHWI]</code>: This variable does not affect PyTorch, but it maybe affect some scripts by changing layouts of the weights given to ZeroCopy2d (it only affects convolution layers that will execute using ZeroCopy2d)

For end-to-end execution, the scripts control these variables automatically.
Thus setting them before the end-to-end scripts has no effect.
They do affect the execution of `LibTorch_ZeroCopy` in the single-conv experiments with the exception of `ZC_ENABLE`.

## Running Models

Use the script `run_torch_model.py` to run a PyTorch model.
- Enable ZeroCopy2d with `--zc-enable`
- For more options, run `run_torch_model.py -h`
- Multithreading works the same as explained in this [section](#multithreading).

### Examples

```sh
# Check how to use flags
./run_torch_model.py -h
# Run mobilenet_v3_large with default PyTorch, 8 threads, using cores 0 to 7
OMP_NUM_THREADS=8 numactl -C 0-7 ./run_torch_model.py --model-name mobilenet_v3_large
# Run mobilenet_v3_large with ZeroCopy2d enabled, 8 threads, using cores 0 to 7
OMP_NUM_THREADS=8 numactl -C 0-7 ./run_torch_model.py --model-name mobilenet_v3_large --zc-enable
```

## Workflow

All scripts can take the `-h` flag to show usage information.

Workflow:

1. Run `benchmark_models` to test performance of running model with different methods
2. Run `summarize_performance_end_to_end` with the output csv generated by the `benchmark_models` to get a summary csv
