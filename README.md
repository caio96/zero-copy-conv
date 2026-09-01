# Zero-Copy GEMM-Based Fast Convolution :zap:

This repository is designed to test the Zero-Copy GEMM-based convolution on CPUs.
The repository is divided into:
- [Docker](#docker)
- [Single convolution testing](#single-convolution-testing)
- [End-to-end model testing](#end-to-end-model-testing)
- [Scalability testing](#scalability-testing)
- [Memory usage testing](#memory-usage-testing)

---

# Docker

This Docker configuration builds the tools required to run and evaluate ZConv.

## Requirements

The built Docker image requires about 8 GB of disk space.
The benchmarks are compiled with `-march=native` and BLIS is configured with `auto`, so the image is specific to the CPU that builds it: build it on the same machine where the evaluation will run (a prebuilt image moved to a different CPU can fail with SIGILL).

## Files

- `Dockerfile`: the docker build script
- `pytorch-zconv.patch`: patch to PyTorch to add a ZConv convolution backend
- `blis-yaconv.patch`: patch with changes to the BLIS library to add Yaconv
- `deeplabv3plus-zconv.patch`: patch to deeplabv3plus to enable running it with ZConv
- `auto-run.sh`: script to automatically run the main evaluation in the container
- `auto-run-extra.sh`: script to run the additional scalability and memory experiments
- `.dockerignore`: keeps `*.tar` files (such as the Pascal VOC archive) out of the build context

## Build

Before building, add the following file to the `./docker` directory:

- `zero-copy-conv.zip`: this repository, archived into a **flat** zip file (the archive must contain `single-conv/`, `end-to-end/`, ... at its top level, with no enclosing `zero-copy-conv/` directory, because the Dockerfile runs `unzip zero-copy-conv.zip -d zero-copy-conv` and then `cd zero-copy-conv/single-conv`). GitHub's "Download ZIP" and `zip -r zero-copy-conv.zip zero-copy-conv/` both produce a nested top-level directory and will break the build. Create it from the root of this repository with:

```bash
cd /path/to/zero-copy-conv
git archive --format=zip -o docker/zero-copy-conv.zip HEAD
```

You will also need the Pascal VOC 2012 train/val archive (`VOCtrainval_11-May-2012.tar`), used by deeplabv3plus in the end-to-end evaluation.
It is *not* baked into the image; it is bind-mounted into the container and extracted there (see [Run](#run)).
Download it from the official Pascal host or a mirror: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Then build the Docker image with:

```bash
cd ./docker
docker build -t zconv .
```

## Run

The container is created with privileged access so that `numactl` works inside it.
`perf` support is best-effort: the image only installs `linux-tools-common`/`linux-tools-generic`, and Ubuntu's `/usr/bin/perf` dispatches to `/usr/lib/linux-tools/$(uname -r)/perf`, so a matching `linux-tools-$(uname -r)` may have to be installed inside the container for the host's kernel.
None of the automated scripts use `perf`; it is only reached through `benchmark_runner.sh --save-profile`.

```bash
# Create the container
docker create -it --privileged --name artifact \
    -v /absolute/path/VOCtrainval_11-May-2012.tar:/home/artifact/VOCtrainval_11-May-2012.tar \
    zconv:latest
# Start it
docker start artifact
# Attach to it
docker exec -it artifact bash
# Stop it
docker stop artifact
```

The Pascal VOC archive is only mounted, never extracted by the image.
Extract it once inside the container so that deeplabv3plus finds it under its default `--data_root` (`./datasets/data` relative to the `~/deeplabv3plus` clone):

```bash
# Inside the container
mkdir -p ~/deeplabv3plus/datasets/data
tar xf ~/VOCtrainval_11-May-2012.tar -C ~/deeplabv3plus/datasets/data
# This creates ~/deeplabv3plus/datasets/data/VOCdevkit/VOC2012
```

## Perf support

To enable running perf, execute in the host:
```bash
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
```

Profiling runs are started with `benchmark_runner.sh --save-profile`.
The events it collects default to a set of Intel hybrid-core event names; override them by exporting `PERF_EVENTS` with a comma-separated list (no spaces) of events valid for your CPU before running the script.

## Automatic run

To run the evaluation script in the docker container automatically, run:
```bash
# 1. Attach to the container
docker exec -it artifact bash
# 2. (optional but recommended) Update the variables at the top of `auto-run.sh` to suit your environment:
# - `CORE_RANGE` limits the cores used by the runs
# - `THREADS` sets the number of threads for multithreaded runs
# - `CONV_LAYERS_MAX` sets a maximum number of convolution layers to run (use `-1` to run all)
# - `REPEAT_COUNT_CONV` controls repeats for standalone convolution measurements
# - `MODEL_MAX` sets a maximum number of end-to-end models to run (use `-1` to run all)
# - `REPEAT_COUNT_MODEL` controls repeats for end-to-end model measurements
vim auto-run.sh # edit with your preferred terminal editor
# 3. Run it
./auto-run.sh
```

# Single Convolution Testing

Compares the following convolution implementations.
Over 9000 convolution layers were extracted from real models to evaluate the different methods.
Data type used is float32.

- Im2col: Transforms the input image with im2col and executes convolution as a GEMM call
- Yaconv: Implementation from this [paper](https://dl.acm.org/doi/10.1145/3570305) with slight improvements
- ZeroCopy: Standalone C++ implementation that executes convolution as a sequence of GEMMs without transforming the input
- LibTorch_ZeroCopy: ZeroCopy Conv2D implemented inside Pytorch, run using its C++ API
- LibTorch: Pytorch Conv2D implementation using the C++ API (the ZeroCopy implementation in Pytorch is disabled)

|                   | Feature Layout | Weight Layout | Output Layout | Multithreading     |
| ----------------- | -------------- | ------------- | ------------- | ------------------ |
| Im2col            | NCHW           | OIHW          | NCHW          | :white_check_mark: |
| Yaconv            | NHWC           | HWIO          | NHWC          | :x:                |
| ZeroCopy          | NHWC           | HWIO          | NHWC          | :white_check_mark: |
| LibTorch_ZeroCopy | NHWC           | HWIO          | NHWC          | :white_check_mark: |
| LibTorch          | NHWC           | OHWI          | NHWC          | :white_check_mark: |

Note:

- Yaconv only supports stride == 1, no grouping, and no dilation.
- The LibTorch methods have a slightly different semantic from other methods: they also allocate their output inside the convolution call. Therefore, output allocation is included in the timing, while other methods only run convolution with a preallocated output.

## Files

- `data`: Contains convolution layer parameters obtained with the script `single-conv/scripts/convolution_extraction.py`. Timm 1.0.13 and TorchVision 0.20.1 were used.
- `include` contains utility function headers
- `src`
  - `driver` is the main file that uses Google Benchmark to call a convolution benchmark. The defines passed at compile time control which convolution method is called
  - `driver_[method_name]` is a specialized driver if the method differs too much from the main driver
  - `kernel_conv_[method_name]` files have the implementation of each convolution method
  - `utils` contains utility functions
  - `verify_correctness` calls all convolutions methods converting their output if necessary to verify results. LibTorch's output is used as the reference.
- `scripts`:
  - `convolution_extraction` extracts convolution layer parameters from multiple models into a csv (adapted from [ConvBench](https://github.com/LucasFernando-aes/ConvBench/))
  - `filter_csv` allows filtering the csv containing convolution layer parameters and removes parameters that cause errors (filter size > input image + padding)
  - `benchmark_runner` controls running convolutions from a csv to measure correctness or performance
  - `summarize_correctness` generates csv files that summarize correctness results based on the logs from `benchmark_runner`
  - `summarize_performance` generates a csv file that summarizes performance results based on the logs from `benchmark_runner`
  - `summarize_profiler` generates a csv file that summarizes perf event counter logs from `benchmark_runner`

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
      -DBUILD_ZCONV_BLIS="[ON/OFF]"                           \
      -DUSE_FIXED_ITERATIONS="[ON/OFF]"                       \
      ..
```

- If USE_MKL is set OFF, Blis is used as a BLAS library, otherwise OneMKL is used.
- If USE_MKL_JIT is set ON, the Zero Copy Convolution may use MKL's jit for some GEMM configurations.
- If both BUILD_ZCONV_BLIS and USE_MKL are set ON, an additional binary is compiled for ZConv that uses BLIS as BLAS.
- If USE_FIXED_ITERATIONS is set ON, benchmarks will run for a fixed number of iterations instead of running for minimum amount of time (useful for profiling runs).

## Running Benchmarks

After building, the build directory (`this-repo/single-conv/build/`) will contain one executable per convolution method with the name `benchmark_[method_name]`.
If you additionally run `cmake --install . --prefix /some/prefix`, the same executables are installed into `/some/prefix/bin/` (this is what the Docker image does, producing `~/install/single-conv/bin/`).
The executables can be run with `--help` to show the parameters they take. If run with no parameters, a default configuration is run.

### Multithreading

To control the number of threads, set the environment variable `OMP_NUM_THREADS`.
`LibTorch` and `Zero-Copy` will automatically parallelize; `Im2col` may require the environment variable be set to the number of threads.
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

Also next to the `benchmark_*` executables (in the build directory, or in `bin/` under the install prefix), the `correctness` executable allows verifying if outputs match.
The output does not say if the results are correct or not, it rather shows the maximum absolute difference between two elements in the output.
The reference outputs are from LibTorch (with the ZeroCopy convolution implementation disabled).
This executable can also be run with `--help` and it takes the same parameters as the `benchmark_` executables.

## Workflow

Scripts gather convolutional layer parameters, run benchmarks, and summarize results.
All scripts can take the `-h` flag to show usage information.

1. Run `convolution_extraction` to generate a csv file containing convolution parameters, or use the csv files provided (`data/conv_layers_all.csv`; `data/conv_layers_timm.csv`, `data/conv_layers_torch.csv` and `data/conv_layers_yaconv_supported.csv` are subsets of it)
2. Run `filter_csv` to remove parameters that cause errors in the csv generated in step 1 and to optionally filter convolution types
3. Build this repo
4. Run `benchmark_runner` with the build dir and the csv from step 2 to test performance or correctness
    - This script will use the `benchmark_*` executables found in the build dir, remove executables if you do not want to execute them
5. Run `summarize_correctness` or `summarize_performance` depending on the type of run with the output csv generated by the runner to summary CSVs
6. To see the effect of the heuristic that decides when ZConv is used, re-run `summarize_performance` on the same runner output with `--include-only-conv-types torch-heuristic`. This restricts the summary to the layers the heuristic selects, and is what `docker/auto-run.sh` uses for the LibTorch vs LibTorch-ZConv comparison. The heuristic itself is the `torch-heuristic` category implemented in `include_only_in_df` in `single-conv/scripts/filter_csv.py`; edit it there to change the heuristic (`summarize_performance` imports that function). The same category names are also accepted by `filter_csv` in step 2 (`--include-only-conv-types` / `--exclude-conv-types`).

---

# End-to-end Model Testing

Evaluates the performance of Zero-Copy convolution integrated to PyTorch, enabling end-to-end runs that use this convolution implementation.
The Zero-Copy convolution is split into two implementations `ZeroCopy2d` and `ZeroCopy2d_Ext`. The `Ext` version supports dilated and grouped convolution.
Zero-Copy Convolution is integrated to the convolution selector in PyTorch and it has its own heuristic, so it may not always be selected.

## Files

- `run_torch_model` executes a model multiple times and reports execution time metrics.
- `filter_models` filters models based on their convolution layers
- `benchmark_models` executes all models with multiple methods (*e.g.* ZeroCopy2d enabled and disabled) and saves results in a csv. **Note that** the number of threads and cores used must be manually set before running this script with `OMP_NUM_THREADS` and `numactl`.
- `summarize_performance_end_to_end` generates a csv file and summarizes performance results based on the logs from `benchmark_models`

## ZeroCopy2d Behavior in PyTorch

The execution of ZeroCopy2d in PyTorch is controlled by the following environment variables.
- <code>ZC_ENABLE=[TRUE/**FALSE**]</code>: Enables ZeroCopy2d
- <code>ZC_TIME=[TRUE/**FALSE**]</code>: Prints the parameters and execution time of the convolution layers run in the model
- <code>ZC_HEURISTIC=[**TRUE**/FALSE]</code>: If set to FALSE, the heuristic that decides whether to use ZeroCopy2d (and ZeroCopy2d_Ext) is ignored. Thus, ZeroCopy2d is always used.
- <code>ZC_WEIGHTS_LAYOUT=[**HWIO**/OHWI]</code>: If set to HWIO, Pytorch only runs ZeroCopy2d if the weight layout is HWIO and scripts will change the weight layout to HWIO before running ZeroCopy2d.

For end-to-end execution, the following scripts control these variables automatically.
Thus setting them before the end-to-end scripts has no effect.

## Running Models

Use the script `run_torch_model.py` to run a PyTorch model.
- The model source (`torch` or `timm`) is a required positional argument
- Enable ZeroCopy2d with `--zc-enable`
- For more options, run `run_torch_model.py -h`
- Multithreading works the same as explained in this [section](#multithreading).

### Examples

```sh
# Check how to use flags
./run_torch_model.py -h
# Run mobilenet_v3_large with default PyTorch, 8 threads, using cores 0 to 7
# (the first positional argument is the model source: `torch` or `timm`, and it is required)
OMP_NUM_THREADS=8 numactl -C 0-7 ./run_torch_model.py torch --model-name mobilenet_v3_large
# Run mobilenet_v3_large with ZeroCopy2d enabled, 8 threads, using cores 0 to 7
OMP_NUM_THREADS=8 numactl -C 0-7 ./run_torch_model.py torch --model-name mobilenet_v3_large --zc-enable
```

## Workflow

All scripts can take the `-h` flag to show usage information.

Workflow:

1. Run `filter_models` to get a list of relevant models by filtering with "torch-heuristic" for example
2. Run `benchmark_models` to test performance of running models with and without ZeroCopy2d
3. Run `summarize_performance_end_to_end` with the output csv generated by the `benchmark_models` to get a summary csv

## Running DeepLabV3+ on Pascal VOC

The segmentation experiment lives in the patched DeepLabV3+ clone at `~/deeplabv3plus` and is **not** part of `auto-run.sh`; it must be invoked manually.
It requires the Pascal VOC data to have been extracted first (see [Run](#run)).
The patch adds `~/deeplabv3plus/run_models.sh`, which takes no arguments and loops over six `deeplabv3plus_*` models, running each one twice (once with default PyTorch and once with `--zc-enable`).
It hardcodes `OMP_NUM_THREADS=8` and `numactl -C 0-7`; edit the script if your machine needs a different core range.
Results are printed to stdout only (a `Validation time:` line and the metrics summary per run), so redirect them to keep them:

```bash
# Inside the container
mkdir -p ~/results
cd ~/deeplabv3plus
./run_models.sh 2>&1 | tee ~/results/deeplabv3plus.log
```

---

# Scalability Testing

Measures how Im2col, LibTorch, LibTorch-ZConv and ZConv scale from 1 to 8 CPU cores for the six representative layers from Table 6 of the paper.
(Yaconv is single-threaded by design and the ZConv-BLIS variant is not built for this experiment, so neither appears in this comparison.)
Speedup is reported relative to the single-core execution of each method.

## Files

- `scalability/layers.csv`: the six benchmark layers in `benchmark_runner.sh` format
- `scalability/run_scalability.sh BUILD_DIR OUTPUT_DIR [REPEATS]`: runs the benchmark at 1, 2, 4, and 8 threads and writes one raw CSV per thread count to `OUTPUT_DIR`
- `scalability/summarize_scalability.py OUTPUT_DIR [--out data.csv]`: aggregates the raw CSVs into a `data.csv` with speedup values
- `scalability/plot_scalability.py [--data PATH] [--output PATH]`: generates the paper figure from `data.csv`

## Workflow

```bash
# Inside the container (or any machine with the binaries built)
./scalability/run_scalability.sh ~/install/single-conv ~/results/scalability/raw 10
python3 ./scalability/summarize_scalability.py ~/results/scalability/raw --out ~/results/scalability/data.csv
# To regenerate the paper figure:
python3 ./scalability/plot_scalability.py --data ~/results/scalability/data.csv --output scalability.png
```

---

# Memory Usage Testing

Measures peak resident set size (RSS) for each convolution method over the six representative layers.
Executables must be built with `USE_FIXED_ITERATIONS=ON` so that each run allocates its working buffers a fixed number of times, giving stable RSS measurements.
Peak RSS is captured by wrapping each run with GNU `/usr/bin/time -v` (from the apt `time` package, which is installed in the image) and reading its "Maximum resident set size" line.

## Files

- `memory/layers.csv`: the six benchmark layers
- `memory/run_memory.sh BUILD_DIR OUTPUT_DIR [REPEATS]`: runs each executable with RSS measurement and writes `memory_raw.csv`
- `memory/summarize_memory.py memory_raw.csv [--out results.csv]`: computes mean RSS per method and layer and prints the three comparison tables (Im2col vs ZConv, Yaconv vs ZConv-BLIS, LibTorch vs LibTorch-ZConv)

## Workflow

```bash
# Run main build (im2col, zconv/mkl, libtorch, libtorch-zconv) — use fixed-iterations install
./memory/run_memory.sh ~/install/single-conv-fixed ~/results/memory/main 10
# Run yaconv build (yaconv, zconv/blis)
./memory/run_memory.sh ~/install/single-conv-yaconv-fixed ~/results/memory/yaconv 10
# Merge: the yaconv build's benchmark_zero_copy is the BLIS variant
cat ~/results/memory/main/memory_raw.csv > ~/results/memory/memory_raw.csv
sed 's/^benchmark_zero_copy,/benchmark_zero_copy_blis,/' ~/results/memory/yaconv/memory_raw.csv | tail -n +2 >> ~/results/memory/memory_raw.csv
# Summarize
python3 ./memory/summarize_memory.py ~/results/memory/memory_raw.csv --out ~/results/memory/memory_results.csv
```

Both experiments are automated by `auto-run-extra.sh`, which runs them sequentially using the appropriate install directories.
To run it in the container:

```bash
# 1. Attach to the container
docker exec -it artifact bash
# 2. (optional but recommended) Update the variables at the top of `auto-run-extra.sh` to suit your environment:
# - `REPEAT_COUNT_SCALABILITY` controls repeats per (layer, thread-count) pair
# - `REPEAT_COUNT_MEMORY` controls repeats per (executable, layer) pair
vim auto-run-extra.sh # edit with your preferred terminal editor
# 3. Run it
./auto-run-extra.sh
```

The scalability thread counts (1, 2, 4 and 8) and the core pinning used by the scalability and memory runs are fixed and cannot be configured in `auto-run-extra.sh`.
They are set inside `scalability/run_scalability.sh` (thread counts and the matching `0-$((THREADS - 1))` core range) and `memory/run_memory.sh` (single core, `numactl --physcpubind 0`); edit those scripts if your machine needs different values.
