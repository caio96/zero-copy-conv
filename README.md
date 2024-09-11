# Pointer Chase Impact

Evaluates the performance of the RebaseDL optimization (region-based pointer inlining) applied to variations of the 179.art train-match region when changing the number of reuse iterations of the region.

All scripts accept a `-h` flag to show a help message that details all the options that it accepts.

## How to use
1. Generate files:
  - Compiles the google benchmark main function (`./179-art-driver.cpp`) with the RebaseDL optimized (`./179-art-opt.c`) and Clang (`./179-art.c`) kernels. There are multiple kernel folders containing these files.

Run:

```sh
mkdir compile-output
./compile-kernel.sh ./kernel-path/ ./compile-output/
```

2. Execute kernels:
  - Benchmarks are executed with Google benchmark.
  - Reuse iterations vary from 1 to 11
  - It prints the summary of results

Execute it with the following, where OUTPUT_PATH is the output folder of the previous script.
```sh
./compile-output/179-art.exe | tee clang.txt
./compile-output/179-art-opt.exe | tee rebasedl.txt
```

3. Generate graphs:
  - Produces the graph with speedup over Clang over the number of reuse iterations.

Run:

```sh
mkdir graph-output
./output-graph.py clang.txt rebasedl.txt ./graph-output/
```

## Automatic mode

All kernels can be compiled, executed and graphs generated using by running:

```sh
mkdir output
./compile-run-graph-all.sh output
```
