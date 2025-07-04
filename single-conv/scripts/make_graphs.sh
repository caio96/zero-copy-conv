#!/bin/env bash

# Get first argument as the output directory
INPUT_DIR=$1

# Check if the output directory exists
if [[ ! -d $INPUT_DIR || -z $INPUT_DIR ]]; then
    echo "Output directory does not exist"
    exit 1
fi

mkdir -p "$INPUT_DIR"/correctness/1-thread "$INPUT_DIR"/correctness/8-threads
./summarize_correctness.py "$INPUT_DIR"/cdol01-correctness-1-thread.csv "$INPUT_DIR"/correctness/1-thread
./summarize_correctness.py "$INPUT_DIR"/cdol01-correctness-8-threads.csv "$INPUT_DIR"/correctness/8-threads

mkdir -p "$INPUT_DIR"/general-graphs/1-thread "$INPUT_DIR"/general-graphs/8-threads
mkdir -p "$INPUT_DIR"/paper-graphs/1-thread "$INPUT_DIR"/paper-graphs/1-thread-zconv-blis "$INPUT_DIR"/paper-graphs/8-threads

./summarize_performance.py "$INPUT_DIR"/cdol01-performance-1-thread.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/1-thread --clip-pos --clip-neg --plot-type log2_speedup --preset-comparisons --incorrect-convs "$INPUT_DIR"/correctness/1-thread/incorrect-convolutions.csv --exclude pointwise
./summarize_performance.py "$INPUT_DIR"/cdol01-performance-8-threads.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/8-threads --clip-pos --clip-neg --plot-type log2_speedup --preset-comparisons --incorrect-convs "$INPUT_DIR"/correctness/8-threads/incorrect-convolutions.csv --exclude pointwise

./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method Yaconv --new-method ZeroCopy_no_transpose_mkl_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_no_transpose_mkl_jit --exclude grouped --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_no_transpose_HWIO --include torch-heuristic --already-merged

./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type log2_speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_no_transpose_mkl_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type log2_speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_no_transpose_HWIO --include torch-heuristic --already-merged

# copy conv_layers_yaconv_supported.csv from data/ to the input dir
./summarize_performance.py "$INPUT_DIR"/cdol01-performance-1-thread-zconv-blis.csv "$INPUT_DIR"/conv_layers_yaconv_supported.csv "$INPUT_DIR"/paper-graphs/1-thread-zconv-blis --plot-type log2_speedup --clip-pos --clip-neg --old-method Yaconv --new-method ZeroCopy_no_transpose_blis

mkdir -p "$INPUT_DIR"/perf/tables

./summarize_profiler.py "$INPUT_DIR"/perf/perf.txt --incorrect-convs "$INPUT_DIR"/correctness/1-thread/incorrect-convolutions.csv --new-method ZeroCopy_no_transpose_blis --old-method Yaconv --relative > "$INPUT_DIR"/perf/tables/zconv_blis_vs_yaconv.txt
./summarize_profiler.py "$INPUT_DIR"/perf/perf.txt --new-method ZeroCopy_no_transpose_mkl_jit --old-method Im2col --relative > "$INPUT_DIR"/perf/tables/zconv_vs_im2col.txt
./summarize_profiler.py "$INPUT_DIR"/perf/perf.txt --new-method LibTorch_ZeroCopy2D_no_transpose_HWIO --old-method LibTorch --relative > "$INPUT_DIR"/perf/tables/libtorch_zconv_vs_libtorch.txt
