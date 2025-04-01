#!/bin/env bash

# Get first argument as the output directory
INPUT_DIR=$1

# Check if the output directory exists
if [[ ! -d $INPUT_DIR || -z $INPUT_DIR ]]; then
    echo "Output directory does not exist"
    exit 1
fi

mkdir -p "$INPUT_DIR"/general-graphs/1-thread "$INPUT_DIR"/general-graphs/8-threads
mkdir -p "$INPUT_DIR"/paper-graphs/1-thread "$INPUT_DIR"/paper-graphs/8-threads

./summarize_performance.py "$INPUT_DIR"/cdol01-performance-1-thread.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/1-thread --clip-pos --clip-neg --plot-type log2_speedup --preset-comparisons --incorrect-convs "$INPUT_DIR"/correctness/1-thread/incorrect-convolutions.csv
./summarize_performance.py "$INPUT_DIR"/cdol01-performance-8-threads.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/8-threads --clip-pos --clip-neg --plot-type log2_speedup --preset-comparisons --exclude pointwise
./summarize_performance.py "$INPUT_DIR"/cdol01-performance-1-thread.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/1-thread --clip-pos --clip-neg --plot-type speedup --preset-comparisons --incorrect-convs "$INPUT_DIR"/correctness/1-thread/incorrect-convolutions.csv
./summarize_performance.py "$INPUT_DIR"/cdol01-performance-8-threads.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/general-graphs/8-threads --clip-pos --clip-neg --plot-type speedup --preset-comparisons --exclude pointwise

./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type speedup --clip-pos --clip-neg --old-method Yaconv --new-method ZeroCopy_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_jit --exclude grouped --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_HWIO_TransformOutput --include torch-heuristic --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method Yaconv --new-method ZeroCopy_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_jit --exclude grouped --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/1-thread/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/1-thread --plot-type log2_speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_HWIO_TransformOutput --include torch-heuristic --already-merged

./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_HWIO_TransformOutput --include torch-heuristic --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type log2_speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_jit --already-merged
./summarize_performance.py "$INPUT_DIR"/general-graphs/8-threads/performance-results.csv "$INPUT_DIR"/all.csv "$INPUT_DIR"/paper-graphs/8-threads --plot-type log2_speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_HWIO_TransformOutput --include torch-heuristic --already-merged
