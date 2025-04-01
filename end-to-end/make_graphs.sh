#!/bin/env bash

# Get first argument as the output directory
INPUT_DIR=$1

# Check if the output directory exists
if [[ ! -d $INPUT_DIR || -z $INPUT_DIR ]]; then
    echo "Output directory does not exist"
    exit 1
fi

mkdir -p "$INPUT_DIR"/graphs/torch/1-thread "$INPUT_DIR"/graphs/torch/8-threads
mkdir -p "$INPUT_DIR"/graphs/timm/1-thread "$INPUT_DIR"/graphs/timm/8-threads

./summarize_performance_end_to_end.py "$INPUT_DIR"/timm/end-to-end-1-thread.csv "$INPUT_DIR"/graphs/timm/1-thread --clip-pos --clip-neg --preset --plot-type speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/timm/end-to-end-8-threads.csv "$INPUT_DIR"/graphs/timm/8-threads --clip-pos --clip-neg --preset --plot-type speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/torch/end-to-end-1-thread.csv "$INPUT_DIR"/graphs/torch/1-thread --clip-pos --clip-neg --preset --plot-type speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/torch/end-to-end-8-threads.csv "$INPUT_DIR"/graphs/torch/8-threads --clip-pos --clip-neg --preset --plot-type speedup

./summarize_performance_end_to_end.py "$INPUT_DIR"/timm/end-to-end-1-thread.csv "$INPUT_DIR"/graphs/timm/1-thread --clip-pos --clip-neg --preset --plot-type log2_speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/timm/end-to-end-8-threads.csv "$INPUT_DIR"/graphs/timm/8-threads --clip-pos --clip-neg --preset --plot-type log2_speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/torch/end-to-end-1-thread.csv "$INPUT_DIR"/graphs/torch/1-thread --clip-pos --clip-neg --preset --plot-type log2_speedup
./summarize_performance_end_to_end.py "$INPUT_DIR"/torch/end-to-end-8-threads.csv "$INPUT_DIR"/graphs/torch/8-threads --clip-pos --clip-neg --preset --plot-type log2_speedup
