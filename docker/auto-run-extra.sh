#!/bin/env bash

set -x

# System configuration (edit these values as needed) ---
# - CORE_RANGE   : CPU core range passed to `numactl` (example: "0-7")
# - THREADS      : maximum number of OpenMP threads; scalability runs
#                  1, 2, 4, and 8 threads up to this limit
# - REPEAT_COUNT_SCALABILITY: repetitions per (layer, thread-count) pair
# - REPEAT_COUNT_MEMORY     : repetitions per (executable, layer) pair
CORE_RANGE="0-7"
THREADS="8"
REPEAT_COUNT_SCALABILITY="10"
REPEAT_COUNT_MEMORY="10"
# ------------------------------------------

WORKDIR="${HOME}/zero-copy-conv"

# Scalability uses the standard builds (variable-time benchmarks)
BUILD_DIR_MAIN="${HOME}/install/single-conv"
BUILD_DIR_YACONV="${HOME}/install/single-conv-yaconv"

# Memory uses fixed-iteration builds so that Im2col allocates its buffer
# a consistent number of times and the allocator pool reaches a steady state
BUILD_DIR_MAIN_FIXED="${HOME}/install/single-conv-fixed"
BUILD_DIR_YACONV_FIXED="${HOME}/install/single-conv-yaconv-fixed"

#
# Scalability benchmarks (Layers 1-6, 1/2/4/8 cores)
#

mkdir -p "${HOME}/results/scalability/raw"

"${WORKDIR}/scalability/run_scalability.sh" \
    "${BUILD_DIR_MAIN}" \
    "${HOME}/results/scalability/raw" \
    "${REPEAT_COUNT_SCALABILITY}"

python3 "${WORKDIR}/scalability/summarize_scalability.py" \
    "${HOME}/results/scalability/raw" \
    --out "${HOME}/results/scalability/data.csv"

#
# Memory benchmarks (Layers 1-6, single core, fixed iterations)
# Run main build (im2col, zconv/mkl, libtorch, libtorch-zconv) then
# yaconv build (benchmark_zero_copy here is the BLIS variant).
#

mkdir -p "${HOME}/results/memory/main"
mkdir -p "${HOME}/results/memory/yaconv"

"${WORKDIR}/memory/run_memory.sh" \
    "${BUILD_DIR_MAIN_FIXED}" \
    "${HOME}/results/memory/main" \
    "${REPEAT_COUNT_MEMORY}"

"${WORKDIR}/memory/run_memory.sh" \
    "${BUILD_DIR_YACONV_FIXED}" \
    "${HOME}/results/memory/yaconv" \
    "${REPEAT_COUNT_MEMORY}"

# Merge: rename the yaconv build's benchmark_zero_copy → benchmark_zero_copy_blis
# so the summarize script correctly separates MKL and BLIS ZConv variants
cat    "${HOME}/results/memory/main/memory_raw.csv" >  "${HOME}/results/memory/memory_raw.csv"
sed 's/^benchmark_zero_copy,/benchmark_zero_copy_blis,/' \
       "${HOME}/results/memory/yaconv/memory_raw.csv" | tail -n +2 >> "${HOME}/results/memory/memory_raw.csv"

python3 "${WORKDIR}/memory/summarize_memory.py" \
    "${HOME}/results/memory/memory_raw.csv" \
    --out "${HOME}/results/memory/memory_results.csv"
