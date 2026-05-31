#!/bin/env bash

set -x

# System configuration (edit these values as needed) ---
# - CORE_RANGE   : CPU core range passed to `numactl` (example: "0-7")
# - THREADS      : maximum number of OpenMP threads (used as the upper
#                  bound; scalability runs 1, 2, 4, and 8 threads)
# - REPEAT_COUNT_SCALABILITY: repetitions per (layer, thread-count) pair
# - REPEAT_COUNT_MEMORY     : repetitions per (executable, layer) pair
CORE_RANGE="0-7"
THREADS="8"
REPEAT_COUNT_SCALABILITY="40"
REPEAT_COUNT_MEMORY="40"
# ------------------------------------------

WORKDIR="${HOME}/zero-copy-conv"
BUILD_DIR_MAIN="${HOME}/install/single-conv"
BUILD_DIR_YACONV="${HOME}/install/single-conv-yaconv"

#
# Scalability benchmarks (Layers 1-6, 1/2/4/8 cores)
# Uses the main build dir; benchmark_runner.sh handles per-thread pinning.
# Yaconv is excluded automatically (unsupported in multithreaded mode).
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
# Memory benchmarks (Layers 1-6, single core)
# Run twice: once for the main build (im2col, zconv, libtorch, libtorch-zconv)
# and once for the yaconv build (yaconv, zconv-blis). Results are merged.
#

mkdir -p "${HOME}/results/memory/main"
mkdir -p "${HOME}/results/memory/yaconv"

"${WORKDIR}/memory/run_memory.sh" \
    "${BUILD_DIR_MAIN}" \
    "${HOME}/results/memory/main" \
    "${REPEAT_COUNT_MEMORY}"

"${WORKDIR}/memory/run_memory.sh" \
    "${BUILD_DIR_YACONV}" \
    "${HOME}/results/memory/yaconv" \
    "${REPEAT_COUNT_MEMORY}"

# Merge the two raw CSVs (header from main only)
cat "${HOME}/results/memory/main/memory_raw.csv" > "${HOME}/results/memory/memory_raw.csv"
tail -n +2 "${HOME}/results/memory/yaconv/memory_raw.csv" >> "${HOME}/results/memory/memory_raw.csv"

python3 "${WORKDIR}/memory/summarize_memory.py" \
    "${HOME}/results/memory/memory_raw.csv" \
    --out "${HOME}/results/memory/memory_results.csv"
