#!/usr/bin/env bash
# Collect scalability data for all 6 layers across 1, 2, 4, and 8 cores.
#
# Usage: ./run_scalability.sh BUILD_DIR OUTPUT_DIR [REPEATS]
#   BUILD_DIR  : directory containing the benchmark_* executables
#   OUTPUT_DIR : directory to write raw CSVs into (created if absent)
#   REPEATS    : repetitions per configuration (default: 10)
#
# Outputs one CSV per core count: OUTPUT_DIR/raw_1threads.csv, raw_2threads.csv, ...
# Run summarize_scalability.py afterwards to produce data.csv.

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RUNNER="${SCRIPT_DIR}/../single-conv/scripts/benchmark_runner.sh"
LAYERS_CSV="${SCRIPT_DIR}/layers.csv"

BUILD_DIR=${1:?"Usage: $0 BUILD_DIR OUTPUT_DIR [REPEATS]"}
OUTPUT_DIR=${2:?"Usage: $0 BUILD_DIR OUTPUT_DIR [REPEATS]"}
REPEATS=${3:-10}

BUILD_DIR=$(realpath "$BUILD_DIR")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

for THREADS in 1 2 4 8; do
    CORE_MAX=$((THREADS - 1))
    OUTPUT="${OUTPUT_DIR}/raw_${THREADS}threads.csv"

    echo "=== Running with ${THREADS} thread(s), cores 0-${CORE_MAX}, ${REPEATS} repeats ==="

    rm -f "$OUTPUT"
    bash "$RUNNER" "$BUILD_DIR" "$LAYERS_CSV" "$OUTPUT" \
        --threads "$THREADS" \
        --core-range "0-${CORE_MAX}" \
        --repeats "$REPEATS"
done

echo ""
echo "Done. Raw CSVs written to: $OUTPUT_DIR"
echo "Run summarize_scalability.py $OUTPUT_DIR to generate data.csv"
