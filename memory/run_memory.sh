#!/usr/bin/env bash
# Measure peak resident set size (RSS) for paper memory table.
#
# Usage: ./run_memory.sh BUILD_DIR OUTPUT_DIR [REPEATS]
#   BUILD_DIR  : directory containing the benchmark_* executables
#   OUTPUT_DIR : directory to write memory_raw.csv into (created if absent)
#   REPEATS    : repetitions per (executable, layer) pair (default: 40)
#
# Each run wraps the executable with /usr/bin/time -v and captures the
# "Maximum resident set size" metric. One row is written per run.
# Run summarize_memory.py afterwards to produce the result table.
#
# Notes:
#   - All runs use a single core (numactl --physcpubind 0) since memory
#     usage does not depend on thread count.
#   - benchmark_yaconv and benchmark_zero_copy_blis are included only if
#     present in BUILD_DIR.
#   - Layer 6 (dilated) is skipped for benchmark_yaconv (unsupported).

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
LAYERS_CSV="${SCRIPT_DIR}/layers.csv"

BUILD_DIR=${1:?"Usage: $0 BUILD_DIR OUTPUT_DIR [REPEATS]"}
OUTPUT_DIR=${2:?"Usage: $0 BUILD_DIR OUTPUT_DIR [REPEATS]"}
REPEATS=${3:-10}

BUILD_DIR=$(realpath "$BUILD_DIR")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

OUTPUT_FILE="${OUTPUT_DIR}/memory_raw.csv"
echo "executable,params,rep,max_rss_kb" > "$OUTPUT_FILE"

# Executables to measure (mandatory)
EXECUTABLES=(
    "benchmark_im2col"
    "benchmark_zero_copy"
    "benchmark_libtorch"
    "benchmark_libtorch_zerocopy"
)

# Optional executables (included only if present anywhere in BUILD_DIR)
for OPT in "benchmark_yaconv" "benchmark_zero_copy_blis"; do
    if find "$BUILD_DIR" -type f -name "${OPT}" 2>/dev/null | grep -q .; then
        EXECUTABLES+=("$OPT")
        echo "Found optional executable: $OPT"
    fi
done

# Read layers from CSV (skip header, take only the conv_parameters column)
mapfile -t LAYERS < <(tail -n +2 "$LAYERS_CSV" | cut -d',' -f1)

total=$(( ${#EXECUTABLES[@]} * ${#LAYERS[@]} * REPEATS ))
current=0

for EXECUTABLE in "${EXECUTABLES[@]}"; do
    # Search recursively like benchmark_runner.sh does
    EXE_PATH=$(find "$BUILD_DIR" -type f -name "${EXECUTABLE}" 2>/dev/null | head -1)
    if [[ -z "$EXE_PATH" ]]; then
        echo "Warning: ${EXECUTABLE} not found in $BUILD_DIR — skipping."
        continue
    fi

    for PARAMS in "${LAYERS[@]}"; do
        # Yaconv does not support groups, dilation, or stride > 1.
        # Layer 5 (GR=384) and Layer 6 (DH=DW=12) will SkipWithError,
        # but the binary still runs and the RSS is meaningful (just the
        # process overhead). Include them; summarize_memory.py can filter.
        for rep in $(seq 1 "$REPEATS"); do
            current=$((current + 1))
            printf "\r[%d/%d] %-35s | %s" \
                "$current" "$total" "$EXECUTABLE" "$PARAMS"

            # Capture RSS from /usr/bin/time -v output.
            # Redirect stdout to /dev/null; send all stderr (benchmark
            # warnings + time stats) through the pipe; grep picks the
            # "Maximum resident set size" line and ignores the rest.
            RSS=$( {
                /usr/bin/time -v \
                    numactl --physcpubind 0 \
                    "$EXE_PATH" $PARAMS \
                    >/dev/null
            } 2>&1 | grep "Maximum resident set size" | awk '{print $NF}' || true )

            if [[ -z "$RSS" ]]; then
                RSS="NA"
            fi

            echo "${EXECUTABLE},\"${PARAMS}\",${rep},${RSS}" >> "$OUTPUT_FILE"
        done
    done
done

echo ""
echo "Done. Memory raw results written to: $OUTPUT_FILE"
echo "Run summarize_memory.py $OUTPUT_FILE to generate the result table."
