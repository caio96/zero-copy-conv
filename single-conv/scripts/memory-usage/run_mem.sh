#!/bin/bash

executables=(
    benchmark_im2col
    benchmark_libtorch
    benchmark_libtorch_zerocopy
    benchmark_yaconv
    benchmark_zero_copy
    benchmark_zero_copy_blis
)

conv_params=(
    "1 32 112 112 64 3 3 1 1 1 1 1 1 1 1 1 0 0"
    "1 64 56 56 64 3 3 1 1 1 1 1 1 1 1 1 0 0"
    "1 224 7 7 224 3 3 1 1 1 1 1 1 1 1 1 0 0"
    "1 320 14 14 320 2 2 0 0 0 0 2 2 1 1 1 0 1"
    "1 384 14 14 384 3 3 1 1 1 1 1 1 1 1 384 0 1"
    "1 960 33 33 256 3 3 12 12 12 12 1 1 12 12 1 0 0"
)

N_RUNS=40   # <-- change this

output_dir="benchmark_results"
mkdir -p "$output_dir"

for exe in "${executables[@]}"; do
    log_file="${output_dir}/${exe}.log"
    echo "==== $exe ====" > "$log_file"

    for params in "${conv_params[@]}"; do
        for ((i=1; i<=N_RUNS; i++)); do
            {
                echo "BEGIN_RUN"
                echo "EXECUTABLE: $exe"
                echo "PARAMS: $params"
                echo "RUN_ID: $i"

                OMP_NUM_THREADS=1 /usr/bin/time -v \
                    numactl -C 0 --membind=0 \
                    ./"$exe" $params

                echo "END_RUN"
                echo ""
            } >> "$log_file" 2>&1
        done
    done
done

echo "Done. Logs in $output_dir/"
