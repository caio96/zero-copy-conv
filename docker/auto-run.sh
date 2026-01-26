#!/bin/env bash

set -x

# System configuration (edit these values as needed) ---
# - CORE_RANGE: CPU core range passed to `numactl` (example: "0-7")
# - THREADS: number of OpenMP/OMP threads to use for multithreaded runs
# - CONV_LAYERS_MAX: if >0, limits number of convolution layers run (use -1 to run all)
# - REPEAT_COUNT_CONV: how many times to repeat standalone convolution measurements
# - MODEL_MAX: if >0, limits number of end-to-end models run (use -1 to run all)
# - REPEAT_COUNT_MODEL: how many times to repeat end-to-end model measurements
CORE_RANGE="0-7"
THREADS="8"
CONV_LAYERS_MAX="10"
REPEAT_COUNT_CONV="1"
MODEL_MAX="10"
REPEAT_COUNT_MODEL="1"
# ------------------------------------------

#
# Standalone convolution benchmarks 
#

# Local workspace used by the runner scripts
WORKDIR="${HOME}/zero-copy-conv/single-conv"
# Path to convolution layers input CSV
CONV_INPUTS_CSV="${WORKDIR}/data/conv_layers_all.csv"

# Create results directory for multithreaded benchmarks (all but Yaconv)
mkdir -p "${HOME}/results/standalone-conv/multithread"

# Filter pointwise and error cases
"${WORKDIR}/scripts/filter_csv.py" ${CONV_INPUTS_CSV} ${HOME}/results/standalone-conv/multithread/inputs.csv --exclude-conv-types pointwise

# If CONV_LAYERS_MAX is set to a positive number, limit the number of convolution layers to run
if [[ ${CONV_LAYERS_MAX} -gt 0 ]]; then
    head -n $((CONV_LAYERS_MAX + 1)) ${HOME}/results/standalone-conv/multithread/inputs.csv > ${HOME}/results/standalone-conv/multithread/inputs_temp.csv
    rm ${HOME}/results/standalone-conv/multithread/inputs.csv
    mv ${HOME}/results/standalone-conv/multithread/inputs_temp.csv ${HOME}/results/standalone-conv/multithread/inputs.csv
fi

# Run multithreaded benchmarks
"${WORKDIR}/scripts/benchmark_runner.sh" --threads ${THREADS} --core-range ${CORE_RANGE} --repeats ${REPEAT_COUNT_CONV} ${HOME}/install/single-conv ${HOME}/results/standalone-conv/multithread/inputs.csv ${HOME}/results/standalone-conv/multithread/outputs.csv

# Summarize multithreaded results
mkdir -p "${HOME}/results/standalone-conv/multithread/summary"
# Im2col vs ZConv
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/multithread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/multithread/summary" --plot-type log2_speedup --clip-positive-outliers --clip-negative-outliers --old-method Im2col --new-method ZeroCopy_no_transpose_mkl_jit --ignore-significance
# LibTorch vs LibTorch ZConv
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/multithread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/multithread/summary" --plot-type log2_speedup --clip-positive-outliers --clip-negative-outliers --old-method LibTorch --new-method LibTorch_ZeroCopy2D_no_transpose_HWIO --include-only-conv-types torch-heuristic --ignore-significance

# Update parameter for singlethreaded benchmarks (Yaconv)
THREADS_BACKUP=${THREADS}
THREADS="1"
CONV_INPUTS_CSV_BACKUP="${CONV_INPUTS_CSV}"
CONV_INPUTS_CSV="${WORKDIR}/data/conv_layers_yaconv_supported.csv" # excludes layers not supported by Yaconv and where Yaconv gives errors

# Create results directory for singlethreaded benchmarks (Yaconv)
mkdir -p "${HOME}/results/standalone-conv/singlethread"

# Filter pointwise, error cases, and layers not supported by Yaconv
"${WORKDIR}/scripts/filter_csv.py" ${CONV_INPUTS_CSV} ${HOME}/results/standalone-conv/singlethread/inputs.csv --exclude-conv-types pointwise grouped strided dilated

# If CONV_LAYERS_MAX is set to a positive number, limit the number of convolution layers to run
if [[ ${CONV_LAYERS_MAX} -gt 0 ]]; then
    head -n $((CONV_LAYERS_MAX + 1)) ${HOME}/results/standalone-conv/singlethread/inputs.csv > ${HOME}/results/standalone-conv/singlethread/inputs_temp.csv
    rm ${HOME}/results/standalone-conv/singlethread/inputs.csv
    mv ${HOME}/results/standalone-conv/singlethread/inputs_temp.csv ${HOME}/results/standalone-conv/singlethread/inputs.csv
fi

# Run singlethreaded benchmarks
"${WORKDIR}/scripts/benchmark_runner.sh" --threads ${THREADS} --core-range ${CORE_RANGE} --repeats ${REPEAT_COUNT_CONV} ${HOME}/install/single-conv-yaconv ${HOME}/results/standalone-conv/singlethread/inputs.csv ${HOME}/results/standalone-conv/singlethread/outputs.csv

# Summarize singlethreaded results
mkdir -p "${HOME}/results/standalone-conv/singlethread/summary"
# Yaconv vs ZConv BLIS
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/singlethread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/singlethread/summary" --plot-type log2_speedup --clip-positive-outliers --clip-negative-outliers --old-method Yaconv --new-method ZeroCopy_no_transpose_blis --ignore-significance

#
# End-to-end model benchmarks 
#

THREADS=${THREADS_BACKUP} # Restore threads for multithreaded benchmarks

WORKDIR="${HOME}/zero-copy-conv/end-to-end"
# Path to convolution layers input CSV
CONV_INPUTS_CSV_TORCH="${HOME}/zero-copy-conv/single-conv/data/conv_layers_torch.csv"
CONV_INPUTS_CSV_TIMM="${HOME}/zero-copy-conv/single-conv/data/conv_layers_timm.csv"

# Create results directory for benchmarks
mkdir -p "${HOME}/results/end-to-end/torch"
mkdir -p "${HOME}/results/end-to-end/timm"

# Filter out model that would not run ZConv due to its heuristic
"${WORKDIR}/filter_models.py" ${CONV_INPUTS_CSV_TORCH} --include torch-heuristic > ${HOME}/results/end-to-end/inputs-torch.csv
"${WORKDIR}/filter_models.py" ${CONV_INPUTS_CSV_TIMM} --include torch-heuristic > ${HOME}/results/end-to-end/inputs-timm.csv

if [[ ${MODEL_MAX} -gt 0 ]]; then
    head -n $((MODEL_MAX + 1)) ${HOME}/results/end-to-end/inputs-torch.csv > ${HOME}/results/end-to-end/inputs_temp.csv
    rm ${HOME}/results/end-to-end/inputs-torch.csv
    mv ${HOME}/results/end-to-end/inputs_temp.csv ${HOME}/results/end-to-end/inputs-torch.csv

    head -n $((MODEL_MAX + 1)) ${HOME}/results/end-to-end/inputs-timm.csv > ${HOME}/results/end-to-end/inputs_temp.csv
    rm ${HOME}/results/end-to-end/inputs-timm.csv
    mv ${HOME}/results/end-to-end/inputs_temp.csv ${HOME}/results/end-to-end/inputs-timm.csv
fi

# Run multithreaded benchmarks for Torch models
OMP_NUM_THREADS=${THREADS} numactl -C ${CORE_RANGE} "${WORKDIR}/benchmark_models.py" --repeats ${REPEAT_COUNT_MODEL} --filter-models ${HOME}/results/end-to-end/inputs-torch.csv torch ${HOME}/results/end-to-end/torch/outputs.csv

# Summarize Torch model results
mkdir -p "${HOME}/results/end-to-end/torch/summary"
"${WORKDIR}/summarize_performance_end_to_end.py" ${HOME}/results/end-to-end/torch/outputs.csv "${HOME}/results/end-to-end/torch/summary" --clip-positive-outliers --clip-negative-outliers --preset-comparisons --plot-type speedup --ignore-significance

# Run multithreaded benchmarks for Timm models
OMP_NUM_THREADS=${THREADS} numactl -C ${CORE_RANGE} "${WORKDIR}/benchmark_models.py" --repeats ${REPEAT_COUNT_MODEL} --filter-models ${HOME}/results/end-to-end/inputs-timm.csv timm ${HOME}/results/end-to-end/timm/outputs.csv

# Summarize Timm model results
mkdir -p "${HOME}/results/end-to-end/timm/summary"
"${WORKDIR}/summarize_performance_end_to_end.py" ${HOME}/results/end-to-end/timm/outputs.csv "${HOME}/results/end-to-end/timm/summary" --clip-positive-outliers --clip-negative-outliers --preset-comparisons --plot-type speedup --ignore-significance