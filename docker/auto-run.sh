#!/bin/env bash

set -x
 
# System configuration, update as needed
CORE_RANGE="0-7"
THREADS="8"
CONV_LAYERS_MAX="10" # Use -1 to run all layers from the inputs, otherwise set a max number of layers to run for quicker tests
REPEAT_COUNT="1"
# ---------------------------------------

# Standalone convolution benchmarks -----------
WORKDIR="${HOME}/zero-copy-conv-outside/single-conv" # TODO: modify this
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
"${WORKDIR}/scripts/benchmark_runner.sh" --threads ${THREADS} --core-range ${CORE_RANGE} --repeats ${REPEAT_COUNT} ${HOME}/install/single-conv ${HOME}/results/standalone-conv/multithread/inputs.csv ${HOME}/results/standalone-conv/multithread/outputs.csv

# Summarize multithreaded results
mkdir -p "${HOME}/results/standalone-conv/multithread/summary"
# Im2col vs ZConv
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/multithread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/multithread/summary" --plot-type log2_speedup --clip-pos --clip-neg --old-method Im2col --new-method ZeroCopy_no_transpose_mkl_jit --ignore-significance
# LibTorch vs LibTorch ZConv
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/multithread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/multithread/summary" --plot-type log2_speedup --clip-pos --clip-neg --old-method LibTorch --new-method LibTorch_ZeroCopy2D_no_transpose_HWIO --include torch-heuristic --ignore-significance

# Update parameter for singlethreaded benchmarks (Yaconv)
THREADS="1"
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
"${WORKDIR}/scripts/benchmark_runner.sh" --threads ${THREADS} --core-range ${CORE_RANGE} --repeats ${REPEAT_COUNT} ${HOME}/install/single-conv-yaconv ${HOME}/results/standalone-conv/singlethread/inputs.csv ${HOME}/results/standalone-conv/singlethread/outputs.csv

# Summarize singlethreaded results
mkdir -p "${HOME}/results/standalone-conv/singlethread/summary"
# Yaconv vs ZConv BLIS
"${WORKDIR}/scripts/summarize_performance.py" ${HOME}/results/standalone-conv/singlethread/outputs.csv ${CONV_INPUTS_CSV} "${HOME}/results/standalone-conv/singlethread/summary" --plot-type log2_speedup --clip-pos --clip-neg --old-method Yaconv --new-method ZeroCopy_no_transpose_blis --ignore-significance