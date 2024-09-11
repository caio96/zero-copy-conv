#include "kernel.h"
#include <benchmark/benchmark.h>

extern "C" void conv_2d(int8_t *__restrict__ input, int8_t *__restrict__ output,
             int8_t *__restrict__ filter, int input_height, int input_width,
             int depth, int filter_height,
             int filter_width, int stride);


static void BM_MAIN(benchmark::State &state) {
  int8_t *input;
  int8_t *filter;
  int8_t *output;

  int input_size = 56;
  int filter_size = 3;
  int depth = 16;
  int stride = 1;
  int output_size = (input_size - filter_size) / stride + 1;

  input = (int8_t *)malloc(input_size * input_size * depth * sizeof(int8_t));
  filter = (int8_t *)malloc(filter_size * filter_size * depth * sizeof(int8_t));
  output = (int8_t *)malloc(output_size * output_size * sizeof(int8_t));

  // Run kernel
  for (auto _ : state) {
    conv_2d(input, output, filter, input_size, input_size, depth, filter_size,
          filter_size, stride);
  }

  free(input);
  free(filter);
  free(output);

  return;
}

BENCHMARK(BM_MAIN)
    ->Name("conv-im2col")
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->ReportAggregatesOnly(true);

// Run the benchmark
BENCHMARK_MAIN();
