#include "utils.hpp"
#include <benchmark/benchmark.h>

extern "C" void
conv_2d_naive(float *__restrict__ input, float *__restrict__ output,
              float *__restrict__ filters, int batch, int input_height,
              int input_width, int input_channels, int filter_height,
              int filter_width, int output_channels, int padding_height,
              int padding_width, int stride_h, int stride_w);

static void Benchmark_Conv2D_Naive(benchmark::State &state) {
  // Convolution parameters
  int batch = state.range(0);
  int input_channels = state.range(1);
  int input_height = state.range(2);
  int input_width = state.range(3);
  int output_channels = state.range(4);
  int filter_height = state.range(5);
  int filter_width = state.range(6);
  int padding_height = state.range(7);
  int padding_width = state.range(8);
  int stride_h = state.range(9);
  int stride_w = state.range(10);

  // Output dimensions
  int output_height =
      (input_height + 2 * padding_height - filter_height) / stride_h + 1;
  int output_width =
      (input_width + 2 * padding_width - filter_width) / stride_w + 1;

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size =
      output_channels * input_channels * filter_height * filter_width;

  // Allocate memory for buffers
  float *input =
      static_cast<float *>(aligned_alloc(64, input_size * sizeof(float)));
  float *output =
      static_cast<float *>(aligned_alloc(64, output_size * sizeof(float)));
  float *filters =
      static_cast<float *>(aligned_alloc(64, filter_size * sizeof(float)));

  // Initialize input and filters
  initialize_data(input, input_size);
  initialize_data(filters, filter_size);

  for (auto _ : state) {
    conv_2d_naive(input, output, filters, batch, input_height, input_width,
                  input_channels, filter_height, filter_width, output_channels,
                  padding_height, padding_width, stride_h, stride_w);
  }

  // Clean up
  free(input);
  free(output);
  free(filters);
}

BENCHMARK(Benchmark_Conv2D_Naive)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 64, 64, 64, 128, 3, 3, 1, 1, 1, 1}); // Example: Conv layer

BENCHMARK_MAIN();
