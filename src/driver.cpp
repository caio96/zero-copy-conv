#include "blis/blis.h"
#include "utils.hpp"
#include <benchmark/benchmark.h>
#include <iostream>
#include <iterator>
#include <sstream>

extern "C" void
conv_2d_naive(float *__restrict__ input, float *__restrict__ output,
              float *__restrict__ filters, int batch, int input_height,
              int input_width, int input_channels, int filter_height,
              int filter_width, int output_height, int output_width,
              int output_channels, int padding_height, int padding_width,
              int stride_h, int stride_w);

extern "C" void
conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w);

extern "C" void
conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w);

extern "C" void
conv_2d_yaconv_v2(float *__restrict__ input, float *__restrict__ output,
                  float *__restrict__ filters, int batch, int input_height,
                  int input_width, int input_channels, int filter_height,
                  int filter_width, int output_height, int output_width,
                  int output_channels, int padding_height, int padding_width,
                  int stride_h, int stride_w);

auto BENCHMARK_CONV2D = [](benchmark::State &state,
                           const std::vector<int> &arguments) {
  // Convolution parameters
  int batch = arguments[0];
  int input_channels = arguments[1];
  int input_height = arguments[2];
  int input_width = arguments[3];
  int output_channels = arguments[4];
  int output_height = arguments[5];
  int output_width = arguments[6];
  int filter_height = arguments[7];
  int filter_width = arguments[8];
  int padding_top = arguments[9];
  int padding_bottom = arguments[10];
  int padding_left = arguments[11];
  int padding_right = arguments[12];
  int stride_h = arguments[13];
  int stride_w = arguments[14];
  int dilation_h = arguments[15];
  int dilation_w = arguments[16];
  int grouped = arguments[17];

  // Ensure that the number of iterations run is at least 10
  state.KeepRunningBatch(10);

  // Output dimensions: This does not always match the arguments if the division
  // is not exact, so we use the argument values instead of the formula.
  // int output_height =
  //     (input_height + 2 * padding_top - filter_height) / stride_h + 1;
  // int output_width =
  //     (input_width + 2 * padding_right - filter_width) / stride_w + 1;

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right)
    state.SkipWithError("Padding height and width do not match!");
  if (dilation_h != 1 || dilation_w != 1)
    state.SkipWithError("Dilation > 1 not supported!");

#if defined YACONV
  if (stride_h > 1 || stride_w > 1)
    state.SkipWithError("Stride > 1 not supported by Yaconv!");
#endif

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
#ifdef NAIVE
    conv_2d_naive(input, output, filters, batch, input_height, input_width,
                  input_channels, filter_height, filter_width, output_height,
                  output_width, output_channels, padding_top, padding_right,
                  stride_h, stride_w);
#elif defined IM2COL
    conv_2d_im2col(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w);
#elif defined YACONV
    conv_2d_yaconv(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w);
#elif defined YACONV_V2
    conv_2d_yaconv_v2(input, output, filters, batch, input_height, input_width,
                      input_channels, filter_height, filter_width,
                      output_height, output_width, output_channels, padding_top,
                      padding_right, stride_h, stride_w);
#else
    state.SkipWithError("Convolution method not defined!");
#endif
  }

  // Clean up
  free(input);
  free(output);
  free(filters);
  bli_finalize();
};

int main(int argc, char **argv) {
  if (argc != 19 && argc != 1) {
    std::cerr << "Usage: " << argv[0]
              << " <Image batch> <Image channel> <Image height> <Image width> "
                 "<Output depth> <Output height> <Output width> <Filter "
                 "height> <Filter width> <Padding top> <Padding bottom> "
                 "<Padding left> <Padding right> <Stride height> <Stride "
                 "width> <Dilation height> <Dilation width> <Grouped>"
              << std::endl;
    return 1;
  }

  std::vector<int> arguments;
  if (argc == 1) {
    // Default arguments
    arguments = {1, 64, 64, 64, 128, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  } else {
    // Command line arguments
    for (int i = 1; i < argc; ++i) {
      arguments.push_back(std::atoi(argv[i]));
    }
  }

#ifdef NAIVE
  std::string name{"Conv2D_Naive"};
#elif defined IM2COL
  std::string name{"Conv2D_Im2col"};
#elif defined YACONV
  std::string name{"Conv2D_Yaconv"};
#elif defined YACONV_V2
  std::string name{"Conv2D_Yaconv_v2"};
#else
  std::string name{"Unknown"};
#endif

  // Transform arguments into a string
  std::stringstream ss;
  ss << name << "/";
  std::copy(arguments.begin(), arguments.end(),
            std::ostream_iterator<int>(ss, "/"));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);

  benchmark::RegisterBenchmark(s, BENCHMARK_CONV2D, arguments)
      ->Unit(benchmark::kMillisecond);

  // With argc set to 1, the benchmark library will not parse the command line
  int argc_benchmark = 1;
  benchmark::Initialize(&argc_benchmark, argv);
  benchmark::RunSpecifiedBenchmarks();
}
