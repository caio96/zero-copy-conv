#if defined IM2COL || defined YACONV || defined ZERO_COPY
#include "blis/blis.h"
#endif

#if defined LIBTORCH
#include <torch/torch.h>
#endif

#include "utils.hpp"
#include <benchmark/benchmark.h>
#include <iterator>
#include <sstream>

#if defined NAIVE
extern "C" void
conv_2d_naive(float *__restrict__ input, float *__restrict__ output,
              float *__restrict__ filters, int batch, int input_height,
              int input_width, int input_channels, int filter_height,
              int filter_width, int output_height, int output_width,
              int output_channels, int padding_height, int padding_width,
              int stride_h, int stride_w, int dilation_h, int dilation_w,
              int groups);
#endif

#if defined IM2COL
extern "C" void
conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups);
#endif

#if defined YACONV
extern "C" void
conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups);
#endif

#if defined ZERO_COPY
extern "C" void
conv_2d_zero_copy_main(float *__restrict__ input, float *__restrict__ output,
                       float *__restrict__ filters, int batch, int input_height,
                       int input_width, int input_channels, int filter_height,
                       int filter_width, int output_height, int output_width,
                       int output_channels, int padding_height,
                       int padding_width, int stride_h, int stride_w,
                       int dilation_h, int dilation_w, int groups);
#endif


#if defined LIBTORCH
// Libtorch allocates in own output, so its output is different
// If output is nullopt, it does not return an output, only executes convolution
// If output is not nullopt, it returns the output in the tensor
void conv_2d_libtorch(float *__restrict__ input, torch::Tensor &output,
                      float *__restrict__ filters, int batch, int input_height,
                      int input_width, int input_channels, int filter_height,
                      int filter_width, int output_height, int output_width,
                      int output_channels, int padding_height,
                      int padding_width, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups);
#endif

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
  int groups = arguments[17];

  // Ensure that the number of iterations run is at least 10
  state.KeepRunningBatch(10);

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right)
    state.SkipWithError("Padding height and width do not match!");
  if (input_channels % groups != 0 || output_channels % groups != 0)
    state.SkipWithError("Input and output channels not divisible by groups!");

#if defined YACONV
  if (stride_h > 1 || stride_w > 1 || dilation_h > 1 || dilation_w > 1 ||
      groups > 1)
    state.SkipWithError(
        "Stride > 1, Dilation > 1, and Groups > 1 not supported by Yaconv!");
#endif

#if defined IM2COL || defined YACONV || defined ZERO_COPY
  bli_init();
#endif

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size = output_channels * (input_channels / groups) *
                       filter_height * filter_width;

  // Allocate memory for buffers
  float *input =
      static_cast<float *>(aligned_alloc(64, input_size * sizeof(float)));
#if defined LIBTORCH
  torch::Tensor output;
#else
  float *output =
      static_cast<float *>(aligned_alloc(64, output_size * sizeof(float)));
#endif
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
                  stride_h, stride_w, dilation_h, dilation_w, groups);
#elif defined IM2COL
    conv_2d_im2col(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w, dilation_h, dilation_w, groups);
#elif defined YACONV
    conv_2d_yaconv(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w, dilation_h, dilation_w, groups);
#elif defined ZERO_COPY
    conv_2d_zero_copy_main(input, output, filters, batch, input_height,
                           input_width, input_channels, filter_height,
                           filter_width, output_height, output_width,
                           output_channels, padding_top, padding_right,
                           stride_h, stride_w, dilation_h, dilation_w, groups);
#elif defined LIBTORCH
    conv_2d_libtorch(input, output, filters, batch, input_height, input_width,
                     input_channels, filter_height, filter_width, output_height,
                     output_width, output_channels, padding_top, padding_right,
                     stride_h, stride_w, dilation_h, dilation_w, groups);
#else
    state.SkipWithError("Convolution method not defined!");
#endif
  }

  // Clean up
  free(input);
#if !defined LIBTORCH
  free(output);
#endif
  free(filters);
#if defined IM2COL || defined YACONV || defined ZERO_COPY
  bli_finalize();
#endif
};

int main(int argc, char **argv) {

  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

#ifdef NAIVE
  std::string name{"Conv2D_Naive"};
#elif defined IM2COL
  std::string name{"Conv2D_Im2col"};
#elif defined YACONV
  std::string name{"Conv2D_Yaconv"};
#elif defined ZERO_COPY
  std::string name{"Conv2D_ZeroCopy"};
#elif defined LIBTORCH
  std::string name{"Conv2D_LibTorch"};
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
