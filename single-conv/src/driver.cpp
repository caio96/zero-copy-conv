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
              int groups, float *__restrict__ bias);
#endif

#if defined IM2COL
extern "C" void
conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups, float *__restrict__ bias);
#endif

#if defined YACONV
extern "C" void
conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups, float *__restrict__ bias);
#endif

#if defined ZERO_COPY && defined USE_MKL_JIT
#include <mkl.h>
#endif
#if defined ZERO_COPY
extern "C" void conv_2d_zero_copy_main(
    float *__restrict__ input, float *__restrict__ output,
    float *__restrict__ filters, int batch, int input_height, int input_width,
    int input_channels, int filter_height, int filter_width, int output_height,
    int output_width, int output_channels, int padding_height,
    int padding_width, int stride_h, int stride_w, int dilation_h,
    int dilation_w, int groups, float *__restrict__ bias, void *jitter);
#endif

auto BENCHMARK_CONV2D = [](benchmark::State &state,
                           const std::vector<int> &arguments) {
  // Convolution parameters
  int batch = arguments[0];
  int input_channels = arguments[1];
  int input_height = arguments[2];
  int input_width = arguments[3];
  int output_channels = arguments[4];
  int filter_height = arguments[5];
  int filter_width = arguments[6];
  int padding_top = arguments[7];
  int padding_bottom = arguments[8];
  int padding_left = arguments[9];
  int padding_right = arguments[10];
  int stride_h = arguments[11];
  int stride_w = arguments[12];
  int dilation_h = arguments[13];
  int dilation_w = arguments[14];
  int groups = arguments[15];
  int is_transposed = arguments[16];
  int has_bias = arguments[17];

  // Compute output dimensions
  int output_height, output_width;
  compute_output_dims(input_height, input_width, filter_height, filter_width,
                      padding_top, padding_bottom, padding_left, padding_right,
                      stride_h, stride_w, dilation_h, dilation_w, output_height,
                      output_width);

  // Ensure that the number of iterations run is at least 10
  state.KeepRunningBatch(10);

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right)
    state.SkipWithError("Padding height and width do not match!");
  if (input_channels % groups != 0 || output_channels % groups != 0)
    state.SkipWithError("Input and output channels not divisible by groups!");
  if (filter_height > input_height + padding_top + padding_bottom ||
      filter_width > input_width + padding_left + padding_right)
    state.SkipWithError("Filter is larger than input with padding!");
  if (is_transposed)
    state.SkipWithError("Transposed convolution is not supported!");

#if defined YACONV
  if (stride_h > 1 || stride_w > 1 || dilation_h > 1 || dilation_w > 1 ||
      groups > 1)
    state.SkipWithError(
        "Stride > 1, Dilation > 1, and Groups > 1 not supported by Yaconv!");
  if (filter_width > input_width)
    state.SkipWithError("Filter width > Input width not supported by Yaconv!");
#endif

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size = output_channels * (input_channels / groups) *
                       filter_height * filter_width;

  // Allocate memory for buffers
  float *input =
      static_cast<float *>(aligned_alloc(64, input_size * sizeof(float)));
  float *output =
      static_cast<float *>(aligned_alloc(64, output_size * sizeof(float)));
  float *filters =
      static_cast<float *>(aligned_alloc(64, filter_size * sizeof(float)));

  float *bias = nullptr;
  if (has_bias) {
    bias = static_cast<float *>(
        aligned_alloc(64, output_channels * sizeof(float)));
  }

  // Initialize input and filters
  initialize_data(input, input_size);
  initialize_data(filters, filter_size);
  if (has_bias) {
    initialize_data(bias, output_channels);
  }

#if defined ZERO_COPY
  void *jitter;
#endif
#if defined ZERO_COPY && defined USE_MKL_JIT
  mkl_jit_status_t status;
  if (dilation_h == 1 && dilation_w == 1 && groups == 1) {
    status = mkl_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS,
                                  MKL_NOTRANS, output_height, output_channels,
                                  filter_width * input_channels, 1.0f,
                                  input_width * input_channels * stride_h,
                                  output_channels, 1.0f, output_channels);
  } else {
    status = mkl_jit_create_sgemm(
        &jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, output_height,
        output_channels / groups, filter_width * input_channels / groups, 1.0f,
        filter_width * input_channels / groups, output_channels, 1.0f,
        output_channels);
  }

  if (status != MKL_JIT_SUCCESS) {
    jitter = NULL;
  }
#endif

  for (auto _ : state) {
#ifdef NAIVE
    conv_2d_naive(input, output, filters, batch, input_height, input_width,
                  input_channels, filter_height, filter_width, output_height,
                  output_width, output_channels, padding_top, padding_right,
                  stride_h, stride_w, dilation_h, dilation_w, groups, bias);
#elif defined IM2COL
    conv_2d_im2col(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w, dilation_h, dilation_w, groups, bias);
#elif defined YACONV
    conv_2d_yaconv(input, output, filters, batch, input_height, input_width,
                   input_channels, filter_height, filter_width, output_height,
                   output_width, output_channels, padding_top, padding_right,
                   stride_h, stride_w, dilation_h, dilation_w, groups, bias);
#elif defined ZERO_COPY
    conv_2d_zero_copy_main(
        input, output, filters, batch, input_height, input_width,
        input_channels, filter_height, filter_width, output_height,
        output_width, output_channels, padding_top, padding_right, stride_h,
        stride_w, dilation_h, dilation_w, groups, bias, jitter);
#else
    state.SkipWithError("Convolution method not defined!");
#endif
  }

#if defined ZERO_COPY && defined USE_MKL_JIT
  mkl_jit_destroy(jitter);
#endif

  // Clean up
  if (has_bias) {
    free(bias);
  }
  free(input);
  free(output);
  free(filters);
};

int main(int argc, char **argv) {

  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

#ifdef NAIVE
  std::string name{"Naive"};
#elif defined IM2COL
  std::string name{"Im2col"};
#elif defined YACONV
  std::string name{"Yaconv"};
#elif defined ZERO_COPY && defined USE_MKL_JIT
  std::string name{"ZeroCopy_jit"};
#elif defined ZERO_COPY
  std::string name{"ZeroCopy"};
#else
  std::string name{"Unknown"};
#endif

  // Transform arguments into a string
  std::stringstream ss;
  ss << name << " ";
  std::copy(arguments.begin(), arguments.end(),
            std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);

  benchmark::RegisterBenchmark(s, BENCHMARK_CONV2D, arguments)
      ->Unit(benchmark::kMillisecond)
      ->MeasureProcessCPUTime()
      ->UseRealTime();

  // With argc set to 1, the benchmark library will not parse the command line
  int argc_benchmark = 1;
  benchmark::Initialize(&argc_benchmark, argv);
  benchmark::RunSpecifiedBenchmarks();
}
