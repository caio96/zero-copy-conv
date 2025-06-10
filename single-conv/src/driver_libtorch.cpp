#include "utils.hpp"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <sstream>
#include <torch/torch.h>

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
  if (is_transposed)
    state.SkipWithError("Transposed convolution is not supported!");

  int padding_height = padding_top;
  int padding_width = padding_left;

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size = output_channels * (input_channels / groups) *
                       filter_height * filter_width;

  // Allocate memory for buffers
  float *input =
      static_cast<float *>(aligned_alloc(64, input_size * sizeof(float)));
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

  c10::InferenceMode guard;

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  std::optional<torch::Tensor> bias_tensor = {};
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {output_channels}, tensor_options);
  }

  torch::Tensor input_tensor =
      torch::from_blob(input,
                       {batch, input_channels, input_height, input_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);
  torch::Tensor filters_tensor =
      torch::from_blob(filters,
                       {output_channels, input_channels / groups, filter_height,
                        filter_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);
  torch::Tensor output;

#if defined ZERO_COPY
  // Get PyTorch ZeroCopy2D related environment variables
  bool zc_weights_HWIO;
  set_zero_copy_2d_env_vars(zc_weights_HWIO);

  // Convert filters to HWIO for ZeroCopy2D if enabled
  if (zc_weights_HWIO) {
    filters_tensor = filters_tensor.permute({2, 3, 1, 0}).contiguous();
    filters_tensor = filters_tensor.permute({3, 2, 0, 1});
  }
#endif

  for (auto _ : state) {
#if defined ZERO_COPY
    if (groups == 1 && dilation_h == 1 && dilation_w == 1) {
      output = torch::zero_copy_conv2d(
          input_tensor, filters_tensor, {filter_height, filter_width},
          bias_tensor, {stride_h, stride_w}, {padding_height, padding_width});
    } else {
      output = torch::zero_copy_conv2d_ext(
          input_tensor, filters_tensor, {filter_height, filter_width},
          bias_tensor, {stride_h, stride_w}, {padding_height, padding_width},
          {dilation_h, dilation_w}, groups);
    }
#else
    output = torch::conv2d(
        input_tensor, filters_tensor, bias_tensor, {stride_h, stride_w},
        {padding_height, padding_width}, {dilation_h, dilation_w}, groups);
#endif
  }

  // Clean up
  if (has_bias) {
    free(bias);
  }
  free(input);
  free(filters);
};

int main(int argc, char **argv) {

  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

#if defined ZERO_COPY
  bool zc_weights_HWIO;
  set_zero_copy_2d_env_vars(zc_weights_HWIO);
  std::string name{"LibTorch_ZeroCopy2D_no_transpose"};
  if (zc_weights_HWIO) {
    name += "_HWIO";
  } else {
    name += "_OHWI";
  }
#else
  // Disable zero copy convolution
  std::string false_str = "FALSE";
  setenv("ZC_ENABLE", false_str.c_str(), 1);
  std::string name{"LibTorch"};
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
#ifdef FIXED_ITERATIONS
      ->Iterations(ITERATION_NUMBER)
#else
      ->MinWarmUpTime(0.1)
      ->MinTime(0.5)
#endif
      ->UseRealTime();

  // With argc set to 1, the benchmark library will not parse the command line
  int argc_benchmark = 1;
  benchmark::Initialize(&argc_benchmark, argv);
  benchmark::RunSpecifiedBenchmarks();
}
