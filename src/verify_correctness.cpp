#include "utils.hpp"
#include <iostream>
#include <iterator>
#include <sstream>
#include <torch/torch.h>
#include <vector>

extern "C" void
conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups, float *__restrict__ bias);

extern "C" void
conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_height, int output_width,
               int output_channels, int padding_height, int padding_width,
               int stride_h, int stride_w, int dilation_h, int dilation_w,
               int groups, float *__restrict__ bias);

extern "C" void conv_2d_zero_copy_main(
    float *__restrict__ input, float *__restrict__ output,
    float *__restrict__ filters, int batch, int input_height, int input_width,
    int input_channels, int filter_height, int filter_width, int output_height,
    int output_width, int output_channels, int padding_height,
    int padding_width, int stride_h, int stride_w, int dilation_h,
    int dilation_w, int groups, float *__restrict__ bias);

// Returns the maximum difference between two arrays
float get_max_diff(const float *output1, const float *output2, size_t size) {
  float diff = 0.0;
  for (size_t i = 0; i < size; ++i)
    diff = std::max(diff, std::fabs(output1[i] - output2[i]));

  return diff;
}

void print_header() {
  std::string header = {
      "conv_type,conv_parameters,max_diff,error_occurred,error_message"};
  std::cout << header << std::endl;
}

void print_diff(const std::string &method_name,
                const std::string &conv_parameters, float diff) {
  std::cout << std::fixed << method_name << "," << conv_parameters << ","
            << diff << ",,," << std::endl;
}

void print_error(const std::string &method_name,
                 const std::string &conv_parameters,
                 const std::string &message) {
  std::cout << method_name << "," << conv_parameters << ",,true," << message
            << std::endl;
}

void print_error_for_all(std::vector<std::string> &methods,
                         const std::string &conv_parameters,
                         const std::string &message) {
  for (auto name : methods) {
    print_error(name, conv_parameters, message);
  }
}

void verify_correctness(const std::vector<int> &arguments) {
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

  // Transform arguments into a string
  std::stringstream parameters_stream;
  std::copy(arguments.begin(), arguments.end(),
            std::ostream_iterator<int>(parameters_stream, " "));
  std::string conv_parameters = parameters_stream.str();
  conv_parameters = conv_parameters.substr(0, conv_parameters.length() - 1);

  std::vector<std::string> method_names = {"Im2col", "Yaconv", "ZeroCopy"};

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right) {
    print_error_for_all(method_names, conv_parameters,
                        "Unequal padding not supported!");
    return;
  }
  if (input_channels % groups != 0 || output_channels % groups != 0) {
    print_error_for_all(method_names, conv_parameters,
                        "Input and output channels not divisible by groups!");
    return;
  }
  if (is_transposed) {
    print_error_for_all(method_names, conv_parameters,
                        "Transposed convolution is not supported!");
    return;
  }

  int padding_height = padding_top;
  int padding_width = padding_left;

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size = output_channels * (input_channels / groups) *
                       filter_height * filter_width;

  // Allocate memory for input
  float *input_NCHW = new float[input_size];
  float *input_NHWC = new float[input_size];

  // Allocate memory for filters
  float *filters_OIHW = new float[filter_size];
  float *filters_HWIO = new float[filter_size];

  // Allocate memory for outputs
  float *output_im2col = new float[output_size];
  float *output_yaconv = new float[output_size];
  float *output_zero_copy = new float[output_size];
  float *output_zero_copy_transposed = new float[output_size];
  const float *output_torch_NCHW;
  float *output_torch_NHWC = new float[output_size];

  float *bias = nullptr;
  if (has_bias) {
    bias = new float[output_channels];
    initialize_data(bias, output_channels);
  }

  // Initialize input and filters
  initialize_data(input_NHWC, input_size);
  initialize_data(filters_HWIO, filter_size);

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32);

  // Use input memory directly without copying
  torch::Tensor input_tensor = torch::from_blob(
      input_NCHW, {batch, input_channels, input_height, input_width},
      tensor_options);

  // Use filter memory directly without copying
  torch::Tensor filters_tensor = torch::from_blob(
      filters_OIHW,
      {output_channels, input_channels / groups, filter_height, filter_width},
      tensor_options);

  std::optional<torch::Tensor> bias_tensor = {};
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {output_channels}, tensor_options);
  }

  // Convert input and filters
  NHWC_to_NCHW(input_NHWC, input_NCHW, batch, input_channels, input_height,
               input_width);
  HWIO_to_OIHW(filters_HWIO, filters_OIHW, output_channels,
               input_channels / groups, filter_height, filter_width);

  // Run all convolution methods
  torch::Tensor output_tensor = torch::conv2d(
      input_tensor, filters_tensor, bias_tensor, {stride_h, stride_w},
      {padding_height, padding_width}, {dilation_h, dilation_w}, groups);
  // Torch output is used as the reference
  output_torch_NCHW = output_tensor.const_data_ptr<float>();
  if (output_tensor.size(2) != output_height &&
      output_tensor.size(3) != output_width) {
    print_error_for_all(method_names, conv_parameters,
                        "Output dimensions do not match with Libtorch!");
    return;
  }
  conv_2d_im2col(input_NCHW, output_im2col, filters_OIHW, batch, input_height,
                 input_width, input_channels, filter_height, filter_width,
                 output_height, output_width, output_channels, padding_top,
                 padding_right, stride_h, stride_w, dilation_h, dilation_w,
                 groups, bias);
  if (stride_w == 1 && stride_h == 1 && dilation_h == 1 && dilation_w == 1 &&
      groups == 1) {
    conv_2d_yaconv(input_NHWC, output_yaconv, filters_HWIO, batch, input_height,
                   input_width, input_channels, filter_height, filter_width,
                   output_height, output_width, output_channels, padding_top,
                   padding_right, stride_h, stride_w, dilation_h, dilation_w,
                   groups, bias);
  }
  conv_2d_zero_copy_main(
      input_NHWC, output_zero_copy, filters_HWIO, batch, input_height,
      input_width, input_channels, filter_height, filter_width, output_height,
      output_width, output_channels, padding_top, padding_right, stride_h,
      stride_w, dilation_h, dilation_w, groups, bias);

  // Convert naive output to channel last
  NCHW_to_NHWC(output_torch_NCHW, output_torch_NHWC, batch, output_channels,
               output_height, output_width);

  // Transpose HW of zero copy as it flips HW to WH
  transpose_HW(output_zero_copy, output_zero_copy_transposed, batch,
               output_channels, output_height, output_width);

  // Print output header
  print_header();
  float diff;

  diff = get_max_diff(output_torch_NCHW, output_im2col, output_size);
  print_diff("Im2col", conv_parameters, diff);

  if (dilation_h != 1 || dilation_w != 1) {
    print_error("Yaconv", conv_parameters, "Dilation > 1 not supported");
  } else if (stride_h > 1 || stride_w > 1) {
    print_error("Yaconv", conv_parameters, "Stride > 1 not supported");
  } else if (groups > 1) {
    print_error("Yaconv", conv_parameters, "Grouped convolution not supported");
  } else {
    diff = get_max_diff(output_torch_NHWC, output_yaconv, output_size);
    print_diff("Yaconv", conv_parameters, diff);
  }

  diff =
      get_max_diff(output_torch_NHWC, output_zero_copy_transposed, output_size);
  print_diff("ZeroCopy", conv_parameters, diff);

  if (has_bias) {
    delete[] bias;
  }

  delete[] input_NCHW;
  delete[] input_NHWC;
  delete[] filters_OIHW;
  delete[] filters_HWIO;
  delete[] output_im2col;
  delete[] output_yaconv;
  delete[] output_zero_copy;
  delete[] output_zero_copy_transposed;
  delete[] output_torch_NHWC;
}

int main(int argc, char *argv[]) {
  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

  // Verify correctness
  verify_correctness(arguments);

  return 0;
}
