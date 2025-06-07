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

#if defined USE_MKL_JIT
#include <mkl.h>
#endif
extern "C" void conv_2d_zero_copy_main(
    float *__restrict__ input, float *__restrict__ output,
    float *__restrict__ filters, int batch, int input_height, int input_width,
    int input_channels, int filter_height, int filter_width, int output_height,
    int output_width, int output_channels, int padding_height,
    int padding_width, int stride_h, int stride_w, int dilation_h,
    int dilation_w, int groups, float *__restrict__ bias, void *jitter);

torch::Tensor
conv_2d_torch(float *__restrict__ input, float *__restrict__ filters, int batch,
              int input_height, int input_width, int input_channels,
              int filter_height, int filter_width, int output_height,
              int output_width, int output_channels, int padding_height,
              int padding_width, int stride_h, int stride_w, int dilation_h,
              int dilation_w, int groups, float *__restrict__ bias) {

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  // Torch input in channel last layout
  torch::Tensor input_tensor =
      torch::from_blob(input,
                       {batch, input_channels, input_height, input_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);

  // Torch filter in channel last layout
  torch::Tensor filters_tensor =
      torch::from_blob(filters,
                       {output_channels, input_channels / groups, filter_height,
                        filter_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);

  // Torch bias if needed
  std::optional<torch::Tensor> bias_tensor = {};
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {output_channels}, tensor_options);
  }

  // Make sure ZeroCopy is disabled
  std::string false_str = "FALSE";
  setenv("ZC_ENABLE", false_str.c_str(), 1);

  // PyTorch reference output
  return torch::conv2d(input_tensor, filters_tensor, bias_tensor,
                       {stride_h, stride_w}, {padding_height, padding_width},
                       {dilation_h, dilation_w}, groups);
}

torch::Tensor conv_2d_torch_zero_copy(
    float *__restrict__ input, float *__restrict__ filters, int batch,
    int input_height, int input_width, int input_channels, int filter_height,
    int filter_width, int output_height, int output_width, int output_channels,
    int padding_height, int padding_width, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int groups, float *__restrict__ bias,
    bool zc_weights_HWIO, bool zc_transform_output) {

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  // Torch input in channel last layout
  torch::Tensor input_tensor =
      torch::from_blob(input,
                       {batch, input_channels, input_height, input_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);

  // Torch filter in channel last layout
  torch::Tensor filters_tensor =
      torch::from_blob(filters,
                       {output_channels, input_channels / groups, filter_height,
                        filter_width},
                       tensor_options)
          .contiguous(torch::MemoryFormat::ChannelsLast);

  // Convert filters to HWIO for ZeroCopy2D if enabled
  if (zc_weights_HWIO) {
    filters_tensor = filters_tensor.permute({2, 3, 1, 0}).contiguous();
    filters_tensor = filters_tensor.permute({3, 2, 0, 1});
  }

  // Torch bias if needed
  std::optional<torch::Tensor> bias_tensor = {};
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {output_channels}, tensor_options);
  }

  // Run PyTorch ZeroCopy2D
  torch::Tensor output_tensor_zc;
  if (groups == 1 && dilation_h == 1 && dilation_w == 1) {
    output_tensor_zc = torch::zero_copy_conv2d(
        input_tensor, filters_tensor, {filter_height, filter_width},
        bias_tensor, {stride_h, stride_w}, {padding_height, padding_width});
  } else {
    output_tensor_zc = torch::zero_copy_conv2d_ext(
        input_tensor, filters_tensor, {filter_height, filter_width},
        bias_tensor, {stride_h, stride_w}, {padding_height, padding_width},
        {dilation_h, dilation_w}, groups);
  }

  // Transpose HW if ZeroCopy2D is enabled and the output transform was disabled
  if (!zc_transform_output) {
    // NCWH -> NCHW (dimension order follows contiguous layout)
    output_tensor_zc = output_tensor_zc.permute({0, 1, 3, 2})
                           .contiguous(torch::MemoryFormat::ChannelsLast);
  }

  return output_tensor_zc;
}

void conv_2d_zero_copy(float *__restrict__ input, float *__restrict__ output,
                       float *__restrict__ filters, int batch, int input_height,
                       int input_width, int input_channels, int filter_height,
                       int filter_width, int output_height, int output_width,
                       int output_channels, int padding_height,
                       int padding_width, int stride_h, int stride_w,
                       int dilation_h, int dilation_w, int groups,
                       float *__restrict__ bias) {
  void *jitter;
#if defined USE_MKL_JIT
  mkl_jit_status_t status;
  if (dilation_h == 1 && dilation_w == 1 && groups == 1) {
    int m_dim = output_height;
    int n_dim = output_channels;
    int k_dim = std::min(filter_width, input_width) * input_channels;
    float alpha = 1.0f;
    int lda = input_width * input_channels * stride_h;
    int ldb = output_channels;
    float beta = 1.0f;
    int ldc = output_channels;
    status =
        mkl_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
                             m_dim, n_dim, k_dim, alpha, lda, ldb, beta, ldc);
  } else {
    int m_dim = output_height;
    int n_dim = output_channels / groups;
    int k_dim = std::min(filter_width, input_width) * input_channels / groups;
    float alpha = 1.0f;
    int lda = std::min(filter_width, input_width) * input_channels / groups;
    int ldb = output_channels;
    float beta = 1.0f;
    int ldc = output_channels;
    status =
        mkl_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
                             m_dim, n_dim, k_dim, alpha, lda, ldb, beta, ldc);
  }

  if (status != MKL_JIT_SUCCESS) {
    jitter = NULL;
  }
#endif
  conv_2d_zero_copy_main(
      input, output, filters, batch, input_height, input_width, input_channels,
      filter_height, filter_width, output_height, output_width, output_channels,
      padding_height, padding_width, stride_h, stride_w, dilation_h, dilation_w,
      groups, bias, jitter);
}

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

void verify_correctness(const std::vector<int> &arguments, bool zc_weights_HWIO,
                        bool zc_transform_output) {
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

  // Name of methods to be checked
  std::vector<std::string> method_names = {"Im2col", "Yaconv", "ZeroCopy",
                                           "Torch_ZeroCopy"};

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
  if (filter_height > input_height + padding_top + padding_bottom ||
      filter_width > input_width + padding_left + padding_right) {
    print_error_for_all(method_names, conv_parameters,
                        "Filter is larger than input with padding!");
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

  // Allocate memory for inputs
  float *input_NCHW = new float[input_size];
  float *input_NHWC = new float[input_size];

  // Allocate memory for filters
  float *filters_OIHW = new float[filter_size];
  float *filters_HWIO = new float[filter_size];

  // Allocate memory for outputs
  float *output_im2col_NCHW = new float[output_size];
  float *output_yaconv_NHWC = new float[output_size];
  float *output_zero_copy_NWHC = new float[output_size];
  float *output_zero_copy_NHWC = new float[output_size];
  const float *output_torch_NCHW;
  const float *output_torch_NHWC;
  const float *output_torch_zc_NHWC;

  // Allocate memory for bias if needed
  float *bias = nullptr;
  if (has_bias) {
    bias = new float[output_channels];
    initialize_data(bias, output_channels);
  }

  // Initialize input and filters
  initialize_data(input_NHWC, input_size);
  initialize_data(filters_HWIO, filter_size);

  // Convert input and filters
  NHWC_to_NCHW(input_NHWC, input_NCHW, batch, input_channels, input_height,
               input_width);
  HWIO_to_OIHW(filters_HWIO, filters_OIHW, output_channels,
               input_channels / groups, filter_height, filter_width);

  // Run PyTorch reference
  torch::Tensor output_tensor_nhwc = conv_2d_torch(
      input_NCHW, filters_OIHW, batch, input_height, input_width,
      input_channels, filter_height, filter_width, output_height, output_width,
      output_channels, padding_height, padding_width, stride_h, stride_w,
      dilation_h, dilation_w, groups, bias).contiguous(torch::MemoryFormat::ChannelsLast);
  output_torch_NHWC = output_tensor_nhwc.const_data_ptr<float>();

  // PyTorch reference output in NCHW layout
  torch::Tensor output_tensor_nchw = output_tensor_nhwc.contiguous();
  output_torch_NCHW = output_tensor_nchw.const_data_ptr<float>();

  // Check if output dimensions match
  if (output_tensor_nhwc.size(0) != batch &&
      output_tensor_nhwc.size(1) != output_channels &&
      output_tensor_nhwc.size(2) != output_height &&
      output_tensor_nhwc.size(3) != output_width) {
    print_error_for_all(method_names, conv_parameters,
                        "Output dimensions do not match with Libtorch!");
    return;
  }

  // Run PyTorch ZeroCopy2D
  torch::Tensor output_tensor_zc = conv_2d_torch_zero_copy(
      input_NCHW, filters_OIHW, batch, input_height, input_width,
      input_channels, filter_height, filter_width, output_height, output_width,
      output_channels, padding_height, padding_width, stride_h, stride_w,
      dilation_h, dilation_w, groups, bias, zc_weights_HWIO,
      zc_transform_output);
  output_torch_zc_NHWC = output_tensor_zc.const_data_ptr<float>();

  // Run Im2col convolution
  conv_2d_im2col(input_NCHW, output_im2col_NCHW, filters_OIHW, batch,
                 input_height, input_width, input_channels, filter_height,
                 filter_width, output_height, output_width, output_channels,
                 padding_height, padding_width, stride_h, stride_w, dilation_h,
                 dilation_w, groups, bias);

  // Run Yaconv convolution
  if (stride_w == 1 && stride_h == 1 && dilation_h == 1 && dilation_w == 1 &&
      groups == 1 && filter_width <= input_width) {
    conv_2d_yaconv(input_NHWC, output_yaconv_NHWC, filters_HWIO, batch,
                   input_height, input_width, input_channels, filter_height,
                   filter_width, output_height, output_width, output_channels,
                   padding_height, padding_width, stride_h, stride_w,
                   dilation_h, dilation_w, groups, bias);
  }

  // Run ZeroCopy convolution
  conv_2d_zero_copy(input_NHWC, output_zero_copy_NWHC, filters_HWIO, batch,
                    input_height, input_width, input_channels, filter_height,
                    filter_width, output_height, output_width, output_channels,
                    padding_height, padding_width, stride_h, stride_w,
                    dilation_h, dilation_w, groups, bias);

  // Transpose HW of zero copy as it flips HW to WH
  NWHC_to_NHWC(output_zero_copy_NWHC, output_zero_copy_NHWC, batch,
               output_channels, output_height, output_width);

  // Print output header
  print_header();
  float diff;

  diff = get_max_diff(output_torch_NCHW, output_im2col_NCHW, output_size);
  print_diff("Im2col", conv_parameters, diff);

  if (dilation_h != 1 || dilation_w != 1) {
    print_error("Yaconv", conv_parameters, "Dilation > 1 not supported");
  } else if (stride_h > 1 || stride_w > 1) {
    print_error("Yaconv", conv_parameters, "Stride > 1 not supported");
  } else if (groups > 1) {
    print_error("Yaconv", conv_parameters, "Grouped convolution not supported");
  } else if (filter_width > input_width) {
    print_error("Yaconv", conv_parameters, "Filter width > input width not supported");
  } else {
    diff = get_max_diff(output_torch_NHWC, output_yaconv_NHWC, output_size);
    print_diff("Yaconv", conv_parameters, diff);
  }

  diff = get_max_diff(output_torch_NHWC, output_zero_copy_NHWC, output_size);
  print_diff("ZeroCopy", conv_parameters, diff);

  diff = get_max_diff(output_torch_NHWC, output_torch_zc_NHWC, output_size);
  print_diff("Torch_ZeroCopy", conv_parameters, diff);

  if (has_bias) {
    delete[] bias;
  }

  delete[] input_NCHW;
  delete[] input_NHWC;
  delete[] filters_OIHW;
  delete[] filters_HWIO;
  delete[] output_im2col_NCHW;
  delete[] output_yaconv_NHWC;
  delete[] output_zero_copy_NWHC;
  delete[] output_zero_copy_NHWC;
}

int main(int argc, char *argv[]) {
  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

  // Get PyTorch ZeroCopy2D related environment variables
  bool zc_weights_HWIO;
  bool zc_transform_output;
  set_zero_copy_2d_env_vars(zc_weights_HWIO, zc_transform_output);

  // Verify correctness
  verify_correctness(arguments, zc_weights_HWIO, zc_transform_output);

  return 0;
}
