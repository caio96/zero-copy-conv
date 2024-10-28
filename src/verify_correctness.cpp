#include "blis/blis.h"
#include "utils.hpp"
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

extern "C" void
conv_2d_naive(float *__restrict__ input, float *__restrict__ output,
              float *__restrict__ filters, int batch, int input_height,
              int input_width, int input_channels, int filter_height,
              int filter_width, int output_channels, int padding_height,
              int padding_width, int stride_h, int stride_w);

extern "C" void
conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_channels, int padding_height,
               int padding_width, int stride_h, int stride_w);

extern "C" void
conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
               float *__restrict__ filters, int batch, int input_height,
               int input_width, int input_channels, int filter_height,
               int filter_width, int output_channels, int padding_height,
               int padding_width, int stride_h, int stride_w);

extern "C" void
conv_2d_yaconv_v2(float *__restrict__ input, float *__restrict__ output,
                  float *__restrict__ filters, int batch, int input_height,
                  int input_width, int input_channels, int filter_height,
                  int filter_width, int output_channels, int padding_height,
                  int padding_width, int stride_h, int stride_w);

// Returns the maximum difference between two arrays
float get_max_diff(float *output1, float *output2, size_t size) {
  float diff = 0.0;
  for (size_t i = 0; i < size; ++i)
    diff = std::max(diff, std::fabs(output1[i] - output2[i]));

  return diff;
}

void print_header() {
  std::string header = {"name,max_diff,error_occurred,error_message"};
  std::cout << header << std::endl;
}

void print_diff(const std::string &method_name, float diff) {
  std::cout << std::fixed << method_name << "," << diff << ",,," << std::endl;
}

void print_error(const std::string &method_name, const std::string &message) {
  std::cout << method_name << ",,true," << message << std::endl;
}

void print_error_for_all(std::vector<std::string> &methods,
                         const std::string &message) {
  for (auto name : methods) {
    print_error(name, message);
  }
}

void verify_correctness(const std::vector<int> &arguments) {
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

  // Output dimensions
  // int output_height =
  //     (input_height + 2 * padding_top - filter_height) / stride_h + 1;
  // int output_width =
  //     (input_width + 2 * padding_right - filter_width) / stride_w + 1;

  // Transform arguments into a string
  std::stringstream parameters;
  std::copy(arguments.begin(), arguments.end(),
            std::ostream_iterator<int>(parameters, "/"));
  std::string s = parameters.str();
  s = s.substr(0, s.length() - 1);

  std::vector<std::string> method_names = {"Im2col", "Yaconv", "Yaconv V2"};

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right) {
    print_error_for_all(method_names, "Unequal padding not supported!");
    return;
  }

  if (dilation_h != 1 || dilation_w != 1) {
    print_error_for_all(method_names, "Dilation > 1 not supported!");
    return;
  }

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size =
      output_channels * input_channels * filter_height * filter_width;

  // Allocate memory for input
  float *input_NCHW = new float[input_size];
  float *input_NHWC = new float[input_size];

  // Allocate memory for filters
  float *filters_OIHW = new float[filter_size];
  float *filters_HWIO = new float[filter_size];

  // Allocate memory for outputs
  float *output_naive_NCHW = new float[output_size];
  float *output_naive_NHWC = new float[output_size];
  float *output_im2col = new float[output_size];
  float *output_yaconv = new float[output_size];
  float *output_yaconv_v2 = new float[output_size];
  float *output_yaconv_v2_transposed = new float[output_size];

  // Initialize input and filters
  initialize_data(input_NHWC, input_size);
  initialize_data(filters_HWIO, filter_size);

  // Convert input and filters
  NHWC_to_NCHW(input_NHWC, input_NCHW, batch, input_channels, input_height,
               input_width);
  HWIO_to_OIHW(filters_HWIO, filters_OIHW, output_channels, input_channels,
               filter_height, filter_width);

  // Run all convolution methods
  conv_2d_naive(input_NCHW, output_naive_NCHW, filters_OIHW, batch,
                input_height, input_width, input_channels, filter_height,
                filter_width, output_channels, padding_top, padding_right,
                stride_h, stride_w);
  conv_2d_im2col(input_NCHW, output_im2col, filters_OIHW, batch, input_height,
                 input_width, input_channels, filter_height, filter_width,
                 output_channels, padding_top, padding_right, stride_h,
                 stride_w);
  if (stride_w == 1 && stride_h == 1) {
    conv_2d_yaconv(input_NHWC, output_yaconv, filters_HWIO, batch, input_height,
                   input_width, input_channels, filter_height, filter_width,
                   output_channels, padding_top, padding_right, stride_h,
                   stride_w);
  }
  conv_2d_yaconv_v2(input_NHWC, output_yaconv_v2, filters_HWIO, batch,
                    input_height, input_width, input_channels, filter_height,
                    filter_width, output_channels, padding_top,
                    padding_right, stride_h, stride_w);

  // Convert naive output to channel last
  NCHW_to_NHWC(output_naive_NCHW, output_naive_NHWC, batch, output_channels,
               output_height, output_width);

  // Transpose HW of yaconv_v2 as it flips HW to WH
  transpose_HW(output_yaconv_v2, output_yaconv_v2_transposed, batch,
               output_channels, output_height, output_width);

  // Print output header
  print_header();
  float diff;

  diff = get_max_diff(output_naive_NCHW, output_im2col, output_size);
  print_diff("Im2col", diff);

  if (stride_w == 1 && stride_h == 1) {
    diff = get_max_diff(output_naive_NHWC, output_yaconv, output_size);
    print_diff("Yaconv", diff);
  } else {
    print_error("Yaconv", "Stride > 1 not supported");
  }

  diff = get_max_diff(output_naive_NHWC, output_yaconv_v2_transposed, output_size);
  print_diff("Yaconv_v2", diff);
}

int main(int argc, char *argv[]) {
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

  // Initialize BLIS
  bli_init();

  // Verify correctness
  verify_correctness(arguments);

  bli_finalize();
  return 0;
}
