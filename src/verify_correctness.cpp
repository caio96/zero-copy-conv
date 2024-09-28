#include "blis/blis.h"
#include "utils.hpp"
#include <iomanip>
#include <iostream>

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

bool almost_equal(float a, float b) { return std::fabs(a - b) < (0.125); }

// Helper function to compare two arrays element-wise
// Returns true if the arrays match, false otherwise
bool compare_outputs(float *output1, float *output2, size_t size) {

  for (size_t i = 0; i < size; ++i) {
    if (!almost_equal(output1[i], output2[i])) {
      std::cout << std::setprecision(8) << "Mismatch at index " << i << ": "
                << output1[i] << " vs " << output2[i] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  // Convolution test parameters
  int batch = 2;
  int input_channels = 64;
  int input_height = 56;
  int input_width = 56;
  int output_channels = 128;
  int filter_height = 3;
  int filter_width = 3;
  int padding_height = 1;
  int padding_width = 1;
  int stride_h = 1;
  int stride_w = 1;

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

  // Initialize input and filters with random values
  initialize_data(input_NCHW, input_size);
  initialize_data(filters_OIHW, filter_size);

  // Convert input, filters
  NCHW_to_NHWC(input_NCHW, input_NHWC, batch, input_channels, input_height,
               input_width);
  OIHW_to_HWIO(filters_OIHW, filters_HWIO, output_channels, input_channels,
               filter_height, filter_width);

  conv_2d_naive(input_NCHW, output_naive_NCHW, filters_OIHW, batch,
                input_height, input_width, input_channels, filter_height,
                filter_width, output_channels, padding_height, padding_width,
                stride_h, stride_w);
  conv_2d_im2col(input_NCHW, output_im2col, filters_OIHW, batch, input_height,
                 input_width, input_channels, filter_height, filter_width,
                 output_channels, padding_height, padding_width, stride_h,
                 stride_w);
  conv_2d_yaconv(input_NHWC, output_yaconv, filters_HWIO, batch, input_height,
                 input_width, input_channels, filter_height, filter_width,
                 output_channels, padding_height, padding_width, stride_h,
                 stride_w);

  // Verify if the Im2col output match
  bool is_correct =
      compare_outputs(output_naive_NCHW, output_im2col, output_size);
  if (is_correct) {
    std::cout << "Im2col produces the same output." << std::endl;
  } else {
    std::cout << "Im2col produces a different output!" << std::endl;
  }

  // Convert naive output to channel last
  NCHW_to_NHWC(output_naive_NCHW, output_naive_NHWC, batch, output_channels,
               output_height, output_width);

  // Verify if the Yaconv output match
  is_correct = compare_outputs(output_naive_NHWC, output_yaconv, output_size);
  if (is_correct) {
    std::cout << "Yaconv produces the same output." << std::endl;
  } else {
    std::cout << "Yaconv produces a different output!" << std::endl;
  }

  // Clean up
  delete[] input_NCHW;
  delete[] input_NHWC;
  delete[] filters_OIHW;
  delete[] filters_HWIO;
  delete[] output_naive_NCHW;
  delete[] output_im2col;
  delete[] output_yaconv;
  delete[] output_naive_NHWC;
  bli_finalize();
}
