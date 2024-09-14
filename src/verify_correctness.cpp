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
               float *__restrict__ filters, float *__restrict__ im2col_buffer,
               int batch, int input_height, int input_width, int input_channels,
               int filter_height, int filter_width, int output_channels,
               int padding_height, int padding_width, int stride_h,
               int stride_w);

bool almost_equal(float a, float b) {
  return std::fabs(a - b) < (0.125);
}

// Helper function to compare two arrays element-wise
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
  // Convolution parameters
  int batch = 1;
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

  // Calculate output dimensions
  int output_height =
      (input_height + 2 * padding_height - filter_height) / stride_h + 1;
  int output_width =
      (input_width + 2 * padding_width - filter_width) / stride_w + 1;

  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size =
      output_channels * input_channels * filter_height * filter_width;
  size_t im2col_size = input_channels * filter_height * filter_width *
                       output_height * output_width;

  // Allocate memory for input, output, filters, and im2col buffer
  float *input = new float[input_size];
  float *output_naive = new float[output_size];
  float *output_im2col = new float[output_size];
  float *filters = new float[filter_size];
  float *im2col_buffer = new float[im2col_size];

  // Initialize input and filters with random values
  initialize_data(input, input_size);
  initialize_data(filters, filter_size);

  // Call convolution implementations and save outputs in different arrays
  conv_2d_naive(input, output_naive, filters, batch, input_height, input_width,
                input_channels, filter_height, filter_width, output_channels,
                padding_height, padding_width, stride_h, stride_w);

  conv_2d_im2col(input, output_im2col, filters, im2col_buffer, batch,
                 input_height, input_width, input_channels, filter_height,
                 filter_width, output_channels, padding_height, padding_width,
                 stride_h, stride_w);

  // Verify if the outputs match
  bool is_correct = compare_outputs(output_naive, output_im2col, output_size);
  // Print the result
  if (is_correct) {
    std::cout << "The two implementations produce the same output."
              << std::endl;
  } else {
    std::cout << "The two implementations produce different outputs!"
              << std::endl;
  }

  // Clean up
  delete[] input;
  delete[] output_naive;
  delete[] output_im2col;
  delete[] filters;
  delete[] im2col_buffer;
  bli_finalize();
}
