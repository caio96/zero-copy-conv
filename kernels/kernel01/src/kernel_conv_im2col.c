#include "blis/blis.h"

// Converts the input image to an im2col matrix
// C,H,W -> C,FH,FW x OH,OW
void im2col(float *__restrict__ input, float *__restrict__ im2col_buffer,
            int input_height, int input_width, int input_channels,
            int filter_height, int filter_width, int padding_height,
            int padding_width, int stride_h, int stride_w) {

  // Calculate the output dimensions
  int output_height =
      (input_height + 2 * padding_height - filter_height) / stride_h + 1;
  int output_width =
      (input_width + 2 * padding_width - filter_width) / stride_w + 1;

  int index = 0;

  // For each sliding window location
  for (int ic = 0; ic < input_channels; ++ic) {
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        // Loop over the output height and width
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            // Calculate the input index (with padding and stride)
            int ih = oh * stride_h + fh - padding_height;
            int iw = ow * stride_w + fw - padding_width;

            // If the input index is within bounds, get the value
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
              int input_idx = (ic * input_height + ih) * input_width + iw;
              im2col_buffer[index++] = input[input_idx];
            } else {
              // Zero-padding case
              im2col_buffer[index++] = 0.0f;
            }
          }
        }
      }
    }
  }
}

// GEMM-based convolution using im2col
// Input is in NCHW format
// Filters are in OIHW format
void conv_2d_im2col(float *__restrict__ input, float *__restrict__ output,
                    float *__restrict__ filters,
                    float *__restrict__ im2col_buffer, int batch,
                    int input_height, int input_width, int input_channels,
                    int filter_height, int filter_width, int output_channels,
                    int padding_height, int padding_width, int stride_h,
                    int stride_w) {

  // Calculate the output dimensions
  int output_height =
      (input_height + 2 * padding_height - filter_height) / stride_h + 1;
  int output_width =
      (input_width + 2 * padding_width - filter_width) / stride_w + 1;

  // Convolve each batch
  for (int b = 0; b < batch; ++b) {
    float *input_buffer =
        &input[b * input_channels * input_height * input_width];
    float *output_buffer =
        &output[b * output_channels * output_height * output_width];

    // Apply im2col to the input
    // C,H,W -> C,FH,FW x OH,OW
    im2col(input_buffer, im2col_buffer, input_height, input_width,
           input_channels, filter_height, filter_width, padding_height,
           padding_width, stride_h, stride_w);

    // Perform convolution using GEMM
    // C = alpha * A * B + beta * C
    // C: output matrix (output_channels, output_height * output_width)
    // A: filter matrix (output_channels, input_channels * filter_height *
    // filter_width)
    // B: im2col matrix (input_channels * filter_height * filter_width,
    // output_height * output_width)
    int M = output_channels;
    int N = output_height * output_width;
    int K = input_channels * filter_height * filter_width;
    float alpha = 1.0f;
    float beta = 0.0f;

    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, filters, K,
              1, im2col_buffer, N, 1, &beta, output_buffer, N, 1);
  }
}
