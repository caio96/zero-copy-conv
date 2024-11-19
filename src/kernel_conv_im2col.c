#if defined USE_MKL
#include <mkl.h>
#else
#include "blis/blis.h"
#endif

// Taken from Caffe implementation
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline unsigned int is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned int)a < (unsigned int)b;
}

// Converts the input image to an im2col matrix
// C,H,W -> C,FH,FW x OH,OW
void im2col(float *__restrict__ input, float *__restrict__ im2col_buffer,
            int input_height, int input_width, int input_channels,
            int filter_height, int filter_width, int output_height,
            int output_width, int padding_height, int padding_width,
            int stride_h, int stride_w, int dilation_h, int dilation_w) {
  int index = 0;
  // For each sliding window location
  for (int ic = 0; ic < input_channels; ++ic) {
    for (int fh = 0; fh < filter_height; ++fh) {
      for (int fw = 0; fw < filter_width; ++fw) {
        // For each output location
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            // Input height and width
            int ih = oh * stride_h + fh * dilation_h - padding_height;
            int iw = ow * stride_w + fw * dilation_w - padding_width;

            // If the input index is within bounds, get the value
            // Otherwise, it is zero-padding
            if (is_a_ge_zero_and_a_lt_b(ih, input_height) &&
                is_a_ge_zero_and_a_lt_b(iw, input_width)) {
              // Input indices
              int input_idx = (ic * input_height + ih) * input_width + iw;
              im2col_buffer[index++] = input[input_idx];
            } else {
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
                    float *__restrict__ filters, int batch, int input_height,
                    int input_width, int input_channels, int filter_height,
                    int filter_width, int output_height, int output_width,
                    int output_channels, int padding_height, int padding_width,
                    int stride_h, int stride_w, int dilation_h, int dilation_w,
                    int groups, float *__restrict__ bias) {

  int pointwise = (filter_height == 1 && filter_width == 1 && stride_h == 1 &&
                   stride_w == 1 && padding_width == 0 && padding_height == 0);

  // Allocate im2col buffer size only if not pointwise
  size_t im2col_size = input_channels * filter_height * filter_width *
                       output_height * output_width;
  float *im2col_buffer;
  if (!pointwise) {
    im2col_buffer = (float *)malloc(im2col_size * sizeof(float));
  }

  // Convolve each batch
  for (int b = 0; b < batch; ++b) {
    // Get the input and output buffer for the current batch
    float *input_buffer =
        &input[b * input_channels * input_height * input_width];
    float *output_buffer =
        &output[b * output_channels * output_height * output_width];

    // Apply im2col to the input
    if (!pointwise) {
      im2col(input_buffer, im2col_buffer, input_height, input_width,
             input_channels, filter_height, filter_width, output_height,
             output_width, padding_height, padding_width, stride_h, stride_w,
             dilation_h, dilation_w);
    } else {
      im2col_buffer = input_buffer;
    }

    // A: filter matrix (OC, IC*FH*FW)
    // B: im2col matrix (IC*FH*FW, OH*OW)
    // C: output matrix (OC, OH*OW)
    int M = output_channels / groups;
    int N = output_height * output_width;
    int K = input_channels / groups * filter_height * filter_width;
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int g = 0; g < groups; ++g) {
      float *a = &filters[g * M * K];
      float *b = &im2col_buffer[g * K * N];
      float *c = &output_buffer[g * M * N];

      // Perform convolution using GEMM
      // C = alpha * A * B + beta * C
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a,
                  K, b, N, beta, c, N);

      if (bias != NULL) {
        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            c[m * N + n] += bias[g * M + m];
          }
        }
      }
    }
  }

  // Deallocate the im2col buffer
  if (!pointwise)
    free(im2col_buffer);
}
