#include "blis/blis.h"
#include <math.h>

// Function to print a matrix
void print_matrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

// Input is in NHWC format
// Filters are in HWIO format
void conv_2d_yaconv_v2_var1(float *__restrict__ input,
                            float *__restrict__ output,
                            float *__restrict__ filters, int N, int H, int W,
                            int C, int FH, int FW, int M, int PH, int PW,
                            int SH, int SW) {
  // Output dimensions
  const int OH = (H + 2 * PH - FH) / SH + 1;
  const int OW = (W + 2 * PW - FW) / SW + 1;

  float *a, *b, *c;
  float *image_buffer =
      (float *)aligned_alloc(4096, OH * FW * C * sizeof(float));

  // Initialize output to zeros
  bli_ssetv(BLIS_NO_CONJUGATE, N * OH * OW * M, bli_s0, output, 1);

  // For every batch element
  for (int n = 0; n < N; ++n) {
    float *single_input = &input[n * H * W * C];

    // For every element in the filter height
    for (int fh = 0; fh < FH; ++fh) {

      // Calculate height slice of size OH and handle edge cases
      int height_offset = fh - PH;
      int height_start;
      if (height_offset < 0) {
        // Use modulo to handle negative values (python implementation of %)
        height_start = bli_max(0, ((height_offset % SH) + SH) % SH);
      } else {
        height_start = bli_max(0, height_offset);
      }
      int height_end = bli_min(H, height_offset + OH * SH);
      int height_slice = ceilf((height_end - height_start) / (float)SH);

      // Print height variables
      // printf("\nheight_offset: %d\n", height_offset);
      // printf("height_start: %d\n", height_start);
      // printf("height_end: %d\n", height_end);
      // printf("height_slice: %d\n", height_slice);

      if (height_slice <= 0)
        continue;

      // For every output width element
      for (int ow = 0; ow < OW; ++ow) {

        // Calculate width slice of size FW and handle edge cases
        int iw = ow * SH - PW;
        int width_start = bli_max(0, iw);
        int width_end = bli_min(W, iw + FW);
        int width_slice = width_end - width_start;

        if (width_slice <= 0)
          continue;

        // Start of the filter block of size 1,FW,C,M
        if (iw < 0) {
          b = &filters[fh * FW * C * M - iw * C * M];
        } else {
          b = &filters[fh * FW * C * M];
        }

        // Copy input slice to image buffer
        int buf_index = 0;
        for (int h = height_start; h < height_end; h += SH) {
          for (int w = width_start; w < width_end; ++w) {
            for (int c = 0; c < C; ++c) {
              image_buffer[buf_index++] = single_input[h * W * C + w * C + c];
            }
          }
        }

        // Start of the image block of size OH,FW,C
        a = image_buffer;

        // Start of the output block of size 1,OH,M
        if (height_offset < 0) {
          int offset = floorf(height_offset / (float)SH);
          c = &output[n * OH * OW * M + ow * OH * M - offset * M];
        } else {
          c = &output[n * OH * OW * M + ow * OH * M];
        }

        int M_dim = height_slice;
        int K_dim = width_slice * C;
        int N_dim = M;
        float alpha = 1.0f;
        float beta = 1.0f;

        // print A and B
        // printf("\nA\n");
        // print_matrix(a, M_dim, K_dim);
        // printf("B\n");
        // print_matrix(b, K_dim, N_dim);

        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M_dim, N_dim, K_dim,
                  &alpha, a, K_dim, 1, b, N_dim, 1, &beta, c, N_dim, 1);
      }
    }
  }

  // Deallocate the image buffer
  free(image_buffer);
}
