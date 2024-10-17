#include "blis/blis.h"

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
void conv_2d_yaconv_naive(float *__restrict__ input, float *__restrict__ output,
                          float *__restrict__ filters, int N, int H, int W,
                          int C, int FH, int FW, int M, int PH, int PW, int SH,
                          int SW) {
  // Output dimensions
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  float *a, *b, *c;
  float *image_buffer =
      (float *)aligned_alloc(4096, OH * W * C * sizeof(float));
  float *output_buffer =
      (float *)aligned_alloc(4096, OW * OH * M * sizeof(float));

  // For every batch element
  for (int n = 0; n < N; ++n) {
    float *single_input = &input[n * H * W * C];

    // Initialize output to zeros
    bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, output_buffer, 1);

    // For every element in the filter height
    for (int fh = 0; fh < FH; ++fh) {

      // Calculate height slice of size OH and handle edge cases
      int height_offset = fh - PH;
      int height_start = bli_max(0, height_offset);
      int height_end = bli_min(H, height_offset + OH);
      int height_slice = height_end - height_start;

      // Copy input slice to image buffer
      for (int h = height_start, h_buf=0; h < height_end; ++h, ++h_buf) {
        for (int w = 0; w < W; ++w) {
          for (int c = 0; c < C; ++c) {
            image_buffer[h_buf * W * C + w * C + c] =
                single_input[h * W * C + w * C + c];
          }
        }
      }

      // printf("Height start: %d\n", height_start);
      // printf("Height end: %d\n", height_end);
      // printf("input buffer\n");
      // print_matrix(image_buffer, OH, W);
      // printf("\n");

      // For every output width element
      for (int ow = 0; ow < OW; ++ow) {

        // Calculate width slice of size FW and handle edge cases
        int ow_offset = ow - PW;
        int width_start = bli_max(0, ow_offset);
        int width_end = bli_min(W, ow_offset + FW);
        int width_slice = width_end - width_start;

        // Start of the filter block of size 1,FW,C,M
        if (ow_offset < 0) {
          b = &filters[fh * FW * C * M - ow_offset * C * M];
        } else {
          b = &filters[fh * FW * C * M];
        }

        // Start of the image block of size OH,FW,C
        a = &image_buffer[width_start * C];

        // Start of the output block of size 1,OH,M
        if (height_offset < 0) {
          c = &output_buffer[ow * OH * M - height_offset * M];
        } else {
          c = &output_buffer[ow * OH * M];
        }

        int M_dim = height_slice;
        int K_dim = width_slice * C;
        int N_dim = M;
        float alpha = 1.0f;
        float beta = 1.0f;

        // printf("A\n");
        // print_matrix(a, M_dim, K_dim);
        // printf("\n");
        //
        // printf("B\n");
        // print_matrix(b, K_dim, N_dim);
        // printf("\n");
        //
        // printf("C\n");
        // print_matrix(c, M_dim, N_dim);
        // printf("\n");

        // TODO: transpose A
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M_dim, N_dim, K_dim,
                  &alpha, a, W * C, 1, b, N_dim, 1, &beta, c, N, 1);

        // printf("Output\n");
        // print_matrix(output_buffer, OW, OH);
        // printf("\n");
      }
    }

    // Copy output buffer to output
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int m = 0; m < M; ++m) {
          output[n * OH * OW * M + oh * OW * M + ow * M + m] = output_buffer[ow * OH * M + oh * M + m];
        }
      }
    }
  }

  // Deallocate the image buffer
  free(image_buffer);
  free(output_buffer);
}
