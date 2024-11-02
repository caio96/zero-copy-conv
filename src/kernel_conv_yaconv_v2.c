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

void print_matrix2(float *matrix, int rsc, int csc, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%f ", matrix[i * rsc + j * csc]);
    }
    printf("\n");
  }
}

int modulo(int a, int b) {
  const int result = a % b;
  return result >= 0 ? result : result + b;
}

// Input is in NHWC format
// Filters are in HWIO format
// Output is in NWHC format (height and width are transposed)
void conv_2d_yaconv_v2_no_copy(float *__restrict__ input,
                               float *__restrict__ output,
                               float *__restrict__ filters, int N, int H, int W,
                               int C, int FH, int FW, int OH, int OW, int M,
                               int PH, int PW, int SH, int SW) {
  float *a, *b, *c;

  // For every batch element
  for (int n = 0; n < N; ++n) {
    float *single_input = &input[n * H * W * C];
    float *single_output = &output[n * OH * OW * M];

    // Initialize output to zeros
    bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, single_output, 1);

    // For every element in the filter height
    for (int fh = 0; fh < FH; ++fh) {

      // Calculate height slice of size OH and handle edge cases
      int height_offset = fh - PH;
      int height_start;
      if (height_offset < 0) {
        height_start = bli_max(0, modulo(height_offset, SH));
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
        int iw = ow * SW - PW;
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

        // Start of the image block of size OH,FW,C
        a = &single_input[height_start * W * C + width_start * C];

        // Start of the output block of size 1,OH,M
        if (height_offset < 0) {
          int offset = floorf(height_offset / (float)SH);
          c = &single_output[ow * OH * M - offset * M];
        } else {
          c = &single_output[ow * OH * M];
        }

        int M_dim = height_slice;
        int K_dim = width_slice * C;
        int N_dim = M;
        float alpha = 1.0f;
        float beta = 1.0f;

        // print A and B
        // printf("\nA\n");
        // print_matrix2(a, W * C * SH, 1, M_dim, K_dim);
        // printf("B\n");
        // print_matrix(b, K_dim, N_dim);

        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M_dim, N_dim, K_dim,
                  &alpha, a, W * C * SH, 1, b, N_dim, 1, &beta, c, N_dim, 1);

        // printf("C\n");
        // print_matrix(single_output, OW, OH);
      }
    }
  }
}

// Input is in NHWC format
// Filters are in HWIO format
// Output is in NWHC format (height and width are transposed)
// This version has additional support for dilated convolution, which requires
// an extra buffer of size OH*FW*C.
void conv_2d_yaconv_v2_copy(float *__restrict__ input,
                            float *__restrict__ output,
                            float *__restrict__ filters, int N, int H, int W,
                            int C, int FH, int FW, int OH, int OW, int M,
                            int PH, int PW, int SH, int SW, int DH, int DW) {
  float *a, *b, *c;
  float *packed_image = aligned_alloc(64, OH * FW * C * sizeof(float));

  // For every batch element
  for (int n = 0; n < N; ++n) {
    float *single_input = &input[n * H * W * C];
    float *single_output = &output[n * OH * OW * M];

    // Initialize output to zeros
    bli_ssetv(BLIS_NO_CONJUGATE, OH * OW * M, bli_s0, single_output,
              1);

    // For every element in the filter height
    for (int fh = 0; fh < FH; ++fh) {

      // Calculate height slice of size OH and handle edge cases
      int height_offset = fh * DH - PH;
      int height_start;
      if (height_offset < 0) {
        height_start = bli_max(0, modulo(height_offset, SH));
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
        int iw = ow * SW - PW;
        int width_start;
        if (iw < 0) {
          width_start = bli_max(0, modulo(iw, DW));
        } else {
          width_start = bli_max(0, iw);
        }
        int width_end = bli_min(W, iw + FW * DW);
        int width_slice = ceilf((width_end - width_start) / (float)DW);

        if (width_slice <= 0)
          continue;

        // Copy input slice to image buffer
        int buf_index = 0;
        for (int h = height_start; h < height_end; h += SH) {
          for (int w = width_start; w < width_end; w += DW) {
            for (int c = 0; c < C; ++c) {
              packed_image[buf_index++] = single_input[h * W * C + w * C + c];
            }
          }
        }

        // Start of the filter block of size 1,FW,C,M
        if (iw < 0) {
          int adjusted_iw = floorf(iw / (float)DW);
          b = &filters[fh * FW * C * M - adjusted_iw * C * M];
        } else {
          b = &filters[fh * FW * C * M];
        }

        // Start of the image block of size OH,FW,C
        a = packed_image;

        // Start of the output block of size 1,OH,M
        if (height_offset < 0) {
          int offset = floorf(height_offset / (float)SH);
          c = &single_output[ow * OH * M - offset * M];
        } else {
          c = &single_output[ow * OH * M];
        }

        int M_dim = height_slice;
        int K_dim = width_slice * C;
        int N_dim = M;
        float alpha = 1.0f;
        float beta = 1.0f;

        // print A and B
        // printf("\nA\n");
        // print_matrix2(a, W * C * SH, DW, M_dim, K_dim);
        // printf("B\n");
        // print_matrix(b, K_dim, N_dim);

        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M_dim, N_dim, K_dim,
                  &alpha, a, K_dim, 1, b, N_dim, 1, &beta, c, N_dim, 1);

        // printf("C\n");
        // print_matrix(single_output, OW, OH);
      }
    }
  }

  free(packed_image);
}

void conv_2d_yaconv_v2(float *__restrict__ input, float *__restrict__ output,
                       float *__restrict__ filters, int N, int H, int W, int C,
                       int FH, int FW, int OH, int OW, int M, int PH, int PW,
                       int SH, int SW, int DH, int DW, int GR) {
  if (DH == 1 && DW == 1) {
    conv_2d_yaconv_v2_no_copy(input, output, filters, N, H, W, C, FH, FW, OH,
                              OW, M, PH, PW, SH, SW);
  } else {
    conv_2d_yaconv_v2_copy(input, output, filters, N, H, W, C, FH, FW, OH, OW,
                           M, PH, PW, SH, SW, DH, DW);
  }
}
