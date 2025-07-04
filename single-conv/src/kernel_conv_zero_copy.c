#if defined USE_MKL
#include <mkl.h>
#else
#include "blis/blis.h"
#endif
#include <math.h>
#include <omp.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

int modulo(int a, int b) {
  const int result = a % b;
  return result >= 0 ? result : result + b;
}

// Performs convolution 2D with zero copies other than the packings that may be
// used in BLIS.
// - Input is in NHWC format
// - Filters are in HWIO format
// - Output is in NHWC format
void conv_2d_zero_copy(float *__restrict__ input, float *__restrict__ output,
                       float *__restrict__ filters, int N, int H, int W, int C,
                       int FH, int FW, int OH, int OW, int M, int PH, int PW,
                       int SH, int SW, float *__restrict__ bias, void *jitter) {

#if defined USE_MKL_JIT
  sgemm_jit_kernel_t jit_gemm = mkl_jit_get_sgemm_ptr(jitter);
#endif

#pragma omp parallel for collapse(2)
  // For every batch element
  for (int n = 0; n < N; ++n) {
    // For every output width element
    for (int ow = 0; ow < OW; ++ow) {

      float *single_input = &input[n * H * W * C];
      float *single_output = &output[n * OH * OW * M];
      float *a, *b, *c;

      // Initialize output to zeros
      for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < M; ++j) {
          if (bias != NULL)
            single_output[i * OW * M + ow * M + j] = bias[j];
          else
            single_output[i * OW * M + ow * M + j] = 0.0f;
        }
      }

      // Calculate width slice of size FW and handle edge cases
      int iw = ow * SW - PW;
      int width_start = max(0, iw);
      int width_end = min(W, iw + FW);
      int width_slice = width_end - width_start;

      if (width_slice <= 0)
        continue;

      // For every element in the filter height
      for (int fh = 0; fh < FH; ++fh) {

        // Calculate height slice of size OH and handle edge cases
        int height_offset = fh - PH;
        int height_start;
        if (height_offset < 0) {
          height_start = max(0, modulo(height_offset, SH));
        } else {
          height_start = height_offset;
        }
        int height_end = min(H, height_offset + OH * SH);
        int height_slice = ceilf((height_end - height_start) / (float)SH);

        if (height_slice <= 0)
          continue;

        // Start of the filter block of size 1,FW,C,M
        if (iw < 0) {
          b = &filters[fh * FW * C * M - iw * C * M];
        } else {
          b = &filters[fh * FW * C * M];
        }

        // Start of the image block of size OH,FW,C
        a = &single_input[height_start * W * C + width_start * C];

        // Start of the output block of size OH,1,M
        if (height_offset < 0) {
          int offset = floorf(height_offset / (float)SH);
          c = &single_output[ow * M - offset * OW * M];
        } else {
          c = &single_output[ow * M];
        }

        int M_dim = height_slice;
        int K_dim = width_slice * C;
        int N_dim = M;
        float alpha = 1.0f;
        float beta = 1.0f;

#if defined USE_MKL_JIT
        if (jitter && height_slice == OH && width_slice == FW) {
          jit_gemm(jitter, a, b, c);
        } else {
#endif
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_dim, N_dim,
                      K_dim, alpha, a, W * C * SH, b, N_dim, beta, c, N_dim * OW);
#if defined USE_MKL_JIT
        }
#endif

      }
    }
  }
}

// This version performs convolution 2D has additional support for dilated
// convolution, which requires an extra buffer of size OH*FW*(C/GR) for every
// thread used.
//
// - Input is in NHWC format
// - Filters are in HWIO format
// - Output is in NHWC format
void conv_2d_zero_copy_ext(float *__restrict__ input,
                           float *__restrict__ output,
                           float *__restrict__ filters, int N, int H, int W,
                           int C, int FH, int FW, int OH, int OW, int M, int PH,
                           int PW, int SH, int SW, int DH, int DW, int GR,
                           float *__restrict__ bias, void *jitter) {
  // Compute channel groupings
  const int C_GR = C / GR;
  const int M_GR = M / GR;

#if defined USE_MKL_JIT
  sgemm_jit_kernel_t jit_gemm = mkl_jit_get_sgemm_ptr(jitter);
#endif

#pragma omp parallel for collapse(2)
  // For every batch element
  for (int n = 0; n < N; ++n) {
    // For every output width element
    for (int ow = 0; ow < OW; ++ow) {

      float *single_input = &input[n * H * W * C];
      float *single_output = &output[n * OH * OW * M];
      float *a, *b, *c;
      float *packed_image =
          (float *)aligned_alloc(64, OH * FW * C_GR * sizeof(float));

      // Initialize output to zeros
      for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < M; ++j) {
          if (bias != NULL)
            single_output[i * OW * M + ow * M + j] = bias[j];
          else
            single_output[i * OW * M + ow * M + j] = 0.0f;
        }
      }

      // Calculate width slice of size FW and handle edge cases
      int iw = ow * SW - PW;
      int width_start;
      if (iw < 0) {
        width_start = max(0, modulo(iw, DW));
      } else {
        width_start = max(0, iw);
      }
      int width_end = min(W, iw + FW * DW);
      int width_slice = ceilf((width_end - width_start) / (float)DW);

      if (width_slice <= 0)
        continue;

      // For every element in the filter height
      for (int fh = 0; fh < FH; ++fh) {

        // Calculate height slice of size OH and handle edge cases
        int height_offset = fh * DH - PH;
        int height_start;
        if (height_offset < 0) {
          height_start = max(0, modulo(height_offset, SH));
        } else {
          height_start = height_offset;
        }
        int height_end = min(H, height_offset + OH * SH);
        int height_slice = ceilf((height_end - height_start) / (float)SH);

        if (height_slice <= 0)
          continue;

        // For every group of channels
        for (int gr = 0; gr < GR; ++gr) {

          // Copy input slice to image buffer following stride
          // height, width dilation, and channel grouping
          int buf_index = 0;
          for (int h = height_start; h < height_end; h += SH) {
            for (int w = width_start; w < width_end; w += DW) {
              for (int c_gr = 0; c_gr < C_GR; ++c_gr) {
                packed_image[buf_index++] =
                    single_input[h * W * C + w * C + c_gr + gr * C_GR];
              }
            }
          }

          // Start of the filter block of size 1,FW,C_GR,M
          if (iw < 0) {
            int adjusted_iw = floorf(iw / (float)DW);
            b = &filters[fh * FW * C_GR * M - adjusted_iw * C_GR * M +
                         gr * M_GR];
          } else {
            b = &filters[fh * FW * C_GR * M + gr * M_GR];
          }

          // Start of the image block of size OH,FW,C_GR
          a = packed_image;

          // Start of the output block of size 1,OH,M
          if (height_offset < 0) {
            int offset = floorf(height_offset / (float)SH);
            c = &single_output[ow * M - offset * OW * M + gr * M_GR];
          } else {
            c = &single_output[ow * M + gr * M_GR];
          }

          int M_dim = height_slice;
          int K_dim = width_slice * C_GR;
          int N_dim = M_GR;
          float alpha = 1.0f;
          float beta = 1.0f;

#if defined USE_MKL_JIT
          if (jitter && height_slice == OH && width_slice == FW) {
            jit_gemm(jitter, a, b, c);
          } else {
#endif
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_dim, N_dim,
                        K_dim, alpha, a, K_dim, b, M, beta, c, M * OW);
#if defined USE_MKL_JIT
          }
#endif
        }
      }
      free(packed_image);
    }
  }
}

void conv_2d_zero_copy_main(float *__restrict__ input,
                            float *__restrict__ output,
                            float *__restrict__ filters, int N, int H, int W,
                            int C, int FH, int FW, int OH, int OW, int M,
                            int PH, int PW, int SH, int SW, int DH, int DW,
                            int GR, float *__restrict__ bias, void *jitter) {
  if (DH == 1 && DW == 1 && GR == 1) {
    conv_2d_zero_copy(input, output, filters, N, H, W, C, FH, FW, OH, OW, M, PH,
                      PW, SH, SW, bias, jitter);
  } else {
    conv_2d_zero_copy_ext(input, output, filters, N, H, W, C, FH, FW, OH, OW, M,
                          PH, PW, SH, SW, DH, DW, GR, bias, jitter);
  }
}
