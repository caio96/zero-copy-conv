#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>

// Initialize data with random values between 0.0 and 1.0
void initialize_data(float *data, size_t size);

// Layout conversion helpers
void NCHW_to_NHWC(float *input, float *output, int batch, int channels,
                  int height, int width);
void NHWC_to_NCHW(float *input, float *output, int batch, int channels,
                  int height, int width);
void yaconv_to_NHWC(float *input, float *output, int batch, int channels,
                    int height, int width, int offset_before, int offset_after);
void OIHW_to_HWIO(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width);
void HWIO_to_OIHW(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width);

// Tensor printers
void print_tensor_NCHW(float *input, int batch, int channels, int height,
                       int width);
void print_tensor_NHWC(float *input, int batch, int channels, int height,
                       int width);
#endif // UTILS_H
