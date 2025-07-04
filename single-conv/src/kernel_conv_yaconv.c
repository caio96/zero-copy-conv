#include "blis/blis.h"

// Yaconv convolution
// Input is in NHWC format
// Filters are in HWIO format
void conv_2d_yaconv(float *__restrict__ input, float *__restrict__ output,
                    float *__restrict__ filters, int batch, int input_height,
                    int input_width, int input_channels, int filter_height,
                    int filter_width, int output_height, int output_width,
                    int output_channels, int padding_height, int padding_width,
                    int stride_h, int stride_w, int dilation_h, int dilation_w,
                    int groups, float *__restrict__ bias) {
  yaconv(input, batch, input_height, input_width, input_channels, filters,
         filter_height, filter_width, output_height, output_width,
         output_channels, output, padding_height, padding_width, bias);
}
