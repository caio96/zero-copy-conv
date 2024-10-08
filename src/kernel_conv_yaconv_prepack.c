#include "blis/blis.h"

// Yaconv convolution
// Input is in NHWC format
// Filters are in HWIO format
void conv_2d_yaconv_prepack(float *__restrict__ input,
                            float *__restrict__ output,
                            float *__restrict__ filters, int batch,
                            int input_height, int input_width,
                            int input_channels, int filter_height,
                            int filter_width, int output_channels,
                            int padding_height, int padding_width, int stride_h,
                            int stride_w) {
  yaconv_prepack(input, batch, input_height, input_width, input_channels,
                 filters, filter_height, filter_width, output_channels, output,
                 padding_height, padding_width);
}
