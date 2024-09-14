// Taken from Caffe implementation
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
inline unsigned int is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned int)a < (unsigned int)b;
}

// Naive convolution in Caffe layouts
// Input is in NCHW format
// Filters are in OIHW format
void conv_2d_naive(float *__restrict__ input, float *__restrict__ output,
                   float *__restrict__ filters, int batch, int input_height,
                   int input_width, int input_channels, int filter_height,
                   int filter_width, int output_channels, int padding_height,
                   int padding_width, int stride_h, int stride_w) {

  int output_height =
      (input_height + 2 * padding_height - filter_height) / stride_h + 1;
  int output_width =
      (input_width + 2 * padding_width - filter_width) / stride_w + 1;

  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < output_channels; ++oc) {
      for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
          float sum = 0.0f;

          for (int ic = 0; ic < input_channels; ++ic) {
            for (int fh = 0; fh < filter_height; ++fh) {
              for (int fw = 0; fw < filter_width; ++fw) {
                int ih = oh * stride_h + fh - padding_height;
                int iw = ow * stride_w + fw - padding_width;

                if (is_a_ge_zero_and_a_lt_b(ih, input_height) &&
                    is_a_ge_zero_and_a_lt_b(iw, input_width)) {
                  int input_idx =
                      ((b * input_channels + ic) * input_height + ih) *
                          input_width +
                      iw;
                  int filter_idx =
                      ((oc * input_channels + ic) * filter_height + fh) *
                          filter_width +
                      fw;

                  sum += input[input_idx] * filters[filter_idx];
                }
              }
            }
          }

          int output_idx =
              ((b * output_channels + oc) * output_height + oh) * output_width +
              ow;
          output[output_idx] = sum;
        }
      }
    }
  }
}
