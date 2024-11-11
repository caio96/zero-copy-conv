#include <stdlib.h>

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
                   int filter_width, int output_height, int output_width,
                   int output_channels, int padding_height, int padding_width,
                   int stride_h, int stride_w, int dilation_h, int dilation_w,
                   int groups, float *__restrict__ bias) {
  int input_channels_per_group = input_channels / groups;
  int output_channels_per_group = output_channels / groups;

  // For each output element
  for (int b = 0; b < batch; ++b) {
    float *single_input =
        &input[b * input_channels * input_height * input_width];
    float *single_output =
        &output[b * output_channels * output_height * output_width];

    for (int g = 0; g < groups; ++g) {
      for (int oc = 0; oc < output_channels_per_group; ++oc) {
        int oc_group = g * output_channels_per_group + oc;

        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            float sum = bias != NULL ? bias[oc_group] : 0.0f;

            // For each element in the filter
            for (int ic = 0; ic < input_channels_per_group; ++ic) {
              int ic_group = g * input_channels_per_group + ic;

              for (int fh = 0; fh < filter_height; ++fh) {
                for (int fw = 0; fw < filter_width; ++fw) {
                  // Input height and width
                  int ih = oh * stride_h + fh * dilation_h - padding_height;
                  int iw = ow * stride_w + fw * dilation_w - padding_width;

                  // If the input index is within bounds, get the value
                  // Otherwise, it is zero-padding, so no contribution
                  if (is_a_ge_zero_and_a_lt_b(ih, input_height) &&
                      is_a_ge_zero_and_a_lt_b(iw, input_width)) {
                    // Input and filter indices
                    int input_idx =
                        (ic_group * input_height + ih) * input_width + iw;
                    int filter_idx =
                        ((oc_group * input_channels_per_group + ic) *
                             filter_height +
                         fh) *
                            filter_width +
                        fw;
                    sum += single_input[input_idx] * filters[filter_idx];
                  }
                }
              }
            }

            // Output index
            int output_idx =
                (oc_group * output_height + oh) * output_width + ow;
            single_output[output_idx] = sum;
          }
        }
      }
    }
  }
}
