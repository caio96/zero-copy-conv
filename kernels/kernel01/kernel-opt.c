#include "kernel.h"

void conv_2d(int8_t *__restrict__ input, int8_t *__restrict__ output,
             int8_t *__restrict__ filter, int input_height, int input_width,
             int depth, int filter_height,
             int filter_width, int stride) {
  depth = 16;
  filter_width = 3;
  filter_height = 3;
  stride = 1;
  int output_height = (input_height - filter_height) / stride + 1;
  int output_width = (input_width - filter_width) / stride + 1;

  for (int h = 0; h < output_height; h++) {
    for (int w = 0; w < output_width; w++) {

      int8_t sum_block = 0;
      for (int i = 0; i < filter_height; i++) {
        for (int j = 0; j < filter_width; j++) {
          int input_h = h * stride + i;
          int input_w = w * stride + j;

          if (input_h >= 0 && input_h < input_height && input_w >= 0 &&
              input_w < input_width) {
            for (int d = 0; d < depth; d++) {
              int8_t data1 = input[(input_h * input_width + input_w) * depth + d];
              int8_t data2 = filter[(i * filter_width + j) * depth + d];
              sum_block += data1 * data2;
            }
          }
        }
      }
      output[h * output_width + w] = sum_block;
    }
  }
}
