#include "utils.hpp"
#include <iomanip>
#include <iostream>

void initialize_data(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void NCHW_to_NHWC(float *input, float *output, int batch, int channels,
                  int height, int width) {
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx_input = ((b * channels + c) * height + h) * width + w;
          int idx_output = ((b * height + h) * width + w) * channels + c;
          output[idx_output] = input[idx_input];
        }
      }
    }
  }
}

void NHWC_to_NCHW(float *input, float *output, int batch, int channels,
                  int height, int width) {
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          int idx_input = ((b * height + h) * width + w) * channels + c;
          int idx_output = ((b * channels + c) * height + h) * width + w;
          output[idx_output] = input[idx_input];
        }
      }
    }
  }
}

void yaconv_to_NHWC(float *input, float *output, int batch, int channels,
                    int height, int width, int offset_before,
                    int offset_after) {
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          int idx_input = (((b * height + h) * width + w) * channels + c) +
                          (b + 1) * offset_before + b * offset_after;
          int idx_output = ((b * height + h) * width + w) * channels + c;
          output[idx_output] = input[idx_input];
        }
      }
    }
  }
}

void OIHW_to_HWIO(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width) {
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int fh = 0; fh < filter_height; ++fh) {
        for (int fw = 0; fw < filter_width; ++fw) {
          int idx_input =
              ((oc * input_channels + ic) * filter_height + fh) * filter_width +
              fw;
          int idx_output = ((fh * filter_width + fw) * input_channels + ic) *
                               output_channels +
                           oc;
          output[idx_output] = input[idx_input];
        }
      }
    }
  }
}

void HWIO_to_OIHW(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width) {
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int ic = 0; ic < input_channels; ++ic) {
      for (int fh = 0; fh < filter_height; ++fh) {
        for (int fw = 0; fw < filter_width; ++fw) {
          int idx_input = ((fh * filter_width + fw) * input_channels + ic) *
                              output_channels +
                          oc;
          int idx_output =
              ((oc * input_channels + ic) * filter_height + fh) * filter_width +
              fw;
          output[idx_output] = input[idx_input];
        }
      }
    }
  }
}

void print_tensor_NCHW(float *input, int batch, int channels, int height,
                       int width) {
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int idx = ((b * channels + c) * height + h) * width + w;
          std::cout << std::setw(4) << std::setprecision(4) << input[idx]
                    << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

void print_tensor_NHWC(float *input, int batch, int channels, int height,
                       int width) {
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channels; ++c) {
          int idx = ((b * height + h) * width + w) * channels + c;
          std::cout << std::setw(4) << std::setprecision(4) << input[idx]
                    << " ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
