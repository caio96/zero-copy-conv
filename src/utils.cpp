#include "utils.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

void initialize_data(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // data[i] = static_cast<float>(i + 1);
  }
}

void NCHW_to_NHWC(const float *input, float *output, int batch, int channels,
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

void NHWC_to_NCHW(const float *input, float *output, int batch, int channels,
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

void transpose_HW(float *input, float *output, int batch, int channels,
                  int height, int width) {
  for (int b = 0; b < batch; ++b) {
    for (int w = 0; w < width; ++w) {
      for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channels; ++c) {
          int idx_output = ((b * height + h) * width + w) * channels + c;
          int idx_input = ((b * width + w) * height + h) * channels + c;
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

void compute_output_dims(int input_height, int input_width, int filter_height,
                         int filter_width, int padding_top, int padding_bottom,
                         int padding_left, int padding_right, int stride_h,
                         int stride_w, int dilation_h, int dilation_w,
                         int &output_height, int &output_width) {

  output_height = floorf((input_height + padding_top + padding_bottom -
                          dilation_h * (filter_height - 1) - 1) /
                             (float)stride_h +
                         1);
  output_width = floorf((input_width + padding_left + padding_right -
                         dilation_w * (filter_width - 1) - 1) /
                            (float)stride_w +
                        1);
}

int parse_command_line_arguments(int argc, char **argv,
                                 std::vector<int> &args) {

  if (argc != 21 && argc != 19 && argc != 1) {
    std::cerr
        << "Usage: " << argv[0] << std::endl
        << "    <Image batch> <Image channels> <Image height> <Image width> "
        << std::endl
        << "    <Output channels> <Output height> <Output width> <Filter height> "
        << std::endl
        << "    <Filter width> <Padding top> <Padding bottom> <Padding left> "
        << std::endl
        << "    <Padding right> <Stride height> <Stride width> <Dilation height> "
        << std::endl
        << "    <Dilation width> <Groups> <Is Transposed> <Has Bias>" << std::endl
        << " - Output height and width can be omitted" << std::endl
        << " - Or omit all arguments to use a default configuration"
        << std::endl;
    return 1;
  }

  args.resize(20);
  if (argc == 1) {
    // Default arguments
    args = {1, 64, 64, 64, 128, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
  } else if (argc == 17) {
    // Command line arguments with automatic output size calculation
    args[0] = std::atoi(argv[1]);
    args[1] = std::atoi(argv[2]);
    args[2] = std::atoi(argv[3]);
    args[3] = std::atoi(argv[4]);
    args[4] = std::atoi(argv[5]);
    args[7] = std::atoi(argv[6]);
    args[8] = std::atoi(argv[7]);
    args[9] = std::atoi(argv[8]);
    args[10] = std::atoi(argv[9]);
    args[11] = std::atoi(argv[10]);
    args[12] = std::atoi(argv[11]);
    args[13] = std::atoi(argv[12]);
    args[14] = std::atoi(argv[13]);
    args[15] = std::atoi(argv[14]);
    args[16] = std::atoi(argv[15]);
    args[17] = std::atoi(argv[16]);
    args[18] = std::atoi(argv[17]);
    args[19] = std::atoi(argv[18]);
    compute_output_dims(args[2], args[3], args[7], args[8], args[9], args[10],
                        args[11], args[12], args[13], args[14], args[15],
                        args[16], args[5], args[6]);
  } else {
    // Command line arguments
    for (int i = 1; i < argc; ++i)
      args[i - 1] = std::atoi(argv[i]);
  }

  return 0;
}
