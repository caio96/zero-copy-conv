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

void NWHC_to_NHWC(float *input, float *output, int batch, int channels,
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

void set_zero_copy_2d_env_vars(bool &weights_HWIO) {
  if (const char *env = std::getenv("ZC_WEIGHTS_LAYOUT")) {
    std::string env_str(env);
    if (env_str == "HWIO") {
      weights_HWIO = true;
    } else if (env_str == "OHWI") {
      weights_HWIO = false;
    } else {
      std::cerr << "Invalid value for ZC_WEIGHTS_LAYOUT: " << env_str
                << std::endl;
    }
  // If the environment variable is not set, use default
  } else {
    weights_HWIO = true;
    std::string hwio_str{"HWIO"};
    setenv("ZC_WEIGHTS_LAYOUT", hwio_str.c_str(), 1);
  }
}

int parse_command_line_arguments(int argc, char **argv,
                                 std::vector<int> &args) {
  if (argc != 19 && argc != 1) {
    std::cerr << "Usage: " << argv[0] << std::endl
              << "\t<Image batch> <Image channels> <Image height>\n\t<Image "
                 "width> <Output channels> <Filter height>\n\t<Filter width> "
                 "<Padding top> <Padding bottom>\n\t<Padding left> <Padding "
                 "right> <Stride height>\n\t<Stride width> <Dilation height> "
                 "<Dilation width>\n\t<Groups> <Is Transposed> <Has Bias>"
              << std::endl
              << " - Or omit all arguments to use a default configuration"
              << std::endl;
    return 1;
  }

  args.resize(18);
  if (argc == 1) {
    // Default arguments
    args = {1, 64, 64, 64, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0};
  } else if (argc == 19) {
    // Command line arguments
    for (int i = 1; i < argc; ++i)
      args[i - 1] = std::atoi(argv[i]);
  }

  return 0;
}
