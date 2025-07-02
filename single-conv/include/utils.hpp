#ifndef UTILS_H
#define UTILS_H

#define ITERATION_NUMBER 100

#include <cstdlib>
#include <vector>

// Initialize data with random values between 0.0 and 1.0
void initialize_data(float *data, size_t size);

// Layout conversion helpers
void NCHW_to_NHWC(const float *input, float *output, int batch, int channels,
                  int height, int width);
void NHWC_to_NCHW(const float *input, float *output, int batch, int channels,
                  int height, int width);
void yaconv_to_NHWC(float *input, float *output, int batch, int channels,
                    int height, int width, int offset_before, int offset_after);
void NWHC_to_NHWC(float *input, float *output, int batch, int channels,
                  int height, int width);
void OIHW_to_HWIO(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width);
void HWIO_to_OIHW(float *input, float *output, int output_channels,
                  int input_channels, int filter_height, int filter_width);

// Tensor printers
void print_tensor_NCHW(float *input, int batch, int channels, int height,
                       int width);
void print_tensor_NHWC(float *input, int batch, int channels, int height,
                       int width);

void compute_output_dims(int input_height, int input_width, int filter_height,
                         int filter_width, int padding_top, int padding_bottom,
                         int padding_left, int padding_right, int stride_h,
                         int stride_w, int dilation_h, int dilation_w,
                         int &output_height, int &output_width);

void set_zero_copy_2d_env_vars(bool &weights_HWIO, bool &transform_output);

// Parse command line
int parse_command_line_arguments(int argc, char **argv,
                                 std::vector<int> &arguments);

#endif // UTILS_H
