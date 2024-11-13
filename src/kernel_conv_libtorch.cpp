#include <torch/torch.h>

void conv_2d_libtorch(torch::Tensor &input, torch::Tensor &output,
                      torch::Tensor &filters, int batch, int input_height,
                      int input_width, int input_channels, int filter_height,
                      int filter_width, int output_height, int output_width,
                      int output_channels, int padding_height,
                      int padding_width, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups,
                      std::optional<torch::Tensor> &bias) {
  c10::InferenceMode guard;
  output = torch::conv2d(input, filters, bias, {stride_h, stride_w},
                         {padding_height, padding_width},
                         {dilation_h, dilation_w}, groups);
}
