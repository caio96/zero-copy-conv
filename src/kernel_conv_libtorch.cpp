#include <torch/torch.h>

void conv_2d_libtorch(float *__restrict__ input, torch::Tensor &output,
                      float *__restrict__ filters, int batch, int input_height,
                      int input_width, int input_channels, int filter_height,
                      int filter_width, int output_height, int output_width,
                      int output_channels, int padding_height,
                      int padding_width, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups,
                      float *__restrict__ bias) {

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32);

  // Use input memory directly without copying
  torch::Tensor input_tensor = torch::from_blob(
      input, {batch, input_channels, input_height, input_width},
      tensor_options);

  // Use filter memory directly without copying
  torch::Tensor weight_tensor = torch::from_blob(
      filters,
      {output_channels, input_channels / groups, filter_height, filter_width},
      tensor_options);

  std::optional<torch::Tensor> bias_tensor = {};
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {output_channels}, tensor_options);
  }
  output = torch::conv2d(input_tensor, weight_tensor, bias_tensor,
                         {stride_h, stride_w}, {padding_height, padding_width},
                         {dilation_h, dilation_w}, groups);
}
