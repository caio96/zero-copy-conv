#include <torch/torch.h>

void conv_2d_libtorch(float *__restrict__ input, torch::Tensor &output,
                      float *__restrict__ filters, int batch, int input_height,
                      int input_width, int input_channels, int filter_height,
                      int filter_width, int output_height, int output_width,
                      int output_channels, int padding_height,
                      int padding_width, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups) {

  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32);
  // .memory_format(torch::MemoryFormat::ChannelsLast);

  // Use input memory directly without copying
  auto input_tensor = torch::from_blob(
      input, {batch, input_channels, input_height, input_width},
      tensor_options);

  // Use filter memory directly without copying
  auto weight_tensor = torch::from_blob(
      filters,
      {output_channels, input_channels / groups, filter_height, filter_width},
      tensor_options);

  // Set convolution parameters (padding, stride, dilation)
  torch::nn::Conv2dOptions conv_options =
      torch::nn::Conv2dOptions(input_channels, output_channels,
                               {filter_height, filter_width})
          .stride({stride_h, stride_w})
          .padding({padding_height, padding_width})
          .dilation({dilation_h, dilation_w})
          .groups(groups);

  // Run convolution
  output = torch::conv2d(
      input_tensor, weight_tensor, /*bias=*/{}, {stride_h, stride_w},
      {padding_height, padding_width}, {dilation_h, dilation_w}, groups);
}
