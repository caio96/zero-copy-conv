#include "utils.hpp"
#include <benchmark/benchmark.h>
#include <dnnl.hpp>
#include <sstream>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

auto BENCHMARK_CONV2D = [](benchmark::State &state,
                           const std::vector<int> &arguments) {
  // Convolution parameters
  int batch = arguments[0];
  int input_channels = arguments[1];
  int input_height = arguments[2];
  int input_width = arguments[3];
  int output_channels = arguments[4];
  int filter_height = arguments[5];
  int filter_width = arguments[6];
  int padding_top = arguments[7];
  int padding_bottom = arguments[8];
  int padding_left = arguments[9];
  int padding_right = arguments[10];
  int stride_h = arguments[11];
  int stride_w = arguments[12];
  int dilation_h = arguments[13];
  int dilation_w = arguments[14];
  int groups = arguments[15];
  int is_transposed = arguments[16];
  int has_bias = arguments[17];

  // Compute output dimensions
  int output_height, output_width;
  compute_output_dims(input_height, input_width, filter_height, filter_width,
                      padding_top, padding_bottom, padding_left, padding_right,
                      stride_h, stride_w, dilation_h, dilation_w, output_height,
                      output_width);

  // Ensure that the number of iterations run is at least 10
  state.KeepRunningBatch(10);

  // Sanity checks
  if (padding_top != padding_bottom || padding_left != padding_right)
    state.SkipWithError("Padding height and width do not match!");
  if (input_channels % groups != 0 || output_channels % groups != 0)
    state.SkipWithError("Input and output channels not divisible by groups!");
  if (is_transposed)
    state.SkipWithError("Transposed convolution is not supported!");

  int padding_height = padding_top;
  int padding_width = padding_left;

  // Buffer sizes
  size_t input_size = batch * input_channels * input_height * input_width;
  size_t output_size = batch * output_channels * output_height * output_width;
  size_t filter_size = output_channels * (input_channels / groups) *
                       filter_height * filter_width;

  // Allocate memory for buffers
  float *input =
      static_cast<float *>(aligned_alloc(64, input_size * sizeof(float)));
  float *output =
      static_cast<float *>(aligned_alloc(64, output_size * sizeof(float)));
  float *filters =
      static_cast<float *>(aligned_alloc(64, filter_size * sizeof(float)));
  float *bias = nullptr;
  if (has_bias) {
    bias = static_cast<float *>(
        aligned_alloc(64, output_channels * sizeof(float)));
  }

  // Initialize input and filters
  initialize_data(input, input_size);
  initialize_data(filters, filter_size);
  if (has_bias) {
    initialize_data(bias, output_channels);
  }

  // Create execution dnnl::engine.
  dnnl::engine engine(engine::kind::cpu, 0);

  // Create dnnl::stream.
  dnnl::stream engine_stream(engine);

  // Source (src), weights, bias, and destination (dst) tensors dimensions.
  memory::dims src_dims = {batch, input_channels, input_height, input_width};
  memory::dims weights_dims = {groups, output_channels / groups,
                               input_channels / groups, filter_height,
                               filter_width};
  memory::dims dst_dims = {batch, output_channels, output_height, output_width};

  // Strides, padding dimensions.
  memory::dims strides_dims = {stride_h, stride_w};
  memory::dims padding_dims_l = {padding_height, padding_width};
  memory::dims padding_dims_r = {padding_height, padding_width};
  memory::dims dilation_dims = {dilation_h - 1, dilation_w - 1};

  // Create memory objects for tensor data (src, weights, dst). The order for
  // image dims is always NCHW, the actual layout is defined with the format
  // tag.
  memory::desc user_src_md = memory::desc(src_dims, dt::f32, tag::nhwc);
  memory::desc user_weights_md =
      memory::desc(weights_dims, dt::f32, tag::hwigo);
  memory::desc user_dst_md = memory::desc(dst_dims, dt::f32, tag::nhwc);
  memory user_src_mem = memory(user_src_md, engine, input);
  memory user_weights_mem = memory(user_weights_md, engine, filters);
  memory user_dst_mem = memory(user_dst_md, engine, output);

  // Create memory descriptor and memory object for input bias.
  memory::dims bias_dims;
  memory::desc user_bias_md;
  memory user_bias_mem;
  if (has_bias) {
    bias_dims = {output_channels};
    user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    user_bias_mem = memory(user_bias_md, engine, bias);
  }

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  memory::desc conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
  memory::desc conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
  memory::desc conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

  primitive_attr conv_attr;

  // Create primitive descriptor.
  convolution_forward::primitive_desc conv_pd =
      convolution_forward::primitive_desc(
          engine, prop_kind::forward_inference, algorithm::convolution_auto,
          conv_src_md, conv_weights_md, user_bias_md, conv_dst_md, strides_dims,
          dilation_dims, padding_dims_l, padding_dims_r, conv_attr);

  // For now, assume that the src, weights, and dst memory layouts generated
  // by the primitive and the ones provided by the user are identical.
  memory conv_src_mem = user_src_mem;
  memory conv_weights_mem = user_weights_mem;
  memory conv_dst_mem = user_dst_mem;

  // Reorder the data in case the src and weights memory layouts generated by
  // the primitive and the ones provided by the user are different. In this
  // case, we create additional memory objects with internal buffers that will
  // contain the reordered data. The data in dst will be reordered after the
  // convolution computation has finalized.
  if (conv_pd.src_desc() != user_src_mem.get_desc()) {
    conv_src_mem = memory(conv_pd.src_desc(), engine);
    reorder(user_src_mem, conv_src_mem)
        .execute(engine_stream, user_src_mem, conv_src_mem);
  }

  if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
    conv_weights_mem = memory(conv_pd.weights_desc(), engine);
    reorder(user_weights_mem, conv_weights_mem)
        .execute(engine_stream, user_weights_mem, conv_weights_mem);
  }

  if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    conv_dst_mem = memory(conv_pd.dst_desc(), engine);
  }

  // Create the primitive.
  convolution_forward conv_prim = convolution_forward(conv_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> conv_args;
  conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
  conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
  conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

  for (auto _ : state) {
    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);

    // // Reorder the data in case the dst memory descriptor generated by the
    // // primitive and the one provided by the user are different.
    // if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    //   reorder(conv_dst_mem, user_dst_mem)
    //       .execute(engine_stream, conv_dst_mem, user_dst_mem);
    // } else {
    //   user_dst_mem = conv_dst_mem;
    // }

    // Wait for the computation to finalize.
    engine_stream.wait();
  }

  // Clean up
  if (has_bias) {
    free(bias);
  }
  free(input);
  free(filters);
  free(output);
};

int main(int argc, char **argv) {

  std::vector<int> arguments;
  int ret = parse_command_line_arguments(argc, argv, arguments);
  if (ret != 0)
    return ret;

  std::string name{"OneDNN_any"};

  // Transform arguments into a string
  std::stringstream ss;
  ss << name << " ";
  std::copy(arguments.begin(), arguments.end(),
            std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);

  benchmark::RegisterBenchmark(s, BENCHMARK_CONV2D, arguments)
      ->Unit(benchmark::kMillisecond)
      ->MeasureProcessCPUTime()
      ->UseRealTime();

  // With argc set to 1, the benchmark library will not parse the command line
  int argc_benchmark = 1;
  benchmark::Initialize(&argc_benchmark, argv);
  benchmark::RunSpecifiedBenchmarks();
}
