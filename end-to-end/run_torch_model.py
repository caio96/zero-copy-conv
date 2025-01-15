#!/usr/bin/env python
import argparse
import os
import sys
import time

import torch
import torch.utils.benchmark as benchmark
import torchvision.models as models


# Based on torch/nn/utils/memory_format.py
# Converts weights to HWIO layout so that ZeroCopy2D can execute without changing the weights layout
# This version works without executing the model, but input sizes will be unknown, therefore it can convert the weights
# for a Conv2D layer where ZeroCopy2D won't be used. It won't affect correctness, but may slowdown execution
def convert_conv2d_weights_to_HWIO_static(module):
    if isinstance(module, torch.nn.Conv2d):
        if torch._C._nn.will_use_zero_copy_conv2d_static(
            module.in_channels,
            module.out_channels,
            module.weight,
            module.kernel_size,
            module.bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.transposed,
        ):
            weight_data = module.weight.detach().clone()
            # weight layout to HWIO, making data contiguous (dimension order follow OIHW)
            weight_data = weight_data.permute(2, 3, 1, 0).contiguous()
            # permute dimensions to the order expected by pytorch (OIHW)
            weight_data = weight_data.permute(3, 2, 0, 1)
            module.weight.data = weight_data.resize_(weight_data.size())
    for child in module.children():
        convert_conv2d_weights_to_HWIO_static(child)
    return module


# Converts weights to HWIO layout so that ZeroCopy2D can execute without changing the weights layout
# This function requires running the model once to obtain the input size for each Conv2D layer
# The commented version above works without running the model
def convert_conv2d_weights_to_HWIO_dynamic(model, input):

    def process_layer(module: torch.nn.Conv2d, inputs, output):
        if torch._C._nn.will_use_zero_copy_conv2d_dynamic(
            inputs[0].to(device="cpu"),
            module.weight.to(device="cpu"),
            module.kernel_size,
            module.bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.transposed,
        ):
            weight_data = module.weight.detach().clone()
            # weight layout to HWIO, making data contiguous (dimension order follow OIHW)
            weight_data = weight_data.permute(2, 3, 1, 0).contiguous()
            # permute dimensions to the order expected by pytorch (OIHW)
            weight_data = weight_data.permute(3, 2, 0, 1)
            module.weight.data = weight_data.resize_(weight_data.size())

    # Register hooks to process each Conv2d layer
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(process_layer))

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        model(input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return model


def run_inference(model, input):
    with torch.no_grad():  # Disable gradient calculation
        return model(input)


def run_model(model_name, compile=False, batch=1, convert_weights_to_hwio=False, csv_output=False, csv_header=False):
    # Load model with default weights
    weights = models.get_model_weights(model_name).DEFAULT
    model = models.get_model(model_name, weights=weights)

    # Pre process input with the model's transforms
    dummy_input = torch.randn(3, 224, 224)
    preprocess = weights.transforms()
    dummy_input = preprocess(dummy_input)
    dummy_shape = list(dummy_input.shape)
    dummy_shape = [batch] + dummy_shape
    input = torch.randn(dummy_shape)

    model.eval()  # Set the model to evaluation mode
    model = model.to(
        device="cpu", memory_format=torch.channels_last
    )  # Replace with your model
    input = input.to(
        device="cpu", memory_format=torch.channels_last
    )  # Replace with your input tensor

    if convert_weights_to_hwio:
        # model = convert_conv2d_weights_to_HWIO_static(model)
        model = convert_conv2d_weights_to_HWIO_dynamic(model, input)

    if compile:
        model = torch.compile(model)

    num_threads = torch.get_num_threads()

    t0 = benchmark.Timer(
        stmt="run_inference(model, input)",
        num_threads=num_threads,
        setup="from __main__ import run_inference",
        globals={"model": model, "input": input},
    )
    m0 = t0.adaptive_autorange(min_run_time=1)

    if csv_output:
        if csv_header:
            print("Model,Mean,Median,IQR,Unit,Runs,Threads")
        print(f"{model_name},{m0.mean*1000},{m0.median*1000},{m0.iqr*1000},ms,{len(m0.times)},{num_threads}")
    else:
        print("Model: ", model_name)
        print(m0)


def get_all_models():
    # Exclude models that never use ZeroCopy2D independent from heuristics
    all_models = models.list_models(exclude=["quantized*", "raft*"])
    # Exclude 3D models
    video_models = models.list_models(module=models.video)
    all_models_minus_video = [
        model for model in all_models if model not in video_models
    ]
    return all_models_minus_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a PyTorch model.")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=get_all_models(),
        default="squeezenet1_1",
        help="The name of the model to convert. Default: squeezenet1_1.",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models.",
    )

    parser.add_argument(
        "--zc-enable",
        action="store_true",
        help="Enable ZeroCopy2D in Pytorch.",
    )

    parser.add_argument(
        "--zc-weights-layout",
        type=str,
        default="HWIO",
        choices=["HWIO", "IHWO"],
        help="Weight layout for ZeroCopy2D. IHWO is the channel last layout. HWIO is the layout preferred by ZeroCopy2D."
        "If HWIO is chosen, weights are converted to HWIO for Conv2d layers that will call ZeroCopy2D before inference, otherwise they are kept in channel last layout."
        "Default: HWIO.",
    )

    parser.add_argument(
        "--zc-disable-output-transform",
        action="store_true",
        help="Disable ZeroCopy2D output transformation (NWHC -> NHWC). WARNING: Not yet supported by Pytorch, results may be incorrect",
    )

    parser.add_argument(
        "--zc-disable-heuristic",
        action="store_true",
        help="Disable heuristics that helps choose when to use ZeroCopy2D in Pytorch.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Size of batch.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to compile the model.",
    )

    parser.add_argument(
        "--show-conv-time",
        action="store_true",
        help="Print the execution time of each convolution in the model.",
    )

    args = parser.parse_args()

    if args.list_models:
        for model in get_all_models():
            print("- ", model)
        exit(0)

    if args.zc_enable:
        os.environ["ZC_ENABLE"] = "TRUE"
    else:
        os.environ["ZC_ENABLE"] = "FALSE"

    if args.show_conv_time:
        os.environ["ZC_TIME"] = "TRUE"
    else:
        os.environ["ZC_TIME"] = "FALSE"

    convert_weights_to_hwio = False
    if args.zc_enable and args.zc_weights_layout == "HWIO":
        convert_weights_to_hwio = True

    if args.zc_disable_output_transform:
        os.environ["ZC_TRANSFORM_OUTPUT"] = "FALSE"
    else:
        os.environ["ZC_TRANSFORM_OUTPUT"] = "TRUE"

    if args.zc_disable_heuristic:
        os.environ["ZC_HEURISTIC"] = "FALSE"
    else:
        os.environ["ZC_HEURISTIC"] = "TRUE"

    run_model(args.model_name, args.compile, args.batch_size, convert_weights_to_hwio)
