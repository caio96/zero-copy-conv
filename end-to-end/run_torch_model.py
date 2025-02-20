#!/usr/bin/env python
import argparse
import os
import sys
import timm
import time
import random
import numpy as np
from tabulate import tabulate
import csv
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark
import torchvision.models as models


def get_model_and_input(model_name, source, batch_size=1):
    if source == "torch":
        # Get model and weights
        weights = models.get_model_weights(model_name).DEFAULT
        model = models.get_model(model_name, weights=weights)
        # Pre process input with the model's transforms
        dummy_input = torch.randn(3, 224, 224)
        preprocess = weights.transforms()
        input = preprocess(dummy_input)
        # Add batch dimension
        input = input.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    elif source == "timm":
        # Get model
        model = timm.create_model(model_name)
        # Get input
        input_shape = (batch_size, *model.default_cfg["input_size"])
        input = torch.randn(input_shape)
    else:
        print("Unknown source", file=sys.stderr)
        sys.exit(1)

    return model, input


# Based on torch/nn/utils/memory_format.py
# Converts weights to HWIO layout so that ZeroCopy2D can execute without changing the weights layout
# This version works without executing the model, but input sizes will be unknown, therefore it can convert the weights
# for a Conv2D layer where ZeroCopy2D won't be used. It won't affect correctness, but may slowdown execution
def convert_conv2d_weights_to_HWIO_static(model):
    for name, module in model.named_modules():
        if type(module) is torch.nn.Conv2d:
            if torch._C._nn.will_use_zero_copy_conv2d_static(
                module.in_channels,
                module.out_channels,
                module.weight,
                module.stride,
                (module._reversed_padding_repeated_twice[-1], module._reversed_padding_repeated_twice[-3]),
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
    return model


# Converts weights to HWIO layout so that ZeroCopy2D can execute without changing the weights layout
# This function requires running the model once to obtain the input size for each Conv2D layer
# The commented version above works without running the model
def convert_conv2d_weights_to_HWIO_dynamic(model, input):

    def process_layer(module: torch.nn.Conv2d, inputs):
        if torch._C._nn.will_use_zero_copy_conv2d_dynamic(
            inputs[0],
            module.weight,
            module.stride,
            (module._reversed_padding_repeated_twice[-1], module._reversed_padding_repeated_twice[-3]),
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
        if type(module) is torch.nn.Conv2d:
            hooks.append(module.register_forward_pre_hook(process_layer))

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        model(input)

     # Remove hooks
    for hook in hooks:
        hook.remove()

    return model


def run_model(
    source,
    model_name,
    compile=False,
    batch_size=1,
    convert_weights_to_hwio=False,
    csv_output=None,
    method_name=None,
    warmup=5,
    runs=50,
):
    model, input_tensor = get_model_and_input(model_name, source, batch_size)

    # Set the model to evaluation mode
    model.eval()

    # Convert model and input to channel last memory format
    model = model.to(device="cpu", memory_format=torch.channels_last)
    input_tensor = input_tensor.to(
        device="cpu", memory_format=torch.channels_last
    )

    if convert_weights_to_hwio:
        # Heuristic changes slightly for the static version
        # model = convert_conv2d_weights_to_HWIO_static(model)
        model = convert_conv2d_weights_to_HWIO_dynamic(model, input_tensor)

    if compile:
        model = torch.compile(model)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start_time = time.perf_counter()
            model(input_tensor)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

    if csv_output:
        csv_output = Path(csv_output)
        if not method_name:
            print("Method name is required for csv output", file=sys.stderr)
            sys.exit(1)

        csv_exists = csv_output.exists()
        with open(csv_output, "a", newline="") as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not csv_exists:
                writer.writerow(["Model", "Method Name", "Time", "Unit"])
            for t in times:
                writer.writerow([model_name, method_name, t, "s"])
    else:
        # Calculate statistics
        median_time = np.median(times)
        iqr = np.percentile(times, 75) - np.percentile(times, 25)
        stats = {
            "Model": [model_name],
            "Median": [median_time * 1000],
            "IQR": [iqr * 1000],
            "Unit": ["ms"],
            "Runs": [runs],
        }
        print(tabulate(stats, headers="keys", tablefmt="psql", floatfmt=".2f"))


def get_all_models(source, filter_models=None):
    all_torch_models = models.list_models()
    exclude_models = models.list_models(module=models.video)
    exclude_models += models.list_models(module=models.quantization)
    exclude_models += models.list_models(module=models.optical_flow)
    torch_models = [model for model in all_torch_models if model not in exclude_models]
    if source == "torch":
        model_names = torch_models
    elif source == "timm":
        all_timm_models = timm.list_models(pretrained=False)
        model_names = [model for model in all_timm_models if model not in torch_models]
    else:
        model_names = []
        print("Unknown source", file=sys.stderr)

    if filter_models:
        model_names = [model for model in model_names if model in filter_models]

    return model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a PyTorch model.")
    parser.add_argument(
        "Source",
        type=str,
        choices=["timm", "torch"],
        default="torch",
        help="Source used to get models.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="The name of the model to convert. Default: squeezenet1_1 for torch, mobilenetv3_small_100.lamb_in1k for timm.",
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
        choices=["HWIO", "OHWI"],
        help="Weight layout for ZeroCopy2D. OHWI is the channel last layout. HWIO is the layout preferred by ZeroCopy2D."
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
    source = args.Source
    model_names = get_all_models(source)

    if args.list_models:
        for model in model_names:
            print("- ", model)
        exit(0)

    if args.model_name is None:
        if source == "torch":
            args.model_name = "squeezenet1_1"
        elif source == "timm":
            args.model_name = "mobilenetv3_small_100"

    if args.model_name not in model_names:
        print(f"Model {args.model_name} not found.", file=sys.stderr)
        exit(1)

    if args.zc_enable:
        os.environ["ZC_ENABLE"] = "TRUE"
    else:
        os.environ["ZC_ENABLE"] = "FALSE"

    if args.show_conv_time:
        os.environ["ZC_TIME"] = "TRUE"
    else:
        os.environ["ZC_TIME"] = "FALSE"

    convert_weights_to_hwio = False
    os.environ["ZC_WEIGHTS_LAYOUT"] = args.zc_weights_layout
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

    # Avoid variability in performance measurements due to random choices
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    run_model(source, args.model_name, args.compile, args.batch_size, convert_weights_to_hwio)
