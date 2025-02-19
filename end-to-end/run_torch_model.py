#!/usr/bin/env python
import argparse
import os
import sys
import timm

import torch
import torch.utils.benchmark as benchmark
import torchvision.models as models


def get_model_and_input(model_name, source):
    if source == "torch":
        # Get model and weights
        weights = models.get_model_weights(model_name).DEFAULT
        model = models.get_model(model_name, weights=weights)
        # Pre process input with the model's transforms
        dummy_input = torch.randn(3, 224, 224)
        preprocess = weights.transforms()
        input = preprocess(dummy_input)
        # Add batch dimension
        input = input.unsqueeze(0)
    elif source == "timm":
        # Get model
        model = timm.create_model(model_name)
        # Get input
        input_shape = (1, *model.default_cfg["input_size"])
        input = torch.randn(input_shape)
    else:
        print("Unknown source", file=sys.stderr)
        sys.exit(1)

    # Set the model to evaluation mode
    model.eval()
    # Convert model and input to channel last memory format
    model = model.to(device="cpu", memory_format=torch.channels_last)
    input = input.to(
        device="cpu", memory_format=torch.channels_last
    )

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
    batch=1,
    convert_weights_to_hwio=False,
    csv_output=None,
    csv_header=False,
    method_name=None,
):

    def run_inference(model, input):
        with torch.no_grad():  # Disable gradient calculation
            return model(input)

    model, input = get_model_and_input(model_name, source)

    if convert_weights_to_hwio:
        # Heuristic changes slightly for the static version
        # model = convert_conv2d_weights_to_HWIO_static(model)
        model = convert_conv2d_weights_to_HWIO_dynamic(model, input)

    if compile:
        model = torch.compile(model)

    num_threads = torch.get_num_threads()

    t0 = benchmark.Timer(
        stmt="run_inference(model, input)",
        num_threads=num_threads,
        globals={"model": model, "input": input, "run_inference": run_inference},
    )

    # Warm up runs: for big models, the adaptive_autorange may not run enough warm up runs
    t0.timeit(8)
    # Actual runs
    m0 = t0.blocked_autorange(min_run_time=5, min_number=10, min_times=5)

    if csv_output:
        if not method_name:
            print("Method name is required for csv output", file=sys.stderr)
            sys.exit(1)

        with open(csv_output, "a") as f:
            if csv_header:
                f.write("Method,Model,Mean,Median,IQR,Unit,Runs,Threads\n")
            f.write(
                f"{method_name},{model_name},{m0.mean*1000},{m0.median*1000},{m0.iqr*1000},ms,{len(m0.times)},{num_threads}\n"
            )
    else:
        print("Model: ", model_name)
        print(m0)


def get_all_models(source):
    if source == "timm":
        model_names = timm.list_models(pretrained=False)
    elif source == "torch":
        all_model_names = models.list_models()
        exclude_models = models.list_models(module=models.video)
        exclude_models += models.list_models(module=models.quantization)
        exclude_models += models.list_models(module=models.optical_flow)
        model_names = [model for model in all_model_names if model not in exclude_models]
    else:
        model_names = []
        print("Unknown source", file=sys.stderr)
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
            args.model_name = "mobilenetv3_small_100.lamb_in1k"

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

    run_model(source, args.model_name, args.compile, args.batch_size, convert_weights_to_hwio)
