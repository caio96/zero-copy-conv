#!/usr/bin/env python
import argparse
import time
import os

import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark


# Based on torch/nn/utils/memory_format.py
# This function is not supported, it is just an example of how to convert the weights of convolution layers to HWIO.
# To support it, Zero Copy Conv would need to be the only convolution implementation to run and it would need to expect HWIO weights (in the contiguous memory format).
# This function assumes that the model is already in the channels last memory format.
def convert_conv2d_weight_OHWI_to_HWIO(module):
    if isinstance(module, torch.nn.Conv2d):
        weight_data = (module.weight.detach().clone())
        # from channel last to channel last, not changing data
        weight_data = weight_data.permute(0, 2, 3, 1)
        # from OHWI to HWIO, making data contiguous
        weight_data = weight_data.permute(1, 2, 3, 0).contiguous()
        # permute dimensions expected order by contiguous format
        weight_data = weight_data.permute(3, 2, 0, 1)
        module.weight.data = weight_data.resize_(weight_data.size())
    for child in module.children():
        convert_conv2d_weight_OHWI_to_HWIO(child)
    return module


def run_inference(model, input):
    with torch.no_grad():  # Disable gradient calculation
        return model(input)


def run_model(model_name, compile=False, batch=1):
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
    model = model.to(device="cpu", memory_format=torch.channels_last)  # Replace with your model
    input = input.to(device="cpu", memory_format=torch.channels_last)  # Replace with your input tensor
    if (compile):
        model = torch.compile(model)

    num_threads = torch.get_num_threads()

    t0 = benchmark.Timer(
        stmt='run_inference(model, input)',
        num_threads=num_threads,
        setup='from __main__ import run_inference',
        globals={'model': model, 'input': input})
    m0 = t0.blocked_autorange(min_run_time=0.5)
    print(m0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a PyTorch model."
    )
    all_models = models.list_models(module=models)
    parser.add_argument(
        "model_name",
        type=str,
        choices=all_models,
        help="The name of the model to convert.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to compile the model.",
    )

    parser.add_argument(
        "--enable-zero-copy-conv",
        action="store_true",
        help="Enable Zero Copy Conv in Pytorch.",
    )

    parser.add_argument(
        "--show-conv-time",
        action="store_true",
        help="Print the execution time of each convolution in the model.",
    )

    parser.add_argument(
        "--ignore-weight-transform",
        action="store_true",
        help="Make Zero Copy Conv ignore the weight transformation required for correct results (OHWI -> HWIO).",
    )

    parser.add_argument(
        "--ignore-output-transform",
        action="store_true",
        help="Make Zero Copy Conv ignore the output transformation required for correct results (NWHC -> NHWC).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Size of batch.",
    )

    args = parser.parse_args()

    if args.enable_zero_copy_conv:
        os.environ["ZERO_COPY_2D"] = "TRUE"
    else:
        os.environ["ZERO_COPY_2D"] = "FALSE"

    if args.show_conv_time:
        os.environ["SHOW_CONV_TIME"] = "TRUE"
    else:
        os.environ["SHOW_CONV_TIME"] = "FALSE"

    if args.ignore_weight_transform:
        os.environ["ZERO_COPY_TRANSFORM_WEIGHTS"] = "FALSE"
    else:
        os.environ["ZERO_COPY_TRANSFORM_WEIGHTS"] = "TRUE"

    if args.ignore_output_transform:
        os.environ["ZERO_COPY_TRANSFORM_OUTPUT"] = "FALSE"
    else:
        os.environ["ZERO_COPY_TRANSFORM_OUTPUT"] = "TRUE"

    run_model(args.model_name, args.compile, args.batch_size)
