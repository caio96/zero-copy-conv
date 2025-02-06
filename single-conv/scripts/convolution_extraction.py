#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sys
import timm
import torch
import torchvision.models as models
from tqdm import tqdm


def save_conv_params(model, model_name, input, conv_parameters):

    def hook(module: torch.nn.Conv2d, inputs, output):
        # Use all parameters as a key
        parameters = (
            1,  # Default batch size
            *(module.in_channels, inputs[0].shape[2], inputs[0].shape[3]),
            module.out_channels,
            *module.kernel_size,
            *module._reversed_padding_repeated_twice,
            *module.stride,
            *module.dilation,
            module.groups,
            int(module.transposed),
            1 if module.bias is not None else 0,
        )
        # Get parameters as a string
        parameters = " ".join(map(str, parameters))
        # Add layer
        conv_parameters[parameters].append(model_name)

    # Register hooks to process each Conv2d layer
    for name, module in model.named_modules():
        if type(module) is torch.nn.Conv2d:
            module.register_forward_hook(hook)

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        model(input)


def get_all_models(source):
    model_names = []
    if source == "timm":
        model_names = timm.list_models(pretrained=False)
    elif source == "torch":
        all_model_names = models.list_models()
        exclude_models = models.list_models(module=models.video)
        exclude_models += models.list_models(module=models.quantization)
        exclude_models += models.list_models(module=models.optical_flow)
        model_names = [model for model in all_model_names if model not in exclude_models]
    return model_names


def get_model_and_input(model_name, source):
    input = None
    model = None
    if source == "torch":
        # Get model and weights
        weights = models.get_model_weights(model_name).DEFAULT
        model = models.get_model(model_name, weights=weights).eval()
        # Pre process input with the model's transforms
        dummy_input = torch.randn(3, 224, 224)
        preprocess = weights.transforms()
        input = preprocess(dummy_input)
        # Add batch dimension
        input = input.unsqueeze(0)
    elif source == "timm":
        # Get model
        model = timm.create_model(model_name).eval()
        # Get input
        input_shape = (1, *model.default_cfg["input_size"])
        input = torch.randn(input_shape)
    return model, input


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gather convolutional layer parameters into a csv file."
    )
    parser.add_argument(
        "Output_Dir", type=str, help="Path to the directory where to save csv output."
    )
    parser.add_argument(
        "Source",
        type=str,
        choices=["timm", "torch", "all"],
        default="torch",
        help="Source used to get models to extract convolutions.",
    )

    args = parser.parse_args()
    output_dir = Path(args.Output_Dir).absolute()
    source_name = args.Source
    sources = []
    if source_name == "all":
        sources = ["timm", "torch"]
    else:
        sources = [source_name]

    # Store results
    conv_parameters = defaultdict(list)

    total_iterations = 0
    for source in sources:
        total_iterations += len(get_all_models(source))

    with tqdm(total=total_iterations) as pbar:
        for source in sources:
            # Get model names from source
            model_names = get_all_models(source)
            # Get model and input and save conv parameters
            for model_name in model_names:
                model, input = get_model_and_input(model_name, source)
                save_conv_params(model, model_name, input, conv_parameters)
                pbar.update(1)
                del model

    # Value is a list of model names that contain the layer (key), set is used to remove duplicates
    # Make list into a string and count the number of times that the layer is used
    conv_parameters = {
        key: (len(value), " ".join(set(value))) for key, value in conv_parameters.items()
    }

    # Save results to pandas dataframe
    df = pd.DataFrame.from_dict(conv_parameters, orient="index")
    df = df.reset_index().rename(
        columns={
            "index": "conv_parameters",
            0: "occurrences",
            1: "models",
        }
    )
    df = df.sort_values("occurrences", ascending=False)

    # Save to csv
    csv_name = f"conv_layers_{source_name}.csv"
    df.to_csv(output_dir / csv_name, index=False)
