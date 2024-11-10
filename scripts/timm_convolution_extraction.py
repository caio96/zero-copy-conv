#!/usr/bin/env python3

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import timm
import torch
from tqdm import tqdm


class VerboseExecution(torch.nn.Module):
    def __init__(self, model, model_name, conv_parameters):
        super().__init__()
        self.model = model
        self.conv_parameters = conv_parameters
        data_config = timm.data.resolve_model_data_config(model)
        channels, height, width = data_config["input_size"]
        self.input_size = (1, channels, height, width)

        # Register a hook for each Conv2D layer
        for _, layer in model.named_modules(remove_duplicate=False):
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self.save_shapes_hook(model_name))

        with torch.no_grad():
            self.forward(torch.rand(self.input_size))

    def save_shapes_hook(self, model_name):
        def fn(module: torch.nn.Conv2d, inputs, outputs):

            # Use all parameters as a key
            parameters = (
                1, # Default batch size
                *inputs[0].shape[1:],
                *outputs[0].shape,
                *module.kernel_size,
                *module._reversed_padding_repeated_twice,
                *module.stride,
                *module.dilation,
                module.groups,
                int(module.transposed),
                1 if module.bias is not None else 0,
            )

            # Add layer
            self.conv_parameters[parameters].append(model_name)

        return fn

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gather convolutional layer parameters from Timm into a pickle file."
    )
    parser.add_argument(
        "Output_Dir", type=str, help="Path to the directory where to save csv output."
    )

    args = parser.parse_args()
    output_dir = Path(args.Output_Dir).absolute()

    # Store results
    conv_parameters = defaultdict(list)

    # For each model in timm
    for model_name in tqdm(timm.list_models()):
        # Create it in evaluation mode
        model = timm.create_model(model_name).eval()
        # Save the parameters for Conv2D layers
        model = VerboseExecution(model, model_name, conv_parameters)
        # Delete it
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
            0: "occurences",
            1: "models",
        }
    )
    df["conv_parameters"] = (
        df["conv_parameters"].str.replace("(", "").str.replace(")", "").str.replace(",", "")
    )
    df.to_csv(output_dir / "conv_parameters.csv", index=False)
