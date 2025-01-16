#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import timm
import torchvision.models as models
import torch
from tqdm import tqdm


class VerboseExecution(torch.nn.Module):
    def __init__(self, model, model_name, conv_parameters, source, weights=None):
        super().__init__()
        self.model = model
        self.conv_parameters = conv_parameters
        if source == "timm":
            data_config = timm.data.resolve_model_data_config(model)
            channels, height, width = data_config["input_size"]
        elif source == "torch":
            # Pre process input with the model's transforms
            dummy_input = torch.randn(3, 224, 224)
            if weights:
                preprocess = weights.transforms()
                dummy_input = preprocess(dummy_input)
            channels, height, width = dummy_input.shape
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
                1,  # Default batch size
                *inputs[0].shape[1:],
                outputs[0].shape[0],  # Output channels
                *module.kernel_size,
                *module._reversed_padding_repeated_twice,
                *module.stride,
                *module.dilation,
                module.groups,
                int(module.transposed),
                1 if module.bias is not None else 0,
            )
            parameters = " ".join(map(str, parameters))

            # Add layer
            self.conv_parameters[parameters].append(model_name)

        return fn

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gather convolutional layer parameters into a csv file."
    )
    parser.add_argument(
        "Output_Dir", type=str, help="Path to the directory where to save csv output."
    )
    parser.add_argument(
        "Source", type=str, choices=["timm", "torch"], default="timm", help="Source used to get models to extract convolutions."
    )

    args = parser.parse_args()
    output_dir = Path(args.Output_Dir).absolute()
    source = args.Source

    # Store results
    conv_parameters = defaultdict(list)

    # For each model in timm
    if source == "timm":
        model_names = timm.list_models()
    elif source == "torch":
        all_model_names = models.list_models()
        # Exclude video models
        video_model_names = models.list_models(module=models.video)
        model_names = [
            model for model in all_model_names if model not in video_model_names
        ]

    for model_name in tqdm(models):
        weights = None
        # Create it in evaluation mode
        if source == "timm":
            model = timm.create_model(model_name).eval()
        elif source == "torch":
            weights = models.get_model_weights(model_name).DEFAULT
            model = models.get_model(model_name, weights=weights)
        # Save the parameters for Conv2D layers
        model = VerboseExecution(model, model_name, conv_parameters, source, weights)
        # Delete it
        del model

    # Value is a list of model names that contain the layer (key), set is used to remove duplicates
    # Make list into a string and count the number of times that the layer is used
    conv_parameters = {
        key: (len(value), " ".join(set(value)))
        for key, value in conv_parameters.items()
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
    df.to_csv(output_dir / "conv_layers.csv", index=False)
