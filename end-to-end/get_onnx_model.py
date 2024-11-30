#!/usr/bin/env python
import argparse

import torch
import torchvision.models as models


def save_model(model_name):
    # Load the pre-trained ResNet50 model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "densenet":
        model = models.densenet161(pretrained=True)
    elif model_name == "inception":
        model = models.inception_v3(pretrained=True)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
    elif model_name == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor with the shape that the model expects (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Define the file path for the output ONNX file
    onnx_file_path = f"{model_name}.onnx"

    # Export the model to an ONNX file
    torch.onnx.export(
        model,  # Model to be exported
        dummy_input,  # Dummy input tensor
        onnx_file_path,  # Output file path
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=20,  # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=["input"],  # Input tensor name
        output_names=["output"],  # Output tensor name
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Dynamic axes for variable batch size
    )

    print(f"Model has been converted to ONNX and saved at {onnx_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to ONNX format."
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=[
            "resnet18", "alexnet", "squeezenet", "vgg16", "densenet", "inception",
            "googlenet", "shufflenet", "mobilenet_v2", "mobilenet_v3_large",
            "mobilenet_v3_small", "resnext50_32x4d", "wide_resnet50_2", "mnasnet"
        ],
        help="The name of the model to convert.",
    )
    args = parser.parse_args()
    save_model(args.model_name)
