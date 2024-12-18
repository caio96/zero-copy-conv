#!/usr/bin/env python
import argparse
import time

import torch
import torchvision.models as models

def run_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=True)
    elif model_name == "alexnet":
        model = models.alexnet(weights=True)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(weights=True)
    elif model_name == "vgg16":
        model = models.vgg16(weights=True)
    elif model_name == "densenet":
        model = models.densenet161(weights=True)
    elif model_name == "inception":
        model = models.inception_v3(weights=True)
    elif model_name == "googlenet":
        model = models.googlenet(weights=True)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=True)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=True)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=True)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=True)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=True)
    elif model_name == "mnasnet":
        model = models.mnasnet1_0(weights=True)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    # Create a dummy input tensor with the shape that the model expects (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)

    model.eval()  # Set the model to evaluation mode
    model = model.to(memory_format=torch.channels_last)  # Replace with your model
    dummy_input = dummy_input.to(memory_format=torch.channels_last)  # Replace with your input tensor
    # model = torch.compile(model)

    with torch.no_grad():  # Disable gradient calculation
        start = time.time()
        output = model(dummy_input)
        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")
        start = time.time()
        output = model(dummy_input)
        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")
        start = time.time()
        output = model(dummy_input)
        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")
        start = time.time()
        output = model(dummy_input)
        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")
        start = time.time()
        output = model(dummy_input)
        end = time.time()
        elapsed_time_ms = (end - start) * 1000
        print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a PyTorch model."
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
    run_model(args.model_name)
