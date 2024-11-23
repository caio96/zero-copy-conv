#!/usr/bin/env python
import torch
import torchvision.models as models

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the shape that the model expects (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 224, 224)

# Define the file path for the output ONNX file
onnx_file_path = "resnet50.onnx"

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
