#!/usr/bin/env python
import time

import onnx
import onnxruntime
import torch


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_model(model_name):
    # Check the ONNX model
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":

    model_name = "resnet50.onnx"
    check_model(model_name)

    # Set the logging level to VERBOSE
    onnxruntime.set_default_logger_severity(0)

    ort_session = onnxruntime.InferenceSession(
        "resnet50.onnx", providers=["CPUExecutionProvider"]
    )

    image = torch.randn(1, 3, 224, 224)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}

    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end = time.time()
    elapsed_time_ms = (end - start) * 1000
    print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")
