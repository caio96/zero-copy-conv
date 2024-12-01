#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime


def check_model(model_path):
    # Check the ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)


def run_model(model_path, batch_size=1, num_threads=1, verbose=False):
    if verbose:
        onnxruntime.set_default_logger_severity(0)

    sess_options = onnxruntime.SessionOptions()
    sess_options.add_session_config_entry("session.NchwcTransformer", "0")
    sess_options.inter_op_num_threads = num_threads
    sess_options.intra_op_num_threads = num_threads
    optimizer_filter = ["NchwcTransformer"]

    ort_session = onnxruntime.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
        disabled_optimizers=optimizer_filter,
    )

    image = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: image}

    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end = time.time()
    elapsed_time_ms = (end - start) * 1000
    print(f"Inference of ONNX model used {elapsed_time_ms:.4f} ms")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a ONNX model with OnnxRuntime.")
    parser.add_argument(
        "model_path",
        type=str,
        help="The to the model to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of batches to run.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pring OnnxRuntime logs.",
    )

    args = parser.parse_args()
    model_path = Path(args.model_path).absolute()
    batch_size = args.batch_size
    num_threads = args.num_threads
    verbose = args.verbose

    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model file {model_path} not found or not a file.")

    check_model(model_path)
    run_model(
        model_path, batch_size=batch_size, num_threads=num_threads, verbose=verbose
    )
