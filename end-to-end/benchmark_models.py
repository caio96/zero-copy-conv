#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

from run_torch_model import get_all_models, run_model
from tqdm import tqdm


def run_model_zc(model_name, batch_size, compile, output_csv, csv_header):
    os.environ["ZC_ENABLE"] = "TRUE"
    os.environ["ZC_TIME"] = "FALSE"
    os.environ["ZC_TRANSFORM_OUTPUT"] = "TRUE"
    os.environ["ZC_HEURISTIC"] = "FALSE"
    try:
        run_model(model_name, compile, batch_size, True, output_csv, csv_header, "ZeroCopy2d")
    except RuntimeError as e:
        with open(f"{output_csv}.err", "a") as f:
            f.write(f"Error running {model_name}, with ZeroCopy2d, {e}")


def run_model_zc_heuristic(source, model_name, batch_size, compile, output_csv, csv_header):
    os.environ["ZC_ENABLE"] = "TRUE"
    os.environ["ZC_TIME"] = "FALSE"
    os.environ["ZC_TRANSFORM_OUTPUT"] = "TRUE"
    os.environ["ZC_HEURISTIC"] = "TRUE"
    try:
        run_model(source, model_name, compile, batch_size, True, output_csv, csv_header, "ZeroCopy2d_Heuristic")
    except RuntimeError as e:
        with open(f"{output_csv}.err", "a") as f:
            f.write(f"Error running {model_name}, with ZeroCopy2d_Heuristic, {e}")


def run_model_torch(source, model_name, batch_size, compile, output_csv, csv_header):
    os.environ["ZC_ENABLE"] = "FALSE"
    os.environ["ZC_TIME"] = "FALSE"
    try:
        run_model(source, model_name, compile, batch_size, False, output_csv, csv_header, "Torch")
    except RuntimeError as e:
        with open(f"{output_csv}.err", "a") as f:
            f.write(f"Error running {model_name}, with Torch, {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all PyTorch models available in the chosen source."
    )

    parser.add_argument(
        "Source",
        type=str,
        choices=["timm", "torch"],
        default="torch",
        help="Source used to get models.",
    )

    parser.add_argument(
        "Output_CSV",
        help="Path to output CSV file",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Size of batch.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repetitions for each run.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to compile the model.",
    )

    args = parser.parse_args()
    source = args.Source
    output_csv = Path(args.Output_CSV)
    repeats = args.repeats

    if output_csv.exists():
        print(f"Output CSV {output_csv} already exists.", file=sys.stderr)
        sys.exit(1)

    models = get_all_models(source)
    if args.list_models:
        for model in models:
            print("- ", model)
        exit(0)

    methods = [run_model_zc, run_model_zc_heuristic, run_model_torch]

    first = True
    total_iterations = repeats * len(models) * len(methods)

    with tqdm(total=total_iterations) as pbar:
        for repeat in range(repeats):
            for model_name in models:
                for method in methods:
                    method(source, model_name, args.batch_size, args.compile, output_csv, first)
                    if first:
                        first = False
                    pbar.update(1)
