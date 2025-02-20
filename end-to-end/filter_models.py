#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import from single-conv scripts
script_dir = Path(__file__).resolve().absolute().parent
sys.path.append(Path(script_dir / "../single-conv/scripts/").resolve().absolute().as_posix())
from filter_csv import exclude_from_df, include_only_in_df, split_parameters, get_categories, remove_problem_parameters


def print_models(df: pd.DataFrame):
    models = set()
    for model in df["models"]:
        models.update(model.split(" "))

    for model in models:
        print(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Print models that contain the specified convolution types")

    parser.add_argument("Input_CSV", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "--exclude-conv-types",
        nargs="+",
        type=str,
        help="List of convolution types to exclude",
        choices=get_categories(),
    )
    parser.add_argument(
        "--include-only-conv-types",
        nargs="+",
        type=str,
        help="Only include the specified convolution types",
        choices=get_categories(),
    )

    args = parser.parse_args()

    input_csv = Path(args.Input_CSV)
    exclude_conv_types = args.exclude_conv_types
    include_only_conv_types = args.include_only_conv_types

    # Check if input file exists
    if (not input_csv.exists()) or (not input_csv.is_file()):
        print("Input not found.", file=sys.stderr)
        sys.exit(-1)

    # Load input
    df = pd.read_csv(input_csv, header=0)
    num_columns = len(df.columns)

    # Split conv_parameters column
    df = split_parameters(df)

    # Remove convolutions that cause problems with LibTorch
    df = remove_problem_parameters(df)

    if include_only_conv_types:
        df = include_only_in_df(df, include_only_conv_types)

    if exclude_conv_types:
        df = exclude_from_df(df, exclude_conv_types)

    # Remove extra columns from split
    df = df.iloc[:, :num_columns]

    print_models(df)
