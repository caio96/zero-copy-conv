#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd


def split_parameters(df):
    conv_parameters_split = [
        "batch size",
        "image channel",
        "image height",
        "image width",
        "output channel",
        "output height",
        "output width",
        "filter height",
        "filter width",
        "padding top",
        "padding bottom",
        "padding left",
        "padding right",
        "stride height",
        "stride width",
        "dilation height",
        "dilation width",
        "groups",
        "is transposed",
        "has bias",
    ]

    df[conv_parameters_split] = df["conv_parameters"].str.split(" ", expand=True)
    df[conv_parameters_split] = df[conv_parameters_split].astype(int)

    return df


def filter_df(df, conv_type):

    if conv_type == "strided":
        df = df.loc[(df["stride height"] != 1) & (df["stride width"] != 1)]
    elif conv_type == "pointwise":
        df = df.loc[(df["filter height"] == 1) & (df["filter width"] == 1)]
    elif conv_type == "grouped":
        df = df.loc[df["groups"] != 1]
    elif conv_type == "dilated":
        df = df.loc[(df["dilation height"] != 1) & (df["dilation width"] != 1)]
    elif conv_type == "transposed":
        df = df.loc[df["is transposed"] == 1]
    elif conv_type == "standard":
        df = df.loc[
            (df["stride height"] == 1)
            & (df["stride width"] == 1)
            & (df["filter height"] != 1)
            & (df["filter width"] != 1)
            & (df["groups"] == 1)
            & (df["dilation height"] == 1)
            & (df["dilation width"] == 1)
            & (df["is transposed"] == 0)
        ]

    return df.reset_index()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter the csv with convolution layers.")

    parser.add_argument("Input_CSV", type=str, help="Path to the input CSV file.")
    parser.add_argument("Output_CSV", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "Conv_Type",
        type=str,
        default="standard",
        help="Type of convolution to select. Standard means convolutions that have stride 1, are not pointwise, not grouped, not dilated, and not transposed.",
        choices=["standard", "strided", "pointwise", "grouped", "dilated", "transposed"],
    )

    args = parser.parse_args()

    input_csv = Path(args.Input_CSV)
    output_csv = Path(args.Output_CSV)
    conv_type = args.Conv_Type

    # Check if input file exists
    if (not input_csv.exists()) or (not input_csv.is_file()):
        print("Input not found.", file=sys.stderr)
        sys.exit(-1)

    # Load input
    df = pd.read_csv(input_csv, header=0)

    # Split conv_parameters column
    df = split_parameters(df)

    # Filter df based on arguments
    df = filter_df(df, conv_type)

    # Save df to csv
    df.loc[:, "conv_parameters":"models"].to_csv(output_csv, index=False)
