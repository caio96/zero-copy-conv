#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def split_parameters(df):
    conv_parameters_split = [
        "batch size",
        "image channel",
        "image height",
        "image width",
        "output channel",
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


# Check if kernel size is bigger than padded input
# There are some examples like this that cause error with Libtorch
def remove_problem_parameters(df):
    df = df.loc[
        (df["filter height"] <= df["image height"] + df["padding top"] + df["padding bottom"])
        & (df["filter width"] <= df["image width"] + df["padding left"] + df["padding right"])
    ]

    return df.reset_index(drop=True)


def include_only_in_df(df: pd.DataFrame, conv_types: list):
    if not conv_types:
        return df

    filtered_df = pd.DataFrame()

    if "unit-stride" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["stride height"] == 1) & (df["stride width"] == 1)]]
        )

    if "strided" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["stride height"] != 1) | (df["stride width"] != 1)]]
        )

    if "not-padded" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["padding top"] == 0) & (df["padding bottom"] == 0) & (df["padding left"] == 0) & (df["padding right"] == 0)]]
        )

    if "padded" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["padding top"] != 0) | (df["padding bottom"] != 0) | (df["padding left"] != 0) | (df["padding right"] != 0)]]
        )

    if "pointwise" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["filter height"] == 1) & (df["filter width"] == 1)]]
        )

    if "small-kernel" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["filter height"] > 1) & (df["filter width"] > 1) & (df["filter height"] <= 3) & (df["filter width"] <= 3)]]
        )

    if "large-kernel" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["filter height"] > 3) & (df["filter width"] > 3)]]
        )

    if "asymmetric-kernel" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["filter height"] != df["filter width"])]]
        )

    if "pixel-input" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[(df["image height"] == 1) & (df["image width"] == 1)]])

    if "global" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[(df["image height"] == df["filter height"]) & (df["image width"] == df["filter width"])]])

    if "direct-gemm" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["filter height"] == 1) & (df["filter width"] == 1) & (df["stride height"] == 1) & (df["stride width"] == 1) & (df["padding top"] == 0) & (df["padding bottom"] == 0) & (df["padding left"] == 0) & (df["padding right"] == 0)]]
        )

    if "overlapped" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[(df["filter height"] > df["stride height"]) | (df["filter width"] > df["stride width"])]])

    if "not-overlapped" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[(df["filter height"] == df["stride height"]) & (df["filter width"] == df["stride width"])]])

    if "grouped" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["groups"] > 1)]]
        )

    if "depthwise" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[(df["groups"] > 1) & (df["image channel"] == df["groups"])]])

    if "dilated" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["dilation height"] != 1) | (df["dilation width"] != 1)]]
        )

    if "not-dilated" in conv_types:
        filtered_df = pd.concat(
            [filtered_df, df.loc[(df["dilation height"] == 1) & (df["dilation width"] == 1)]]
        )

    if "transposed" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[df["is transposed"] == 1]])

    if "im2col-singlethread-heuristic" in conv_types:
        filtered_df = pd.concat([filtered_df, df.loc[
                                 (df["image channel"] != df["groups"]) &                     # not depthwise
                                 ((df["stride height"] == 1) & (df["stride width"] == 1)) &  # unit-stride
                                 ((df["filter height"] > df["stride height"]) | (df["filter width"] > df["stride width"])) &  # overlapped (and not pointwise)
                                 ((df["dilation height"] == 1) & (df["dilation width"] == 1))  # not dilated
                                 ]])

    if "torch-singlethread-heuristic" in conv_types:
        # filtered_df = pd.concat([filtered_df, df.loc[(df["image height"] == 1) & (df["image width"] == 1)]])
        df["output height"] = np.floor(
            (
                df["image height"]
                + df["padding top"] + df["padding bottom"]
                - df["dilation height"] * (df["filter height"] - 1)
                - 1
            )
            / df["stride height"]
            + 1
        )
        # todo: test heuristic
        filtered_df = pd.concat([filtered_df, df.loc[
                                 (df["groups"] == 1)                     # not grouped
                                 & ((df["filter height"] != 1) | (df["filter width"] != 1)) # not pointwise
                                 & ((df["stride height"] == 1) & (df["stride width"] == 1))  # unit-stride
                                 & (df["output height"] < df["image channel"])
                                 ]])

    if "torch-multithread-heuristic" in conv_types:
        df["output height"] = np.floor(
            (
                df["image height"]
                + df["padding top"] + df["padding bottom"]
                - df["dilation height"] * (df["filter height"] - 1)
                - 1
            )
            / df["stride height"]
            + 1
        )
        df["dim k"] = df["filter width"] * df["image channel"] / df["groups"]
        # filtered_df = pd.concat([filtered_df, df.loc[(df["groups"] == 1) & (df["output height"] < df["dim k"]) & (df["output height"] != 1) & (df["output channel"] < df["dim k"])]])
        # todo: test heuristic
        filtered_df = pd.concat([filtered_df, df.loc[
                                 (df["groups"] == 1)                     # not grouped
                                 & ((df["filter height"] != 1) | (df["filter width"] != 1)) # not pointwise
                                 & (df["output height"] < df["image channel"])
                                 ]])

    if "onednn-heuristic" in conv_types:
        df["output height"] = np.floor(
            (
                df["image height"]
                + df["padding top"] + df["padding bottom"]
                - df["dilation height"] * (df["filter height"] - 1)
                - 1
            )
            / df["stride height"]
            + 1
        )
        df["dim k"] = df["filter width"] * df["image channel"] / df["groups"]
        # filtered_df = pd.concat([filtered_df, df.loc[
        #                          (df["groups"] == 1) &
        #                          (((df["image height"] == 1) & (df["image width"] == 1)) |
        #                          (((df["filter height"] != 1) & (df["filter width"] != 1)) & (df["output height"] < df["dim k"]) & (df["output channel"] < df["dim k"])))
        #                          ]])
        # todo: test heuristic
        filtered_df = pd.concat([filtered_df, df.loc[
                                 (df["groups"] == 1) &                     # not grouped
                                 ((df["stride height"] == 1) & (df["stride width"] == 1)) &  # unit-stride
                                 ((df["filter height"] != 1) | (df["filter width"] != 1)) # not pointwise
                                 # (df["output channel"] < df["image channel"])
                                 # (df["output channel"] < df["image channel"] / df["groups"])
                                 # (df["filter height"] != df["filter width"])
                                 ]])

    # Drop duplicates to avoid duplicating rows if they match multiple types
    return filtered_df.drop_duplicates().reset_index(drop=True)


# Exclude from the data frame all convolution types listed in conv_types
def exclude_from_df(df: pd.DataFrame, conv_types: list):
    if not conv_types:
        return df

    included_df = include_only_in_df(df, conv_types)
    excluded_df = df.loc[~df['conv_parameters'].isin(included_df['conv_parameters'])]
    return excluded_df.reset_index(drop=True)


def get_categories():
    return [
            "unit-stride",
            "strided",
            "not-padded",
            "padded",
            "pointwise",
            "small-kernel",
            "large-kernel",
            "asymmetric-kernel",
            "pixel-input",
            "global",
            "direct-gemm",
            "overlapped",
            "not-overlapped",
            "grouped",
            "depthwise",
            "dilated",
            "not-dilated",
            "transposed",
            "im2col-singlethread-heuristic",
            "torch-multithread-heuristic",
            "torch-singlethread-heuristic",
            "onednn-heuristic",
        ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter the csv with convolution layers.")

    parser.add_argument("Input_CSV", type=str, help="Path to the input CSV file.")
    parser.add_argument("Output_CSV", type=str, help="Path to the output CSV file.")
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
    output_csv = Path(args.Output_CSV)
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

    # Remove name of models column
    if "models" in df.columns:
        df = df.drop(columns=["models"])
        num_columns -= 1

    # Make sure there are no duplicates
    df = (
        df.groupby("conv_parameters")
        .agg(
            conv_parameters=("conv_parameters", "first"),
            occurrences=("occurrences", "sum"),
        )
        .sort_values(by=["occurrences"], ascending=False)
    )

    # Save df to csv removing extra split columns
    df.to_csv(output_csv, index=False)
