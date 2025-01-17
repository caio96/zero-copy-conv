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
        (
            df["filter height"]
            <= df["image height"] + df["padding top"] + df["padding bottom"]
        )
        & (
            df["filter width"]
            <= df["image width"] + df["padding left"] + df["padding right"]
        )
    ]

    return df.reset_index(drop=True)


def include_only_in_df(df: pd.DataFrame, conv_type: list):
    if not conv_type:
        return df

    filtered_df = pd.DataFrame()

    if "strided" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[(df["stride height"] != 1) | (df["stride width"] != 1)]])

    if "pointwise" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[(df["filter height"] == 1) & (df["filter width"] == 1)]])

    if "grouped" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[df["groups"] != 1]])

    if "dilated" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[(df["dilation height"] != 1) | (df["dilation width"] != 1)]])

    if "transposed" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[df["is transposed"] == 0]])

    if "depthwise" in conv_type:
        filtered_df = pd.concat([filtered_df, df.loc[(df["image channel"] == df["groups"])]])

    # Drop duplicates to avoid duplicating rows if they match multiple types
    return filtered_df.drop_duplicates().reset_index(drop=True)


# Exclude from the data frame all convolution types listed in conv_types
def exclude_from_df(df: pd.DataFrame, conv_types: list):
    if not conv_types:
        return df

    if "strided" in conv_types:
        df = df.loc[(df["stride height"] == 1) & (df["stride width"] == 1)]

    if "pointwise" in conv_types:
        df = df.loc[(df["filter height"] != 1) | (df["filter width"] != 1)]

    if "grouped" in conv_types:
        df = df.loc[df["groups"] == 1]

    if "dilated" in conv_types:
        df = df.loc[(df["dilation height"] == 1) & (df["dilation width"] == 1)]

    if "transposed" in conv_types:
        df = df.loc[df["is transposed"] == 0]

    if "depthwise" in conv_types:
        df = df.loc[(df["image channel"] != df["groups"])]

    return df.reset_index(drop=True)


# Remove convolutions that only differ in "has bias" and padding values
# keeping the one with the most occurrences (first in the dataframe)
def reduce_redundacies(df):
    # All columns except bias and padding
    group_columns = [
        "batch size",
        "image channel",
        "image height",
        "image width",
        "output channel",
        "filter height",
        "filter width",
        "stride height",
        "stride width",
        "dilation height",
        "dilation width",
        "groups",
        "is transposed",
    ]

    # Group by the specified columns and keep the first occurrence
    df.sort_values(by=["occurrences"], ascending=False, inplace=True)
    df_reduced = (
        df.groupby(group_columns)
        .agg(
            conv_parameters=("conv_parameters", "first"),
            occurrences=("occurrences", "sum"),
        )
    )

    return df_reduced.reset_index(drop=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Filter the csv with convolution layers."
    )

    parser.add_argument("Input_CSV", type=str, help="Path to the input CSV file.")
    parser.add_argument("Output_CSV", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "--exclude-conv-types",
        nargs="+",
        type=str,
        help="List of convolution types to exclude",
        choices=[
            "strided",
            "pointwise",
            "depthwise",
            "grouped",
            "dilated",
            "transposed",
        ],
    )
    parser.add_argument(
        "--include-only-conv-types",
        nargs="+",
        type=str,
        help="Only include the specified convolution types",
        choices=[
            "strided",
            "pointwise",
            "depthwise",
            "grouped",
            "dilated",
            "transposed",
        ],
        default=None,
    )
    parser.add_argument(
        "--reduce-redundancies",
        action="store_true",
        help="Keep only one convolution if there are multiple that only differ in padding and bias",
    )

    args = parser.parse_args()

    input_csv = Path(args.Input_CSV)
    output_csv = Path(args.Output_CSV)
    reduce_redundancies = args.reduce_redundancies
    exclude_conv_types = args.exclude_conv_types
    include_only_conv_types = args.include_only_conv_types

    # Check if input file exists
    if (not input_csv.exists()) or (not input_csv.is_file()):
        print("Input not found.", file=sys.stderr)
        sys.exit(-1)

    # Load input
    df = pd.read_csv(input_csv, header=0)
    num_columns = len(df.columns)

    # Remove name of models column
    if "models" in df.columns:
        df = df.drop(columns=["models"])
        num_columns -= 1

    # Split conv_parameters column
    df = split_parameters(df)

    # Remove convolutions that cause problems with LibTorch
    df = remove_problem_parameters(df)

    if include_only_conv_types:
        df = include_only_in_df(df, include_only_conv_types)

    if exclude_conv_types:
        df = exclude_from_df(df, exclude_conv_types)

    # Remove redundancies
    if reduce_redundancies:
        df = reduce_redundacies(df)

    # Remove extra columns from split
    df = df.iloc[:, :num_columns]

    # Make sure there are no duplicates
    df = df.groupby("conv_parameters").agg(
            conv_parameters=("conv_parameters", "first"),
            occurrences=("occurrences", "sum"),
        ).sort_values(by=["occurrences"], ascending=False)

    # Save df to csv removing extra split columns
    df.to_csv(output_csv, index=False)
