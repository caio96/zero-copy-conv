#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd


def summarize_results(df: pd.DataFrame, occurrences_df: pd.DataFrame, output_dir):

    # Split the 'name' column into 'conv_type' and 'conv_parameters'
    df[["conv_type", "conv_parameters"]] = df["name"].str.split(" ", n=1, expand=True)
    df = df.drop(columns=["name"])
    df["conv_parameters"] = df["conv_parameters"].str.split("/", n=1).str[0]

    # Separate df by 'conv_type'
    groups = df.groupby(by=["conv_type"])
    df_dict = {}
    for name, group in groups:
        name = name[0]
        # Aggregate results of repeated runs (that have the same 'conv_parameters' value)
        df_dict[name] = (
            group.groupby(by="conv_parameters", as_index=False)
            .agg(
                total_iterations=("iterations", "sum"),
                mean_time=("real_time", "mean"),
                std_time=("real_time", "std"),
                time_unit=("time_unit", "first"),
                error_occurred=("error_occurred", "any"),
            )
            .sort_values(by=["conv_parameters"])
            .reset_index(drop=True)
        )

    # Join results by 'conv_parameters'
    method_names = list(df_dict.keys())

    if len(method_names) == 1:
        print("Only one method found. No comparison possible.", file=sys.stderr)
        sys.exit(-1)

    joined_results = pd.merge(
        df_dict[method_names[0]],
        df_dict[method_names[1]].drop(columns=["time_unit"]),
        how="inner",
        on="conv_parameters",
        suffixes=("_" + method_names[0], "_" + method_names[1]),
    )
    for method_name in method_names[2:]:
        joined_results = joined_results.merge(
            df_dict[method_name].drop(columns=["time_unit"]).add_suffix("_" + method_name),
            how="inner",
            left_on="conv_parameters",
            right_on="conv_parameters_" + method_name,
            suffixes=(None, None),
        ).drop(columns=["conv_parameters_" + method_name])

    # Add occurrences column
    joined_results = joined_results.merge(occurrences_df, how="left", on="conv_parameters")

    # Save joined results
    joined_results.to_csv(output_dir / f"performance-results.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with performance logs and summarize them into another csv."
    )

    parser.add_argument(
        "CSV_Input", type=str, help="Path to the input CSV file with performance logs."
    )
    parser.add_argument(
        "Occurrences_CSV", type=str, help="Path to the CSV file with conv_parameters and occurrences."
    )
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")

    args = parser.parse_args()

    csv_input = Path(args.CSV_Input)
    occurrences_csv = Path(args.Occurrences_CSV)
    output_dir = Path(args.Output_Dir)

    # Check if csv file exists
    if (not csv_input.exists()) or (not csv_input.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    if (not occurrences_csv.exists()) or (not occurrences_csv.is_file()):
        print("CSV with occurrences not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)

    df = pd.read_csv(csv_input, header=0, index_col=False)
    occurrences_df = pd.read_csv(occurrences_csv, header=0, index_col=False)

    summarize_results(df, occurrences_df, output_dir)
