#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd


def summarize_results(df: pd.DataFrame, output_dir):

    # Separate df by Method
    groups = df.groupby(by=["Method"])
    df_dict = {}
    for name, group in groups:
        name = name[0]
        # Aggregate results of repeated runs (that have the same 'conv_parameters' value)
        df_dict[name] = (
            group.groupby(by=["Model"], as_index=False)
            .agg(
                Mean=("Mean", "mean"),
                Median=("Median", "median"),
                Unit=("Unit", "first"),
                Runs=("Runs", "sum"),
                Threads=("Threads", "first"),
            )
            .sort_values(by=["Model"])
            .reset_index(drop=True)
        )

    # Join results by method
    method_names = list(df_dict.keys())

    if len(method_names) == 1:
        print("Only one method found. No comparison possible.", file=sys.stderr)
        sys.exit(-1)

    joined_results = pd.merge(
        df_dict[method_names[0]],
        df_dict[method_names[1]].drop(columns=["Unit", "Threads"]),
        how="inner",
        on="Model",
        suffixes=("_" + method_names[0], "_" + method_names[1]),
    )
    for method_name in method_names[2:]:
        joined_results = joined_results.merge(
            df_dict[method_name].drop(columns=["Unit", "Threads"]).add_suffix("_" + method_name),
            how="inner",
            left_on="Model",
            right_on="Model_" + method_name,
            suffixes=(None, None),
        ).drop(columns=["Model_" + method_name])

    # Save joined results
    joined_results.to_csv(output_dir / f"performance-results.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with performance logs and summarize them into another csv."
    )

    parser.add_argument(
        "CSV_Input", type=str, help="Path to the input CSV file with performance logs."
    )
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")

    args = parser.parse_args()

    csv_input = Path(args.CSV_Input)
    output_dir = Path(args.Output_Dir)

    # Check if csv file exists
    if (not csv_input.exists()) or (not csv_input.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)

    df = pd.read_csv(csv_input, header=0, index_col=False)
    summarize_results(df, output_dir)
