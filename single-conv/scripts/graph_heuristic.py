#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters
from graph_performance import plot_speedup
from learn_heuristic import get_data


# Define the heuristic here
def heuristic(speedup_results: pd.DataFrame):
    num_columns = len(speedup_results.columns)

    # Get all features used by heuristic from conv parameters
    speedup_results = get_data(speedup_results)

    # Get convolutions selected by heuristic
    selection = speedup_results.query("(`groups` == 1 and `dilation height` == 1 and `dilation width` == 1) and `k dim` > `n dim`")
    selection_ext = speedup_results.query("(`groups` != 1 or `dilation height` != 1 or `dilation width` != 1) and `filter height` <= `n dim`")
    selection = pd.concat([selection, selection_ext])

    # Remove extra columns
    selection = selection.iloc[:, :num_columns]
    return selection


def simulate_heuristic_speedup(joined_results: pd.DataFrame, old_method_name, new_method_name):

    # Remove rows where an error occurred in either method
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    speedup_results = pd.DataFrame()
    speedup_results["conv_parameters"] = joined_results["conv_parameters"]

    speedup_results = heuristic(speedup_results)

    # Compute speedup
    speedup_results["speedup"] = (
        joined_results["mean_time_" + old_method_name]
        - joined_results["mean_time_" + new_method_name]
    ) / joined_results["mean_time_" + new_method_name]
    speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

    return speedup_results


# Saves a csv with results and produces an speedup graph
def compare_methods(joined_results: pd.DataFrame, old_method_name, new_method_name, only_stats):

    speedup_results = simulate_heuristic_speedup(joined_results, old_method_name, new_method_name)
    heuristic_name = f"Heuristic_{new_method_name}"

    # Save resulst to csv
    speedup_results.to_csv(
        output_dir / f"conv2d_{heuristic_name}_vs_{old_method_name}.csv", index=False
    )

    speedup = speedup_results["speedup"]
    plot_speedup(speedup, old_method_name, heuristic_name, output_dir, only_stats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulates a heuristic that chooses between two convolution methods and generates a speedup graph using that heuristic. Heuristic is hardcoded in this script"
    )

    parser.add_argument("CSV_Results", type=str, help="Path to the output CSV file.")
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")

    parser.add_argument(
        "Old_Method",
        type=str,
        help="Set old method for speedup comparison. This should be something like default Libtorch or OneDNN",
    )
    parser.add_argument(
        "New_Method",
        type=str,
        help="Set new method for speedup comparison. This should be ZeroCopy convolution.",
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
        default=None,
    )
    parser.add_argument(
        "--only-stats",
        action="store_true",
        help="Skip generating graphs and only print stats",
    )

    args = parser.parse_args()

    csv_results = Path(args.CSV_Results)
    output_dir = Path(args.Output_Dir)
    old_method = args.Old_Method
    new_method = args.New_Method
    exclude_conv_types = args.exclude_conv_types
    include_only_conv_types = args.include_only_conv_types
    only_stats = args.only_stats

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)
        sys.exit(-1)

    df = pd.read_csv(csv_results, header=0, index_col=False)

    # Filter convs if needed
    num_columns = len(df.columns)
    df = split_parameters(df)
    df = include_only_in_df(df, include_only_conv_types)
    df = exclude_from_df(df, exclude_conv_types)
    df = df.iloc[:, :num_columns]

    methods = [col.replace("mean_time_", "") for col in df.columns if "mean_time" in col]

    # Check if both methods are present
    if f"mean_time_{old_method}" not in df.columns:
        print(f"Method {old_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)
    if f"mean_time_{new_method}" not in df.columns:
        print(f"Method {new_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)

    compare_methods(df, old_method, new_method, only_stats)
