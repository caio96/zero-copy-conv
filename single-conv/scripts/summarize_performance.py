#!/usr/bin/env python3

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters
from tabulate import tabulate


def merge_results(df: pd.DataFrame, occurrences_df: pd.DataFrame, output_dir, only_stats=False):

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

    if not only_stats:
        joined_results.to_csv(output_dir / "performance-results.csv", index=False)

    return joined_results


# Define the heuristic here
def heuristic(speedup_results: pd.DataFrame):
    num_columns = len(speedup_results.columns)

    # Get all features used by heuristic from conv parameters
    from learn_heuristic import get_data  # Here to avoid circular import

    speedup_results = get_data(speedup_results)

    # Get convolutions selected by heuristic
    selection = speedup_results.query(
        "(`groups` == 1 and `dilation height` == 1 and `dilation width` == 1) and ((`dim k` > `dim n` and `dim k` > `dim m`) or `output width` == 1 or `output height` == 1)"
    )
    selection_ext = speedup_results.query(
        "(`groups` != 1 or `dilation height` != 1 or `dilation width` != 1) and (`dim m` < `dim n`)"
    )
    selection = pd.concat([selection, selection_ext])

    # Remove extra columns
    selection = selection.iloc[:, :num_columns]
    return selection


def get_speedup(
    joined_results: pd.DataFrame, old_method_name, new_method_name, use_heuristic=False
):

    # Remove rows where an error occurred in either method
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    speedup_results = pd.DataFrame()
    speedup_results["conv_parameters"] = joined_results["conv_parameters"]
    speedup_results["occurrences"] = joined_results["occurrences"]

    if use_heuristic:
        speedup_results = heuristic(speedup_results)

    # Compute speedup
    speedup_results["speedup"] = (
        joined_results["mean_time_" + old_method_name]
        - joined_results["mean_time_" + new_method_name]
    ) / joined_results["mean_time_" + new_method_name]
    speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

    return speedup_results


def plot_speedup(
    speedup_results: pd.DataFrame,
    old_method_name,
    new_method_name,
    output_dir,
    only_stats=False,
    clip_pos=False,
    clip_neg=False,
):

    def weighted_median(df: pd.DataFrame):
        df = df.sort_values("speedup")
        cumsum = df["occurrences"].cumsum()
        cutoff = df["occurrences"].sum() / 2.0
        median = df["speedup"][cumsum >= cutoff].iloc[0]
        return median

    speedup_results = speedup_results.reset_index(drop=True)
    speedup = speedup_results["speedup"]
    num_points = speedup_results.shape[0]

    inflection = num_points
    for i in range(0, num_points - 1):
        if speedup.iloc[i] > 0 and speedup.iloc[i + 1] < 0:
            inflection = i + 0.5

    pos = speedup_results.loc[lambda x: x.speedup >= 0]
    neg = speedup_results.loc[lambda x: x.speedup < 0]
    pos_speedup = pos["speedup"]
    neg_speedup = neg["speedup"]

    stats = {
        f"{new_method_name} vs {old_method_name}": ["Speedup", "Slowdown"],
        "Count": [pos_speedup.shape[0], neg_speedup.shape[0]],
        "Median": [pos_speedup.median(), neg_speedup.median()],
        "Max": [pos_speedup.max(), neg_speedup.min()],
        "Occurrences": [int(pos["occurrences"].sum()), int(neg["occurrences"].sum())],
        "Weighted Median": [weighted_median(pos), weighted_median(neg)],
    }
    print(tabulate(stats, headers="keys", tablefmt="psql", floatfmt=".2f"))
    if only_stats:
        return

    # Clip positive outliers if enabled
    if clip_pos:
        pos_threshold = pos_speedup.quantile(0.99)
        pos_speedup = np.clip(pos_speedup, 0, pos_threshold)
    # Clip negative outliers if enabled
    if clip_neg:
        neg_threshold = neg_speedup.quantile(0.01)
        neg_speedup = np.clip(neg_speedup, neg_threshold, 0)

    fig, ax = plt.subplots()

    # barplot
    ax.bar(pos_speedup.index, pos_speedup, color="#2c7bb6")
    ax.bar(
        range(pos_speedup.shape[0], pos_speedup.shape[0] + neg_speedup.shape[0], 1),
        neg_speedup.values,
        color="#d7191c",
    )

    # Add line showing that positive outliers clipped
    if clip_pos:
        ax.axhline(y=pos_threshold, color="gray", linestyle="--", linewidth=0.5)
    # Add line showing that positive outliers clipped
    if clip_neg:
        ax.axhline(y=neg_threshold, color="gray", linestyle="--", linewidth=0.5)

    # boxplot
    _, x_max = ax.get_xlim()
    ax.set_xlim((-x_max * 0.05, num_points + x_max * 0.05))
    ax.boxplot(
        [pos_speedup, neg_speedup],
        showfliers=False,
        positions=[-x_max * 0.025, num_points + x_max * 0.025],
        widths=x_max * 0.02,
    )

    ax.set_ylabel("Speedup/Slowdown")
    ax.set_xlabel("Convolutions Layers")
    ax.set_xticks([0, inflection, num_points], [0, int(inflection), num_points])

    y_min, y_max = ax.get_ylim()
    y_total = y_max - y_min

    ax.hlines(-y_total * 0.05, 1, inflection, "#2c7bb6")
    ax.vlines(1, -y_total * 0.05 - y_total * 0.01, -y_total * 0.05 + y_total * 0.01, "#2c7bb6")
    ax.vlines(
        inflection,
        -y_total * 0.05 - y_total * 0.01,
        -y_total * 0.05 + y_total * 0.01,
        "#2c7bb6",
    )
    ax.text(
        (inflection / 2),
        -y_total * 0.08,
        f"{pos_speedup.shape[0]}",
        horizontalalignment="center",
        verticalalignment="center",
    )

    if neg_speedup.shape[0] != 0:
        ax.hlines(y_total * 0.05, inflection, num_points, "#d7191c")
        ax.vlines(
            inflection,
            y_total * 0.05 - y_total * 0.01,
            y_total * 0.05 + y_total * 0.01,
            "#d7191c",
        )
        ax.vlines(
            num_points,
            y_total * 0.05 - y_total * 0.01,
            y_total * 0.05 + +y_total * 0.01,
            "#d7191c",
        )
        ax.text(
            ((num_points + inflection) / 2),
            y_total * 0.08,
            f"{neg_speedup.shape[0]}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # save figure
    plt.savefig(
        output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# Saves a csv with results and produces an speedup graph
def compare_methods(
    joined_results: pd.DataFrame,
    old_method_name,
    new_method_name,
    output_dir,
    only_stats,
    clip_pos,
    clip_neg,
    use_heuristic,
):

    speedup_results = get_speedup(joined_results, old_method_name, new_method_name)

    if use_heuristic:
        # Only print original results if heuristic is used
        plot_speedup(
            speedup_results, old_method_name, new_method_name, output_dir, True, clip_pos, clip_neg
        )
        speedup_results = get_speedup(
            joined_results, old_method_name, new_method_name, use_heuristic
        )
        new_method_name = f"Heuristic_{new_method_name}"

    # Save results to csv
    if not only_stats:
        speedup_results.to_csv(
            output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}.csv", index=False
        )

    plot_speedup(
        speedup_results,
        old_method_name,
        new_method_name,
        output_dir,
        only_stats,
        clip_pos,
        clip_neg,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with performance results and summarize them into graphs."
    )

    parser.add_argument(
        "CSV_Input", type=str, help="Path to the input CSV file (generated by benchmark_runner)."
    )
    parser.add_argument(
        "Occurrences_CSV",
        type=str,
        help="Path to the CSV file with conv_parameters and occurrences (generated by filter_csv or convolution_extraction).",
    )
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")
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
    )
    parser.add_argument(
        "--old-method",
        type=str,
        help="Set old method for speedup comparison. If not set, all methods will be compared",
    )
    parser.add_argument(
        "--new-method",
        type=str,
        help="Set new method for speedup comparison. If not set, all methods will be compared.",
    )
    parser.add_argument(
        "--only-stats",
        action="store_true",
        help="Do not save csv files or graphs, only print stats",
    )
    parser.add_argument(
        "--clip-positive-outliers",
        action="store_true",
        help="Clip positive outliers in the speedup graph",
    )
    parser.add_argument(
        "--clip-negative-outliers",
        action="store_true",
        help="Clip negative outliers in the speedup graph",
    )
    parser.add_argument(
        "--use-heuristic",
        action="store_true",
        help="Simulate a hardcoded heuristic defined in this script",
    )

    args = parser.parse_args()

    csv_input = Path(args.CSV_Input)
    occurrences_csv = Path(args.Occurrences_CSV)
    output_dir = Path(args.Output_Dir)
    exclude_conv_types = args.exclude_conv_types
    include_only_conv_types = args.include_only_conv_types
    old_method = args.old_method
    new_method = args.new_method
    only_stats = args.only_stats
    clip_pos = args.clip_positive_outliers
    clip_neg = args.clip_negative_outliers
    use_heuristic = args.use_heuristic

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
        sys.exit(-1)

    df = pd.read_csv(csv_input, header=0, index_col=False)
    occurrences_df = pd.read_csv(occurrences_csv, header=0, index_col=False)

    # Merge results by conv_params and aggregate multiple runs
    df = merge_results(df, occurrences_df, output_dir, only_stats)

    # Filter convs if needed
    num_columns = len(df.columns)
    df = split_parameters(df)
    df = include_only_in_df(df, include_only_conv_types)
    df = exclude_from_df(df, exclude_conv_types)
    df = df.iloc[:, :num_columns]

    methods = [col.replace("mean_time_", "") for col in df.columns if "mean_time" in col]

    # Check if both methods are present
    if old_method and f"mean_time_{old_method}" not in df.columns:
        print(f"Method {old_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)
    if new_method and f"mean_time_{new_method}" not in df.columns:
        print(f"Method {new_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)

    if old_method and new_method:
        compare_methods(
            df, old_method, new_method, output_dir, only_stats, clip_pos, clip_neg, use_heuristic
        )
    elif old_method:
        for method in methods:
            if method == old_method:
                continue
            compare_methods(
                df, old_method, method, output_dir, only_stats, clip_pos, clip_neg, use_heuristic
            )
    elif new_method:
        for method in methods:
            if method == new_method:
                continue
            compare_methods(
                df, method, new_method, output_dir, only_stats, clip_pos, clip_neg, use_heuristic
            )
    else:
        for method1, method2 in itertools.combinations(methods, 2):
            compare_methods(
                df, method1, method2, output_dir, only_stats, clip_pos, clip_neg, use_heuristic
            )
