#!/usr/bin/env python3

import argparse
import sys
import itertools
import numpy as np
from pathlib import Path
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters


def get_speedup(joined_results: pd.DataFrame, old_method_name, new_method_name):

    # Remove rows where an error occurred in either method
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    speedup_results = pd.DataFrame()
    speedup_results["conv_parameters"] = joined_results["conv_parameters"]
    speedup_results["occurrences"] = joined_results["occurrences"]

    # Compute speedup
    speedup_results["speedup"] = (
        joined_results["mean_time_" + old_method_name]
        - joined_results["mean_time_" + new_method_name]
    ) / joined_results["mean_time_" + new_method_name]
    speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

    return speedup_results


def plot_speedup(speedup_results: pd.DataFrame, old_method_name, new_method_name, output_dir, only_stats=False, clip_pos=False, clip_neg=False):

    def weighted_median(df: pd.DataFrame):
        df = df.sort_values('speedup')
        cumsum = df["occurrences"].cumsum()
        cutoff = df["occurrences"].sum() / 2.0
        median = df["speedup"][cumsum >= cutoff].iloc[0]
        return median

    speedup = speedup_results["speedup"]
    occurrences = speedup_results["occurrences"]
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
        "Weighted Median": [weighted_median(pos), weighted_median(neg)]
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
    pos_bars = ax.bar(pos_speedup.index, pos_speedup, color="#2c7bb6")
    neg_bars = ax.bar(range(pos_speedup.shape[0], pos_speedup.shape[0] + neg_speedup.shape[0], 1), neg_speedup.values, color="#d7191c")

    # Add line showing that positive outliers clipped
    if clip_pos:
        ax.axhline(y=pos_threshold, color='gray', linestyle='--', linewidth=0.5)
    # Add line showing that positive outliers clipped
    if clip_neg:
        ax.axhline(y=neg_threshold, color='gray', linestyle='--', linewidth=0.5)

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

    y_factor = 0.1
    y_min, y_max = ax.get_ylim()
    if y_min < 0:
        relative_y = min(-y_min * y_factor, y_max * y_factor)
    else:
        relative_y = y_max * y_factor

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
def compare_methods(joined_results: pd.DataFrame, old_method_name, new_method_name, output_dir, only_stats, clip_pos, clip_neg):

    speedup_results = get_speedup(joined_results, old_method_name, new_method_name)

    # Save resulst to csv
    if not only_stats:
        speedup_results.to_csv(
            output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}.csv", index=False
        )

    plot_speedup(speedup_results, old_method_name, new_method_name, output_dir, only_stats, clip_pos, clip_neg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with performance results and summarize them into graphs."
    )

    parser.add_argument("CSV_Results", type=str, help="Path to the output CSV file.")
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
        help="Skip generating graphs and only print stats",
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

    args = parser.parse_args()

    csv_results = Path(args.CSV_Results)
    output_dir = Path(args.Output_Dir)
    exclude_conv_types = args.exclude_conv_types
    include_only_conv_types = args.include_only_conv_types
    old_method = args.old_method
    new_method = args.new_method
    only_stats = args.only_stats
    clip_pos = args.clip_positive_outliers
    clip_neg = args.clip_negative_outliers

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
    if old_method and f"mean_time_{old_method}" not in df.columns:
        print(f"Method {old_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)
    if new_method and f"mean_time_{new_method}" not in df.columns:
        print(f"Method {new_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)

    if old_method and new_method:
        compare_methods(df, old_method, new_method, output_dir, only_stats, clip_pos, clip_neg) 
    elif old_method:
        for method in methods:
            if method == old_method:
                continue
            compare_methods(df, old_method, method, output_dir, only_stats, clip_pos, clip_neg) 
    elif new_method:
        for method in methods:
            if method == new_method:
                continue
            compare_methods(df, method, new_method, output_dir, only_stats, clip_pos, clip_neg) 
    else:
        for method1, method2 in itertools.combinations(methods, 2):
            compare_methods(df, method1, method2, output_dir, only_stats, clip_pos, clip_neg)
