#!/usr/bin/env python3

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters
from tabulate import tabulate
import scipy.stats as st


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
        # "`groups` == 1 and (`output channel` < `image channel` or `output height` == 1)"
        # "`groups` == 1 and (`output channel` < `image channel` or `output height` < `image channel`)"
        "`groups` == 1 and ((`output channel` < `image channel` and `output height` < `image channel` and `output height` > 1) or (`output channel` >= `image channel` and `output height` == 1))"
    )

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
    speedup_results["speedup"] = None

    if use_heuristic:
        speedup_results = heuristic(speedup_results)
        if speedup_results.empty:
            return speedup_results

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
    show_boxplot=True,
    show_counts=True,
    show_inflection=True,
):

    def weighted_median(df: pd.DataFrame):
        if df.empty:
            return float("nan")
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
    df_stats = pd.DataFrame(stats).fillna(0).set_index(f"{new_method_name} vs {old_method_name}")
    print(tabulate(df_stats, headers="keys", tablefmt="psql", floatfmt=".2f"))
    if only_stats:
        return

    # Clip positive outliers if enabled
    if clip_pos:
        pos_threshold = pos_speedup.quantile(0.99)
        max_pos = pos_speedup.max()
        clipped_pos_indices = pos_speedup[pos_speedup > pos_threshold].index.to_series()
        pos_speedup = np.clip(pos_speedup, 0, pos_threshold)
    # Clip negative outliers if enabled
    if clip_neg:
        neg_threshold = neg_speedup.quantile(0.01)
        min_neg = neg_speedup.min()
        clipped_neg_indices = neg_speedup[neg_speedup < neg_threshold].index.to_series()
        neg_speedup = np.clip(neg_speedup, neg_threshold, 0)

    fig, ax = plt.subplots()

    # barplot
    ax.bar(pos_speedup.index, pos_speedup, color="#0571b0", label=f"Speedup: {pos_speedup.shape[0]}")
    ax.bar(
        range(pos_speedup.shape[0], pos_speedup.shape[0] + neg_speedup.shape[0], 1),
        neg_speedup.values,
        color="#ca0020",
        label=f"Slowdown: {neg_speedup.shape[0]}"
    )

    # Add line showing that positive outliers clipped
    if clip_pos and len(clipped_pos_indices) > 0:
        mid_x_pos = clipped_pos_indices.mean()
        cutoff_x_start = clipped_pos_indices.min() - 3 * mid_x_pos
        cutoff_x_end = clipped_pos_indices.max() + 3 * mid_x_pos
        ax.hlines(pos_threshold, cutoff_x_start, cutoff_x_end, "gray", linewidth=0.5)
        # Annotate clipped value
        ax.text(
            mid_x_pos,
            pos_threshold,
            f"{max_pos:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )
    # Add line showing that positive outliers clipped
    if clip_neg:
        mid_x_neg = clipped_neg_indices.mean()
        cutoff_x_start = clipped_neg_indices.min() - 3 * mid_x_neg
        cutoff_x_end = clipped_neg_indices.max() + 3 * mid_x_neg
        ax.hlines(neg_threshold, cutoff_x_start, cutoff_x_end, "gray", linewidth=0.5)
        # Annotate clipped value
        ax.text(
            mid_x_neg,
            neg_threshold,
            f"{min_neg:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    # boxplot
    if show_boxplot:
        _, x_max = ax.get_xlim()
        ax.set_xlim((-x_max * 0.05, num_points + x_max * 0.05))
        ax.boxplot(
            [pos_speedup, neg_speedup],
            showfliers=False,
            positions=[-x_max * 0.025, num_points + x_max * 0.025],
            widths=x_max * 0.02,
        )
    else:
        x_total = pos_speedup.shape[0] + neg_speedup.shape[0]
        ax.set_xlim(left=-x_total*0.02, right=x_total*1.02)

    legend = plt.legend(frameon=True, framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    ax.set_ylabel("\% Speedup")
    ax.set_xlabel("Conv2D Layers")
    if show_inflection:
        ax.set_xticks([0, inflection, num_points], [0, int(inflection), num_points])
    else:
        ax.set_xticks([0, num_points], [0, num_points])
        ax.xaxis.set_label_coords(.5, -.05)

    if show_counts:
        y_min, y_max = ax.get_ylim()
        y_total = y_max - y_min

        ax.hlines(-y_total * 0.05, 1, inflection, "#0571b0")
        ax.vlines(1, -y_total * 0.05 - y_total * 0.01, -y_total * 0.05 + y_total * 0.01, "#0571b0")
        ax.vlines(
            inflection,
            -y_total * 0.05 - y_total * 0.01,
            -y_total * 0.05 + y_total * 0.01,
            "#0571b0",
        )
        ax.text(
            (inflection / 2),
            -y_total * 0.08,
            f"{pos_speedup.shape[0]}",
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )

        if neg_speedup.shape[0] != 0:
            ax.hlines(y_total * 0.05, inflection, num_points, "#ca0020")
            ax.vlines(
                inflection,
                y_total * 0.05 - y_total * 0.01,
                y_total * 0.05 + y_total * 0.01,
                "#ca0020",
            )
            ax.vlines(
                num_points,
                y_total * 0.05 - y_total * 0.01,
                y_total * 0.05 + +y_total * 0.01,
                "#ca0020",
            )
            ax.text(
                ((num_points + inflection) / 2),
                y_total * 0.08,
                f"{neg_speedup.shape[0]}",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
            )
    else:
        ax.set_ylim(top=pos_speedup.max(), bottom=neg_speedup.min())

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

    if not only_stats:
        # Add graph with execution times for comparison
        methods = [col.replace("mean_time_", "") for col in joined_results.columns if "mean_time" in col]
        graph_execution_times(joined_results, methods, output_dir, old_method, new_method)

        # Save results to csv
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


# Function to estimate the FLOPs of a convolution
# Used to sort the convolutions by complexity
def compute_conv_flops(
    input_channels, input_height, input_width,
    output_channels, kernel_height, kernel_width,
    stride_height=1, stride_width=1, padding_height=0, padding_width=0,
    dilation_height=1, dilation_width=1, groups=1
):
    # Compute output dimensions
    output_height = ((input_height + 2 * padding_height - (dilation_height * (kernel_height - 1) + 1)) // stride_height) + 1
    output_width = ((input_width + 2 * padding_width - (dilation_width * (kernel_width - 1) + 1)) // stride_width) + 1
    # Channels per group
    input_channels_per_group = input_channels // groups
    # Compute FLOPs
    flops = output_channels * output_height * output_width * input_channels_per_group * kernel_height * kernel_width
    return flops


def graph_execution_times(df: pd.DataFrame, methods, output_dir, old_method=None, new_method=None):
    fig, ax = plt.subplots()

    df = split_parameters(df)
    df["flops"] = compute_conv_flops(df["image channel"], df["image height"], df["image width"],
                                     df["output channel"], df["filter height"], df["filter width"],
                                     df["stride height"], df["stride width"], df["padding top"], df["padding left"],
                                     df["dilation height"], df["dilation width"], df["groups"])
    df = df.sort_values(by=["flops"])
    marker=['o', 'v', '^', '<', '>', 's', 'p', '*', 'X']

    name_translation = {
            "ZeroCopy_jit": "ZConv",
            "OneDNN_jit": "OneDNN",
            "LibTorch_ZeroCopy2D_HWIO_TransformOutput": "Torch_ZConv",
            "LibTorch": "Torch",
            "Im2col": "Im2col",
            "Yaconv": "Yaconv",
    }

    for idx, method in enumerate(sorted(methods)):
        if old_method and new_method and method not in (old_method, new_method):
            continue
        label_name = name_translation[method] if method in name_translation else method
        method_means = df[f"mean_time_{method}"]
        method_iterations = df[f"total_iterations_{method}"]

        # Note that std deviation only sees the means from repeated runs, not the individual runs
        method_std = df[f"std_time_{method}"]

        # Calculate the 95% confidence interval
        conf_low, conf_high = st.norm.interval(confidence=0.95, loc=method_means, scale=method_std/np.sqrt(method_iterations))
        conf = (conf_high - conf_low) / 2

        ax.errorbar(range(df.shape[0]), method_means, yerr=conf, label=label_name, markersize=2, markeredgecolor='black', markeredgewidth=0.1, fmt=marker[idx%len(marker)], alpha=0.8, ecolor='black', elinewidth=0.5)

    legend = plt.legend(frameon=True, framealpha=1, markerscale=3)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    ax.set_ylabel("Execution time (ms)")
    ax.set_xlabel("Conv2D Layers")

    comparison_name = ""
    if old_method:
        comparison_name += f"{old_method}_"
    else:
        comparison_name += "all_"
    comparison_name += "vs_"
    if new_method:
        comparison_name += new_method
    else:
        comparison_name += "all"

    # save figure
    plt.savefig(
        output_dir / f"conv2d_{comparison_name}_execution_times.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


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

    if not only_stats:
        rc('font', **{'family': 'serif', 'serif': ['Libertine']})
        rc('text', usetex=True)
        rc('text.latex', preamble="\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{libertine}",
            r"\usepackage{newtxtext,newtxmath}",
        ]))
        plt.rcParams.update({
            "font.size": 16,
            "legend.fontsize": 16,
        })

        # Add graph with execution times for all methods
        graph_execution_times(df, methods, output_dir)

    # Check if both methods are present
    if old_method and old_method not in methods:
        print(f"Method {old_method} not found in results.", file=sys.stderr)
        print(f"Available methods: {methods}", file=sys.stderr)
        sys.exit(-1)
    if new_method and new_method not in methods:
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
