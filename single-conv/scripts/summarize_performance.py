#!/usr/bin/env python3

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters, get_categories
from tabulate import tabulate
import scipy.stats as st
from matplotlib.ticker import FuncFormatter


def merge_results(df: pd.DataFrame, occurrences_df: pd.DataFrame, output_dir, only_stats=False, incorrect_convs: pd.DataFrame=None):

    # Split the 'name' column into 'conv_type' and 'conv_parameters'
    df[["conv_type", "conv_parameters"]] = df["name"].str.split(" ", n=1, expand=True)
    df = df.drop(columns=["name"])
    df["conv_parameters"] = df["conv_parameters"].str.split("/", n=1).str[0]

    # Remove rows where error_occurred is True (happens when Yaconv is not supported)
    df = df.loc[df["error_occurred"] != True]

    # Removes the rows from df if the conv_type and conv_parameters are present in incorrect_convs
    if incorrect_convs is not None:
        df = df.merge(incorrect_convs, how="left", on=["conv_type", "conv_parameters"], indicator=True)
        df = df.loc[df["_merge"] == "left_only"]
        df = df.drop(columns=["_merge", "max_diff", "tolerance"])

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
                repeats=("real_time", "count"),
                mean_time=("real_time", "mean"),
                sem_time=("real_time", "sem"),
                time_unit=("time_unit", "first"),
                error_occurred=("error_occurred", "any"),
            )
            .sort_values(by=["conv_parameters"])
            .reset_index(drop=True)
        )

    # Join results by 'conv_parameters'
    method_names = list(df_dict.keys())
    # Add the 95% confidence interval
    for method_name in method_names:
        means = df_dict[method_name][f"mean_time"]
        iterations = df_dict[method_name][f"repeats"]
        sems = df_dict[method_name][f"sem_time"]

        conf_low, conf_high = st.norm.interval(confidence=0.95, loc=means, scale=sems)
        conf = (conf_high - conf_low) / 2
        df_dict[method_name][f"95_confidence"] = conf

    if len(method_names) == 1:
        print("Only one method found. No comparison possible.", file=sys.stderr)
        sys.exit(-1)

    joined_results = pd.merge(
        df_dict[method_names[0]],
        df_dict[method_names[1]].drop(columns=["time_unit"]),
        how="left",
        on="conv_parameters",
        suffixes=("_" + method_names[0], "_" + method_names[1]),
    )
    for method_name in method_names[2:]:
        joined_results = joined_results.merge(
            df_dict[method_name].drop(columns=["time_unit"]).add_suffix("_" + method_name),
            how="left",
            left_on="conv_parameters",
            right_on="conv_parameters_" + method_name,
            suffixes=(None, None),
        ).drop(columns=["conv_parameters_" + method_name])

    # Add occurrences column
    joined_results = joined_results.merge(occurrences_df, how="left", on="conv_parameters")

    return joined_results


def get_speedup(
    joined_results: pd.DataFrame, old_method_name, new_method_name
):
    # Remove rows where an error occurred in either method
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    speedup_results = pd.DataFrame()
    speedup_results["conv_parameters"] = joined_results["conv_parameters"]
    speedup_results["occurrences"] = joined_results["occurrences"]
    speedup_results["speedup"] = joined_results["mean_time_" + old_method_name] / joined_results["mean_time_" + new_method_name]
    speedup_results["slowdown"] = joined_results["mean_time_" + new_method_name] / joined_results["mean_time_" + old_method_name]

    # Compute speedup and slowdown -> results in asymmetric speedup and slowdown where 0 means no change
    speedup_results["relative_change"] = speedup_results.apply(
        lambda row: row["speedup"]-1 if row["speedup"] >= 1 else (row["slowdown"]-1)*-1, axis=1
    )

    # Compute log2 speedup -> results in symmetric speedup and slowdown
    speedup_results["log2_speedup"] = np.log2(joined_results["mean_time_" + old_method_name] / joined_results["mean_time_" + new_method_name])

    # Compute time difference between methods
    speedup_results["time_diff"] = joined_results["mean_time_" + old_method_name] - joined_results["mean_time_" + new_method_name]
    speedup_results["time_unit"] = joined_results["time_unit"]

    old_means = joined_results["mean_time_" + old_method_name]
    old_conf = joined_results["95_confidence_" + old_method_name]
    old_lower = old_means - old_conf
    old_higher = old_means + old_conf

    new_means = joined_results["mean_time_" + new_method_name]
    new_conf = joined_results["95_confidence_" + new_method_name]
    new_lower = new_means - new_conf
    new_higher = new_means + new_conf

    # Check if the difference is significant (i.e. the confidence intervals do not overlap)
    speedup_results["significant"] = (old_lower > new_higher) | (new_lower > old_higher)

    return speedup_results


def plot_speedup(
    speedup_results: pd.DataFrame,
    old_method_name,
    new_method_name,
    output_dir,
    plot_type,
    only_stats=False,
    clip_pos=False,
    clip_neg=False,
    show_boxplot=False,
    show_counts=False,
    show_inflection=False,
):
    if speedup_results.empty:
        print("No data to plot.", file=sys.stderr)
        return

    if plot_type == "time_diff":
        speedup_results = speedup_results.sort_values(by="time_diff", ascending=False)
    else:
        speedup_results = speedup_results.sort_values(by="relative_change", ascending=False)

    def weighted_median(df: pd.DataFrame):
        if df.empty:
            return float("nan")
        df = df.sort_values("relative_change")
        cumsum = df["occurrences"].cumsum()
        cutoff = df["occurrences"].sum() / 2.0
        median = df["relative_change"][cumsum >= cutoff].iloc[0]
        return median

    non_significant_results = speedup_results.loc[lambda x: x.significant == False]
    non_significant_count = non_significant_results.shape[0]
    speedup_results = speedup_results.loc[lambda x: x.significant == True]

    # Save non-significant models to csv if the performance change is greater than 1%
    try_rerun_layers = pd.DataFrame()
    if not only_stats:
        try_rerun_layers["conv_parameters"] = non_significant_results.loc[lambda x: np.abs(x.relative_change) > 0.01]["conv_parameters"]
        try_rerun_layers["rerun_methods"] = f"{old_method_name},{new_method_name}"
        try_rerun_layers.to_csv(output_dir / f"{new_method_name}_vs_{old_method_name}_try_rerun.csv", index=False)

    speedup_results = speedup_results.reset_index(drop=True)
    num_points = speedup_results.shape[0]

    inflection = num_points
    for i in range(0, num_points - 1):
        if speedup_results["relative_change"].iloc[i] >= 0 and speedup_results["relative_change"].iloc[i + 1] < 0:
            inflection = i + 0.5

    pos = speedup_results.loc[lambda x: x.relative_change >= 0]
    neg = speedup_results.loc[lambda x: x.relative_change < 0]
    pos_speedup = pos["relative_change"]
    neg_speedup = neg["relative_change"]
    unit = speedup_results["time_unit"].iloc[0]

    stats = {
        f"{new_method_name} vs {old_method_name}": ["Speedup", "Slowdown"],
        "Count": [pos_speedup.shape[0], neg_speedup.shape[0]],
        "Median": [pos_speedup.median(), neg_speedup.median()],
        "Max": [pos_speedup.max(), neg_speedup.min()],
        f"Time Difference ({unit})": [(pos["time_diff"]).sum(), (neg["time_diff"]).sum()],
        "Occurrences": [int(pos["occurrences"].sum()), int(neg["occurrences"].sum())],
        "Weighted Median": [weighted_median(pos), weighted_median(neg)],
        f"Weighted Time Difference ({unit})": [(pos["time_diff"] * pos["occurrences"]).sum(), (neg["time_diff"] * neg["occurrences"]).sum()],
        "No significant change": [non_significant_count, ""],
    }
    df_stats = pd.DataFrame(stats).fillna(0).set_index(f"{new_method_name} vs {old_method_name}")
    print(tabulate(df_stats, headers="keys", tablefmt="psql", floatfmt=".2f"))
    if only_stats:
        return
    df_stats.to_csv(output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}_stats.csv")

    if plot_type == "log2_speedup":
        pos_speedup = pos["log2_speedup"]
        neg_speedup = neg["log2_speedup"]
    elif plot_type == "time_diff":
        pos_speedup = pos["time_diff"]
        neg_speedup = neg["time_diff"]

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

    # fig, ax = plt.subplots(figsize=(7.75, 4.8))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    if plot_type == "log2_speedup":
        def custom_formatter(x, pos):
            x = 2**x
            if (x < 1):
                x = 1/x
                return f"$\\frac{{1}}{{{x:.2g}}}$"
            return f"{x:.2g}"

        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    elif plot_type == "speedup":
        def custom_formatter(x, pos):
            if (x >= 0):
                x += 1
                return f"{x:.3g}"
            else:
                x -= 1
                x *= -1
                return f"$\\frac{{1}}{{{x:.3g}}}$"

        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # barplot
    if not pos_speedup.empty:
        label = f"Count: {pos_speedup.shape[0]}" if not neg_speedup.empty else None
        ax.bar(pos_speedup.index, pos_speedup, color="#0571b0", label=label)
    if not neg_speedup.empty:
        label = f"Count: {neg_speedup.shape[0]}" if not pos_speedup.empty else None
        ax.bar(
            range(pos_speedup.shape[0], pos_speedup.shape[0] + neg_speedup.shape[0], 1),
            neg_speedup.values,
            color="#ca0020",
            label=label
        )

    # Add line showing that positive outliers clipped
    if clip_pos and len(clipped_pos_indices) > 0:
        mid_x_pos = clipped_pos_indices.mean()
        cutoff_x_start = clipped_pos_indices.min() - 3 * mid_x_pos
        cutoff_x_end = clipped_pos_indices.max() + 3 * mid_x_pos
        ax.hlines(pos_threshold, cutoff_x_start, cutoff_x_end, "gray", linewidth=0.5)
        # Annotate clipped value
        text = ""
        if plot_type == "log2_speedup":
            text = f"Max: {2**max_pos:.1f}"
        elif plot_type == "speedup":
            max_pos += 1
            text = f"Max: {max_pos:.3g}"
        else:
            text = f"Max: {max_pos:.2g}"
        ax.text(
            mid_x_pos,
            pos_threshold,
            text,
            ha="left",
            va="bottom",
            fontsize=20,
            color="black",
        )
    # Add line showing that positive outliers clipped
    if clip_neg and len(clipped_neg_indices) > 0:
        mid_x_neg = clipped_neg_indices.mean()
        cutoff_x_start = clipped_neg_indices.min() - 3 * mid_x_neg
        cutoff_x_end = clipped_neg_indices.max() + 3 * mid_x_neg
        ax.hlines(neg_threshold, cutoff_x_start, cutoff_x_end, "gray", linewidth=0.5)
        # Annotate clipped value
        text = ""
        if plot_type == "log2_speedup":
            min_neg = 1/(2**min_neg)
            text = f"Min: $\\frac{{1}}{{{min_neg:.1f}}}$"
        elif plot_type == "speedup":
            min_neg -= 1
            min_neg *= -1
            text = f"Min: $\\frac{{1}}{{{min_neg:.3g}}}$"
        else:
            text = f"Min: {min_neg:.2g}"
        y_min, y_max = ax.get_ylim()
        y_total = y_max - y_min
        ax.text(
            mid_x_neg,
            neg_threshold - y_total * 0.02,
            text,
            ha="right",
            va="top",
            fontsize=20,
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

    if not pos_speedup.empty and not neg_speedup.empty:
        legend = plt.legend(frameon=True, framealpha=1, handlelength=1)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')

    if plot_type == "time_diff":
        ax.set_ylabel(f"Time Difference ({unit})")
    else:
        ax.set_ylabel("Speedup")
    ax.set_xlabel(f"Conv2D Layers ({num_points} Total)")
    ax.xaxis.set_label_coords(0.5, -0.025)
    if show_inflection:
        ax.set_xticks([0, inflection, num_points], [0, int(inflection), num_points])
    else:
        ax.set_xticks([])

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
            fontsize=20,
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
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center",
            )
    else:
        top = pos_speedup.max() if clip_pos and not pos_speedup.empty else None
        bottom = neg_speedup.min() if clip_neg and not neg_speedup.empty else None
        ax.set_ylim(top=top, bottom=bottom)

    # save figure
    plt.savefig(
        output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}_{plot_type}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


def speedup_per_category(speedup_results: pd.DataFrame, output_csv: Path, only_stats: bool=False):
    speedup_results = split_parameters(speedup_results)
    unit = speedup_results["time_unit"].iloc[0]
    stats = {
        "Category": [],
        "Speedup Count": [],
        "Slowdown Count": [],
        "Count Ratio": [],
        f"Speedup Time ({unit})": [],
        f"Slowdown Time ({unit})": [],
        "Time Ratio": [],
    }

    # Remove rows where speedup or slowdown is less than 0.01
    speedup_results = speedup_results.loc[lambda x: x.relative_change.abs() >= 0.01].copy()

    for category in get_categories():
        df = include_only_in_df(speedup_results, [category])
        if df.empty:
            continue
        pos = df.loc[lambda x: x.relative_change >= 0]
        neg = df.loc[lambda x: x.relative_change < 0]
        stats["Category"].append(category)
        stats["Speedup Count"].append(pos.shape[0])
        stats["Slowdown Count"].append(neg.shape[0])
        stats["Count Ratio"].append(pos.shape[0] / neg.shape[0] if neg.shape[0] != 0 else 0)
        stats[f"Speedup Time ({unit})"].append(pos["time_diff"].sum())
        stats[f"Slowdown Time ({unit})"].append(neg["time_diff"].sum())
        stats["Time Ratio"].append(pos["time_diff"].sum() / neg["time_diff"].sum() * -1 if neg["time_diff"].sum() != 0 else 0)

    df_stats = pd.DataFrame(stats).fillna(0).set_index("Category")
    print(tabulate(df_stats, headers="keys", tablefmt="psql", floatfmt=".2f"))
    if not only_stats:
        df_stats.to_csv(
            output_csv
        )


# Saves a csv with results and produces an speedup graph
def compare_methods(
    joined_results: pd.DataFrame,
    old_method_name,
    new_method_name,
    output_dir,
    only_stats,
    clip_pos,
    clip_neg,
    plot_type,
):

    speedup_results = get_speedup(joined_results, old_method_name, new_method_name)

    if not only_stats:
        # Add graph with execution times for comparison
        methods = [col.replace("mean_time_", "") for col in joined_results.columns if "mean_time" in col]
        graph_execution_times(joined_results, methods, output_dir, old_method, new_method)

        if plot_type == "time_diff":
            speedup_results = speedup_results.sort_values(by="time_diff", ascending=False)
        else:
            speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

        # Save results to csv
        speedup_results.to_csv(
            output_dir / f"conv2d_{new_method_name}_vs_{old_method_name}.csv", index=False
        )

    speedup_per_category(speedup_results, output_dir/f"conv2d_{new_method_name}_vs_{old_method_name}_categories.csv", only_stats)

    plot_speedup(
        speedup_results,
        old_method_name,
        new_method_name,
        output_dir,
        plot_type,
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
            "ZeroCopy_jit": "ZConv_T", # transposed HW
            "ZeroCopy_no_transpose_mkl_jit": "ZConv", # transposed HW
            "OneDNN_jit": "OneDNN",
            "LibTorch_ZeroCopy2D_HWIO_TransformOutput": "Torch_ZConv_T", # transposed HW
            "LibTorch_ZeroCopy2D_no_transpose_HWIO": "Torch_ZConv",
            "LibTorch": "Torch",
            "Im2col": "Im2col",
            "Yaconv": "Yaconv",
    }

    for idx, method in enumerate(sorted(methods)):
        if old_method and new_method and method not in (old_method, new_method):
            continue
        label_name = name_translation[method] if method in name_translation else method
        method_means = df[f"mean_time_{method}"]
        conf = df[f"95_confidence_{method}"]

        ax.errorbar(range(df.shape[0]), method_means, yerr=conf, label=label_name, markersize=2, markeredgecolor='black', markeredgewidth=0.1, fmt=marker[idx%len(marker)], alpha=0.8, ecolor='black', elinewidth=0.5)

    legend = plt.legend(frameon=True, framealpha=1, markerscale=3)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    unit = df["time_unit"].iloc[0]
    ax.set_ylabel(f"Execution time ({unit})")
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
        choices=get_categories(),
        default=None,
    )
    parser.add_argument(
        "--exclude-conv-types",
        nargs="+",
        type=str,
        help="List of convolution types to exclude",
        choices=get_categories(),
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
        "--preset-comparisons",
        action="store_true",
        help="Use preset comparisons between methods to generate results.",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["speedup", "log2_speedup", "time_diff"],
        default="log2_speedup",
        help="Data to plot. Speedup is the relative speedup and slowdown, log2_speedup is the log2 of speedup, and time_diff is the difference in time between methods. Default is log2_speedup",
    )
    parser.add_argument(
        "--incorrect-convs",
        type=str,
        help="Path to csv that contains convolution that generated incorrect results. Excludes these convolutions from the graphs if the method that generated them is part of the comparison.",
    )
    parser.add_argument(
        "--already-merged",
        action="store_true",
        help="Indicates that input csv points to the performance-results.csv previously generated by this scripts instead of the benchmark_runner.",
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
    preset_comparisons = args.preset_comparisons
    plot_type = args.plot_type
    incorrect_convs = args.incorrect_convs
    already_merged = args.already_merged

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

    if not already_merged:
        incorrect_conv_df = None
        if incorrect_convs:
            incorrect_convs = Path(incorrect_convs)
            if (not incorrect_convs.exists()) or (not incorrect_convs.is_file()):
                print("CSV with incorrect convolutions not found.", file=sys.stderr)
                sys.exit(-1)
            incorrect_conv_df = pd.read_csv(incorrect_convs, header=0, index_col=False)

        df = pd.read_csv(csv_input, header=0, index_col=False, dtype={"error_occurred": "boolean", "error_message": str})
        df["error_occurred"] = df["error_occurred"].fillna(False)
        occurrences_df = pd.read_csv(occurrences_csv, header=0, index_col=False)

        # Merge results by conv_params and aggregate multiple runs
        df = merge_results(df, occurrences_df, output_dir, only_stats, incorrect_conv_df)
    else:
        if incorrect_convs:
            print("Incorrect convolutions cannot be used with already merged results.", file=sys.stderr)
            sys.exit(-1)
        df = pd.read_csv(csv_input, header=0, index_col=False)

    # Filter convs if needed
    num_columns = len(df.columns)
    df = split_parameters(df)
    df = include_only_in_df(df, include_only_conv_types)
    df = exclude_from_df(df, exclude_conv_types)
    df = df.iloc[:, :num_columns]

    if df.empty:
        print("No data to process.", file=sys.stderr)
        sys.exit(-1)

    if not only_stats and not already_merged:
        df.to_csv(output_dir / "performance-results.csv", index=False)

    methods = [col.replace("mean_time_", "") for col in df.columns if "mean_time" in col]

    if not only_stats:
        rc('font', **{'family': 'serif', 'serif': ['Libertine']})
        rc('text', usetex=True)
        rc('text.latex', preamble="\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{libertine}",
            r"\usepackage{newtxtext,newtxmath}",
            r"\usepackage{amsmath}",
        ]))
        plt.rcParams.update({
            "font.size": 22,
            "legend.fontsize": 20,
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
            df, old_method, new_method, output_dir, only_stats, clip_pos, clip_neg, plot_type
        )
    elif old_method:
        for method in methods:
            if method == old_method:
                continue
            compare_methods(
                df, old_method, method, output_dir, only_stats, clip_pos, clip_neg, plot_type
            )
    elif new_method:
        for method in methods:
            if method == new_method:
                continue
            compare_methods(
                df, method, new_method, output_dir, only_stats, clip_pos, clip_neg, plot_type
            )
    else:
        if preset_comparisons:
            comparisons = [("Im2col", "ZeroCopy_no_transpose_mkl_jit"), ("Yaconv", "ZeroCopy_no_transpose_mkl_jit"), ("OneDNN_any", "ZeroCopy_no_transpose_mkl_jit"), ("LibTorch", "LibTorch_ZeroCopy2D_no_transpose_HWIO")]
            for method1, method2 in comparisons:
                if method1 not in methods or method2 not in methods:
                    continue
                compare_methods(
                    df, method1, method2, output_dir, only_stats, clip_pos, clip_neg, plot_type
                )
        else:
            for method1, method2 in itertools.combinations(methods, 2):
                compare_methods(
                    df, method1, method2, output_dir, only_stats, clip_pos, clip_neg, plot_type
                )

    # Merge try_rerun csvs
    if not only_stats and not already_merged:
        rerun_df_merge = pd.DataFrame()
        for file in output_dir.glob("*_try_rerun.csv"):
            rerun_df = pd.read_csv(file)
            rerun_df_merge = pd.concat([rerun_df_merge, rerun_df], ignore_index=True)
        # Aggregate rerun results by conv_params
        aggregated_rerun_df = rerun_df_merge.groupby("conv_parameters").agg(conv_parameters=("conv_parameters", "first"), rerun_methods=("rerun_methods", lambda x: ",".join(x)))
        # Remove duplicates from rerun methods
        aggregated_rerun_df["rerun_methods"] = aggregated_rerun_df["rerun_methods"].apply(lambda x: ",".join(set(x.split(","))))
        aggregated_rerun_df.to_csv(output_dir / "try_rerun.csv", index=False)
