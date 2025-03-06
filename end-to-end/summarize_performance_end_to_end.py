#!/usr/bin/env python3

import argparse
import itertools
import sys
from pathlib import Path

import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.ticker import FuncFormatter


def merge_results(df: pd.DataFrame, output_dir, only_stats=False):

    # Separate df by 'conv_type'
    groups = df.groupby(by=["Method Name"])
    df_dict = {}
    for name, group in groups:
        name = name[0]
        # Aggregate results of repeated runs (that have the same 'conv_parameters' value)
        df_dict[name] = (
            group.groupby(by="Model", as_index=False)
            .agg(
                Iterations=("Time", "count"),
                Median=("Time", "median"),
                IQR=("Time", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                Unit=("Unit", "first"),
            )
            .sort_values(by=["Model"])
            .reset_index(drop=True)
        )

    method_names = list(df_dict.keys())

    # Add the iqr and confidence columns
    for method_name in method_names:
        medians = df_dict[method_name]["Median"]
        iqrs = df_dict[method_name]["IQR"]
        df_dict[method_name]["Confident"] = (iqrs / medians) < 0.1

    if len(method_names) == 1:
        print("Only one method found. No comparison possible.", file=sys.stderr)
        sys.exit(-1)

    joined_results = pd.merge(
        df_dict[method_names[0]],
        df_dict[method_names[1]].drop(columns=["Unit"]),
        how="left",
        on="Model",
        suffixes=("_" + method_names[0], "_" + method_names[1]),
    )
    for method_name in method_names[2:]:
        joined_results = joined_results.merge(
            df_dict[method_name].drop(columns=["Unit"]).add_suffix("_" + method_name),
            how="left",
            left_on="Model",
            right_on="Model_" + method_name,
            suffixes=(None, None),
        ).drop(columns=["Model_" + method_name])

    # Add overall confident column
    joined_results[f"Overall_Confident"] = True
    for method_name in method_names:
        joined_results[f"Overall_Confident"] = joined_results[f"Overall_Confident"] & joined_results[f"Confident_{method_name}"]

    return joined_results


def graph_execution_times(df: pd.DataFrame, methods, output_dir, old_method=None, new_method=None):
    fig, ax = plt.subplots()

    df = df.sort_values(by=["Median_Torch"])
    marker=['o', 'v', '^', '<', '>', 's', 'p', '*', 'X']

    name_translation = {
            "ZeroCopy2d_Heuristic": "ZConv",
            "Torch": "Torch",
    }

    for idx, method in enumerate(sorted(methods)):
        if old_method and new_method and method not in (old_method, new_method):
            continue
        label_name = name_translation[method] if method in name_translation else method
        method_medians = df[f"Median_{method}"]
        method_conf = df[f"IQR_{method}"]

        ax.errorbar(range(df.shape[0]), method_medians, yerr=method_conf, label=label_name, markersize=2, markeredgecolor='black', markeredgewidth=0.1, fmt=marker[idx%len(marker)], alpha=0.8, ecolor='black', elinewidth=0.5)

    legend = plt.legend(frameon=True, framealpha=1, markerscale=3)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    unit = df["Unit"].iloc[0]
    ax.set_ylabel(f"Execution time ({unit})")
    ax.set_xlabel("Models")

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
        output_dir / f"end_to_end_{comparison_name}_execution_times.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def get_speedup(
    joined_results: pd.DataFrame, old_method_name, new_method_name
):
    speedup_results = pd.DataFrame()
    speedup_results["Model"] = joined_results["Model"]
    speedup_results["speedup"] = None

    # Compute speedup and slowdown -> results in asymmetric speedup and slowdown where 0 means no change
    speedup = (joined_results["Median_" + old_method_name] / joined_results["Median_" + new_method_name]) - 1
    slowdown = (-1 * joined_results["Median_" + new_method_name] / joined_results["Median_" + old_method_name]) + 1
    speedup_results["speedup"] = speedup.where(speedup >= 1, slowdown)

    # Compute log2 speedup -> results in symmetric speedup and slowdown
    speedup_results["log2_speedup"] = np.log2(joined_results["Median_" + old_method_name] / joined_results["Median_" + new_method_name])

    # Compute time difference between methods
    speedup_results["time_diff"] = joined_results["Median_" + old_method_name] - joined_results["Median_" + new_method_name]
    speedup_results["Unit"] = joined_results["Unit"]

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

    if plot_type == "time_diff":
        speedup_results = speedup_results.sort_values(by="time_diff", ascending=False)
    else:
        speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

    # Remove rows where speedup or slowdown is less than 0.01
    small_change_count = speedup_results.loc[lambda x: x.speedup.abs() < 0.01].shape[0]
    speedup_results = speedup_results.loc[lambda x: x.speedup.abs() >= 0.01]

    speedup_results = speedup_results.reset_index(drop=True)
    num_points = speedup_results.shape[0]

    inflection = num_points
    for i in range(0, num_points - 1):
        if speedup_results["speedup"].iloc[i] >= 0 and speedup_results["speedup"].iloc[i + 1] < 0:
            inflection = i + 0.5

    pos = speedup_results.loc[lambda x: x.speedup >= 0]
    neg = speedup_results.loc[lambda x: x.speedup < 0]
    pos_speedup = pos["speedup"]
    neg_speedup = neg["speedup"]
    unit = speedup_results["Unit"].iloc[0]

    stats = {
        f"{new_method_name} vs {old_method_name}": ["Speedup", "Slowdown"],
        "Count": [pos_speedup.shape[0], neg_speedup.shape[0]],
        "Median": [pos_speedup.median(), neg_speedup.median()],
        "Max": [pos_speedup.max(), neg_speedup.min()],
        f"Time Difference ({unit})": [(pos["time_diff"]).sum(), (neg["time_diff"]).sum()],
        "Less than 1% change": [small_change_count, ""],
    }
    df_stats = pd.DataFrame(stats).fillna(0).set_index(f"{new_method_name} vs {old_method_name}")
    print(tabulate(df_stats, headers="keys", tablefmt="psql", floatfmt=".2f"))
    if only_stats:
        return
    df_stats.to_csv(output_dir / f"end_to_end_{new_method_name}_vs_{old_method_name}_stats.csv")

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
        def power_of_two_formatter(x, pos):
            if x == 0:
                return "1"
            return f"$2^{{{x:.1g}}}$"

        # Apply the custom formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(power_of_two_formatter))

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
        text = ""
        if plot_type == "log2_speedup":
            text = f"Max: $2^{{{max_pos:.2g}}}$"
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
            text = f"Min: $2^{{{min_neg:.2g}}}$"
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

    legend = plt.legend(frameon=True, framealpha=1, handlelength=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    if plot_type == "log2_speedup":
        ax.set_ylabel("Speedup")
    elif plot_type == "time_diff":
        ax.set_ylabel(f"Time Difference ({unit})")
    else:
        ax.set_ylabel("Relative Speedup")
    ax.set_xlabel(f"Models ({num_points} total)")
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
        top = pos_speedup.max() if clip_pos else None
        bottom = neg_speedup.min() if clip_neg else None
        ax.set_ylim(top=top, bottom=bottom)

    # save figure
    plt.savefig(
        output_dir / f"end_to_end_{new_method_name}_vs_{old_method_name}_{plot_type}.png",
        bbox_inches="tight",
        dpi=200,
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
    plot_type,
):
    speedup_results = get_speedup(joined_results, old_method_name, new_method_name)
    if not only_stats:
        # Save results to csv
        speedup_results.to_csv(
            output_dir / f"end_to_end_{new_method_name}_vs_{old_method_name}.csv", index=False
        )
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with performance results and summarize them into graphs."
    )

    parser.add_argument(
        "CSV_Input", type=str, help="Path to the input CSV file (generated by benchmark_models)."
    )
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")
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
        default="speedup",
        help="Data to plot. Speedup is the relative speedup and slowdown, log2_speedup is the log2 of speedup, and time_diff is the difference in time between methods. Default is speedup.",
    )

    args = parser.parse_args()

    csv_input = Path(args.CSV_Input)
    output_dir = Path(args.Output_Dir)
    old_method = args.old_method
    new_method = args.new_method
    only_stats = args.only_stats
    clip_pos = args.clip_positive_outliers
    clip_neg = args.clip_negative_outliers
    preset_comparisons = args.preset_comparisons
    plot_type = args.plot_type

    # Check if csv file exists
    if (not csv_input.exists()) or (not csv_input.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)
        sys.exit(-1)

    df = pd.read_csv(csv_input, header=0, index_col=False)

    # Merge results by conv_params and aggregate multiple runs
    df = merge_results(df, output_dir, only_stats)

    if not only_stats:
        df.to_csv(output_dir / "performance-results.csv", index=False)
        non_confident_convs = df.loc[df["Overall_Confident"] == False]["Model"]
        non_confident_convs.to_csv(output_dir / "non-confident-models.csv", index=False)

    methods = [col.replace("Median_", "") for col in df.columns if "Median" in col]

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
            comparisons = [("Torch", "ZeroCopy2d_Heuristic")]
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
