#!/usr/bin/env python3

import argparse
import sys
from math import ceil, floor, log10
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Saves a csv with results and produces an speedup graph
def compare_methods(joined_results, old_method_name, new_method_name):

    # Remove rows where an error occurred in either method
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    speedup_results = pd.DataFrame()
    speedup_results["conv_parameters"] = joined_results["conv_parameters"]

    # Compute speedup
    speedup_results["speedup"] = (
        joined_results["mean_time_" + old_method_name]
        - joined_results["mean_time_" + new_method_name]
    ) / joined_results["mean_time_" + new_method_name]
    speedup_results = speedup_results.sort_values(by="speedup", ascending=False)

    # Save resulst to csv
    speedup_results.to_csv(
        output_dir / f"conv2d_{old_method_name}_vs_{new_method_name}.csv", index=False
    )

    speedup = speedup_results["speedup"]
    num_points = speedup.shape[0]

    inflection = num_points
    for i in range(0, num_points - 1):
        if speedup.iloc[i] > 0 and speedup.iloc[i + 1] < 0:
            inflection = i + 0.5

    pos = speedup.loc[lambda x: x >= 0].reset_index(drop=True)
    neg = speedup.loc[lambda x: x < 0].reset_index(drop=True)

    fig, ax = plt.subplots()

    # barplot
    ax.bar(pos.index, pos, color="#2c7bb6")
    ax.bar(range(pos.shape[0], pos.shape[0] + neg.shape[0], 1), neg.values, color="#d7191c")

    # boxplot
    _, x_max = ax.get_xlim()
    ax.set_xlim((-x_max * 0.05, num_points + x_max * 0.05))
    ax.boxplot(
        [pos, neg],
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
        inflection, -y_total * 0.05 - y_total * 0.01, -y_total * 0.05 + y_total * 0.01, "#2c7bb6"
    )
    ax.text(
        (inflection / 2),
        -y_total * 0.08,
        f"{pos.shape[0]}",
        horizontalalignment="center",
        verticalalignment="center",
    )

    if neg.shape[0] != 0:
        ax.hlines(y_total * 0.05, inflection, num_points, "#d7191c")
        ax.vlines(
            inflection, y_total * 0.05 - y_total * 0.01, y_total * 0.05 + y_total * 0.01, "#d7191c"
        )
        ax.vlines(
            num_points, y_total * 0.05 - y_total * 0.01, y_total * 0.05 + +y_total * 0.01, "#d7191c"
        )
        ax.text(
            ((num_points + inflection) / 2),
            y_total * 0.08,
            f"{neg.shape[0]}",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # save figure
    plt.savefig(
        output_dir / f"conv2d_{old_method_name}_vs_{new_method_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with results and summarize them into graphs."
    )

    parser.add_argument("CSV_Results", type=str, help="Path to the output CSV file.")
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Maximum difference tolerance between results. Default is 0.125.",
        default=0.125,
    )

    args = parser.parse_args()

    csv_results = Path(args.CSV_Results)
    output_dir = Path(args.Output_Dir)
    tolerance = args.tolerance

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)
        sys.exit(-1)

    df = pd.read_csv(csv_results, header=0, index_col=False)
    df["conv_parameters"] = df["conv_parameters"].str.replace("/", " ")
    df["tolerance"] = tolerance

    # Save all conv parameters that had incorrect results
    incorrect_df = df.loc[(df["max_diff"] > tolerance) & (df["error_occurred"] != True)]
    incorrect_df = incorrect_df.drop(columns=["error_message", "error_occurred"])
    incorrect_df = incorrect_df.sort_values(["conv_type", "max_diff"], ascending=False)
    incorrect_df.to_csv(output_dir / f"incorrect-convolutions.csv", index=False)

    # Separate df by 'conv_type'
    groups = df.groupby(by=["conv_type"])
    summary_diffs = []
    summary_methods = []
    summary_incorrect_results = []
    summary_correct_results = []
    for name, group in groups:
        name = name[0]
        summary_methods.append(name)

        summary_diffs.append(group["max_diff"].max())

        incorrect_group = group.loc[(group["max_diff"] > tolerance) & (group["error_occurred"] != True)]
        incorret_convs = len(incorrect_group)
        summary_incorrect_results.append(incorret_convs)

        total_convs = len(group.loc[group["error_occurred"] != True])
        summary_correct_results.append(total_convs - incorret_convs)

    # Construct summary dataframe and save it
    summary = pd.DataFrame(
        {
            "Method": summary_methods,
            "Maximum Difference": summary_diffs,
            "Correct results": summary_correct_results,
            "Incorrect results": summary_incorrect_results,
        }
    )
    summary["Tolerance"] = tolerance
    summary = summary.sort_values(by="Method", ascending=False)
    summary.to_csv(output_dir / f"correctness-summary.csv", index=False)
