#!/usr/bin/env python3

import argparse
import sys
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with results and summarize them into graphs."
    )

    parser.add_argument("CSV_Results", type=str, help="Path to the output CSV file.")
    parser.add_argument("Output_Dir", type=str, help="Path to directory to store outputs.")

    args = parser.parse_args()

    csv_results = Path(args.CSV_Results)
    output_dir = Path(args.Output_Dir)

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # Check if output dir exists
    if (not output_dir.exists()) or (not output_dir.is_dir()):
        print("Output directory not found.", file=sys.stderr)
        sys.exit(-1)

    df = pd.read_csv(csv_results, header=0, index_col=False)

    # Split the 'name' column into 'conv_type' and 'conv_parameters'
    df[["conv_type", "conv_parameters"]] = df["name"].str.split("/", n=1, expand=True)
    df = df.drop(columns=["name"])

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

    old_method_name = "Im2col"
    new_method_name = "Yaconv_v2"

    old_method = df_dict.pop("Conv2D_" + old_method_name)
    new_method = df_dict.pop("Conv2D_" + new_method_name)

    joined_results = pd.merge(
        old_method.loc[:, ["conv_parameters", "mean_time", "error_occurred"]],
        new_method.loc[:, ["conv_parameters", "mean_time", "error_occurred"]],
        how="inner",
        on="conv_parameters",
        suffixes=("_" + old_method_name, "_" + new_method_name),
    )
    joined_results = joined_results.loc[
        (joined_results["error_occurred_" + old_method_name] == False)
        & (joined_results["error_occurred_" + new_method_name] == False)
    ]

    distribution = (
        (
            joined_results["mean_time_" + old_method_name]
            - joined_results["mean_time_" + new_method_name]
        )
        / joined_results["mean_time_" + new_method_name]
    ).sort_values(ascending=False)

    num_points = distribution.shape[0]

    inflection = num_points
    for i in range(0, num_points - 1):
        if distribution.iloc[i] > 0 and distribution.iloc[i + 1] < 0:
            inflection = i + 0.5

    pos = distribution.loc[lambda x: x >= 0].reset_index(drop=True)
    neg = distribution.loc[lambda x: x < 0].reset_index(drop=True)

    fig, ax = plt.subplots()
    # fig.set_size_inches((18, 12))

    # barplot
    ax.bar(pos.index, pos)
    ax.bar(range(pos.shape[0], pos.shape[0] + neg.shape[0], 1), neg.values, color="r")

    ax.set_ylabel("Speedup/Slowdown")
    # ax.set_ylim((-25, 50.0))
    # ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    # pos boxplot
    ax.boxplot([pos, neg], showfliers=False, positions=[-25, num_points + 25], widths=20)

    ax.set_xlabel("Convolution")
    ax.set_xlim((-50.0, num_points + 50))
    ax.set_xticks([0, inflection, num_points], [0, int(inflection), num_points])

    # plt.vlines(inflection, -1, 2.0, color="r", linewidth=1.4, alpha=.4)

    ax.vlines(1, -0.65, -0.35, "r")
    ax.hlines(-0.5, 1, inflection, "r")
    ax.vlines(inflection, -0.65, -0.35, "r")
    ax.text((inflection / 2), -0.75, f"{pos.shape[0]} Convolutions", horizontalalignment="center")

    if neg.shape[0] != 0:
        ax.vlines(inflection, 0.65, 0.35, "r")
        ax.hlines(0.5, inflection, num_points, "r")
        ax.vlines(num_points, 0.65, 0.35, "r")
        ax.text(
            ((num_points + inflection) / 2),
            0.75,
            f"{neg.shape[0]} Convolutions",
            horizontalalignment="center",
        )

    # save figure
    plt.savefig(
        output_dir / f"conv2d_{old_method_name}_vs_{new_method_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
