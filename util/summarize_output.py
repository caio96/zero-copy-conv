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
    # Create a new DataFrame with the split columns and remove the name column
    df = df[df.columns.tolist()[1:]]
    # Group by 'conv_type'
    groups = df.groupby(by=["conv_type"])
    # Convert groupby object to dictionary of conv_type to dataframe
    df_dict = {}
    for name, group in groups:
        df_dict[str(name[0])] = group.sort_values(by=["conv_type"]).reset_index(drop=True)
        # TODO: avoid deleting other columns
        df_dict[str(name[0])] = (
            df_dict[str(name[0])]
            .groupby(by="conv_parameters")
            .agg(time_mean=("real_time", "mean"), time_std=("real_time", "std"))
            .reset_index()
        )

    # distribution = (
    #     df_dict["Conv2D_Im2col"]["time_mean"] / df_dict["Conv2D_Yaconv_v2"]["time_mean"]
    # ).sort_values(ascending=False)
    # distribution = distribution - 1

    distribution = (
        (df_dict["Conv2D_Im2col"]["time_mean"] - df_dict["Conv2D_Yaconv_v2"]["time_mean"])
        / df_dict["Conv2D_Yaconv_v2"]["time_mean"]
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
    ax.set_ylim((-1.5, 4.0))
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

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
    plt.savefig(output_dir / "conv2d_im2col_vs_yaconv_v2.png", bbox_inches="tight", dpi=300)
    plt.close()
