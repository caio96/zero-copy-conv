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
        name = name[0].replace("Conv2D_", "")
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

    # Save joined results
    joined_results.to_csv(output_dir / f"results.csv", index=False)

    # These tuples represent the comparisons that will be made into graphs
    comparisons = [("Im2col", "Yaconv"), ("Im2col", "Yaconv_v2"), ("Yaconv", "Yaconv_v2")]

    for old_method, new_method in comparisons:
        compare_methods(joined_results, old_method, new_method)
