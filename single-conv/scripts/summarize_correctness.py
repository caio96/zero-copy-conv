#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse CSV with results and summarize them into graphs."
    )

    parser.add_argument("CSV_Results", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "Output_Dir", type=str, help="Path to directory to store outputs."
    )
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
    df["tolerance"] = tolerance

    # Save all conv parameters that had incorrect results
    incorrect_df = df.loc[(df["max_diff"] > tolerance) & (df["error_occurred"] != True)]
    incorrect_df = incorrect_df.drop(columns=["error_message", "error_occurred"])
    incorrect_df = incorrect_df.sort_values(["conv_type", "max_diff"], ascending=False)
    incorrect_df.to_csv(output_dir / "incorrect-convolutions.csv", index=False)

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

        incorrect_group = group.loc[
            (group["max_diff"] > tolerance) & (group["error_occurred"] != True)
        ]
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
    summary.to_csv(output_dir / "correctness-summary.csv", index=False)
