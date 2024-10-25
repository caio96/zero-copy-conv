#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from math import log10

import pandas as pd
import matplotlib.pyplot as plt

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
    groups = df.sort_values(by=["conv_type"]).groupby(by=["conv_type"])
    # Convert groupby object to dictionary of conv_type to dataframe
    df_dict = {}
    for name, group in groups:
        df_dict[str(name[0])] = group
