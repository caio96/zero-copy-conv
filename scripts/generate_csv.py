#!/usr/bin/env python3

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


## split pickle key
def split_key(key):
    inp, out, kernel, bias = key.split("_")

    inp = list(map(int, inp.split(",")))
    out = list(map(int, out.split(",")))

    kernel = kernel.split("-")
    kernel_size = list(map(int, kernel[0].split(",")))
    padding = list(map(int, kernel[1].split(",")))
    stride = list(map(int, kernel[2].split(",")))
    dilation = list(map(int, kernel[3].split(",")))
    groups = int(kernel[4])
    transposed = bool(1 if kernel[5] == "True" else 0)

    bias = bool(key[3])

    return [
        key,
        *inp,
        *out,
        *kernel_size,
        *padding,
        *stride,
        *dilation,
        groups,
        transposed,
        out[1] if bias else 0,
    ]


def get_filter_expression(filtering):
    expression = []

    if not filtering["elementwise"]:
        expression.append("(df['Filter height'] != 1) & (df['Filter width'] != 1)")

    if not filtering["grouped"]:
        expression.append("(df['Grouped'] == 1)")

    if not filtering["dilated"]:
        expression.append("(df['Dilation height'] == 1) & (df['Dilation width'] == 1)")

    if filtering["square_filters_only"]:
        expression.append("(df['Filter height'] == df['Filter width'])")

    if filtering["square_images_only"]:
        expression.append("(df['Input height'] == df['Input width'])")

    if filtering["symmetrical_padding_only"]:
        expression.append(
            "(df['Padding top'] == df['Padding bottom']) & (df['Padding left'] == df['Padding right'])"
        )

    # Check if expression is empty
    if not expression:
        return "df"

    # Construct the final filtering string
    return "df.loc[" + " & ".join(expression) + ", :].reset_index()"


def pickle_to_df(pickle_data):
    conv_info = [
        "key",
        "Image batch",
        "Image channel",
        "Image height",
        "Image width",
        "Output batch",
        "Output depth",
        "Output height",
        "Output width",
        "Filter height",
        "Filter width",
        "Padding top",
        "Padding bottom",
        "Padding left",
        "Padding right",
        "Stride height",
        "Stride width",
        "Dilation height",
        "Dilation width",
        "Grouped",
        "Transposed",
        "Bias",
        "model",
        "model attr",
    ]

    table = []
    for key in pickle_data.keys():
        path = pickle_data[key]
        models = [x.partition(".")[0] for x in path]
        model_attr = [x.partition(".")[2] for x in path]
        table.append([*split_key(key), models, model_attr])

    # Create pandas dataframe
    df = pd.DataFrame(table, columns=conv_info)
    df = df.drop(columns=["key", "Output batch", "Transposed", "Bias", "model", "model attr"])

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate CSV from pickle file with filtering options."
    )

    parser.add_argument("Pickle_File", type=str, help="Path to the pickle file.")
    parser.add_argument("CSV_Output", type=str, help="Path to the output CSV file.")

    parser.add_argument(
        "--elementwise",
        action="store_true",
        help="Enable elementwise convolution.",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Enable grouped convolution.",
    )
    parser.add_argument(
        "--dilated",
        action="store_true",
        help="Enable dilated convolution.",
    )
    parser.add_argument(
        "--square_filters_only",
        action="store_true",
        help="Enable square filters only.",
    )
    parser.add_argument(
        "--square_images_only",
        action="store_true",
        help="Enable square images only.",
    )
    parser.add_argument(
        "--symmetrical_padding_only",
        action="store_true",
        help="Enable symmetrical padding only.",
    )

    args = parser.parse_args()

    pickle_file = Path(args.Pickle_File)
    output_csv = Path(args.CSV_Output)

    filtering = {
        "elementwise": args.elementwise,
        "grouped": args.grouped,
        "dilated": args.dilated,
        "square_filters_only": args.square_filters_only,
        "square_images_only": args.square_images_only,
        "symmetrical_padding_only": args.symmetrical_padding_only,
    }

    # Check if pickle file exists
    if (not pickle_file.exists()) or (not pickle_file.is_file()):
        print("Pickle file not found.", file=sys.stderr)
        sys.exit(-1)

    # Construct the filtering expression
    expression = get_filter_expression(filtering)
    print(f'Final Filtering Expression: "{expression}"')

    # Load pickle file
    with open(pickle_file, "rb") as handle:
        pickle_data = pickle.load(handle)

    # Process pickle data to a pandas dataframe
    df = pickle_to_df(pickle_data)

    # Filter pandas dataframe
    filtered_df = eval(expression)

    # Save df to csv
    filtered_df.loc[:, "Image batch":"Grouped"].to_csv(output_csv, index_label="ID")
