#!/usr/bin/env python3

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from filter_csv import exclude_from_df, include_only_in_df, split_parameters


def get_data(df: pd.DataFrame):
    # mute performance warnings
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    # Separate the parameters
    df = split_parameters(df)
    df = df.drop(columns=["conv_parameters"])

    # Remove redundant padding columns
    df = df.drop(columns=["padding bottom", "padding right"])
    df.rename(
        columns={"padding top": "padding height", "padding left": "padding width"}, inplace=True
    )

    # Add output height and width
    df["output height"] = np.floor(
        (
            df["image height"]
            + 2 * df["padding height"]
            - df["dilation height"] * (df["filter height"] - 1)
            - 1
        )
        / df["stride height"]
        + 1
    )
    df["output width"] = np.floor(
        (
            df["image width"]
            + 2 * df["padding width"]
            - df["dilation width"] * (df["filter width"] - 1)
            - 1
        )
        / df["stride width"]
        + 1
    )

    # Transposed convolution is not supported
    df = exclude_from_df(df, ["transposed"])
    df = df.drop(columns=["is transposed"])

    # Add matrix dimensions used by ZeroCopy2D
    # They are independent of the convolution type
    df["m dim"] = df["output height"]
    df["k dim"] = df["filter width"] * df["image channel"] / df["groups"]
    df["n dim"] = df["output channel"] / df["groups"]
    df["A size"] = df["m dim"] * df["k dim"]
    df["B size"] = df["k dim"] * df["n dim"]
    df["C size"] = df["m dim"] * df["n dim"]

    return df


def get_X_y(df: pd.DataFrame, mode: str, speedup_threshold: float = 0.0):

    # Filter by convolution type
    if mode == "normal":
        df = exclude_from_df(df, ["dilated", "grouped"])
        df = df.drop(columns=["dilation height", "dilation width", "groups"])
    elif mode == "extended":
        df = include_only_in_df(df, ["dilated", "grouped"])
    else:
        raise ValueError(f"Invalid mode {mode}.")

    # Target column
    y = (df["speedup"] > speedup_threshold).astype(int)
    df = df.drop(columns=["speedup"])

    # Features
    X = pd.DataFrame()
    print(df.columns.values.tolist())

    # Only add binary comparison features
    for feature1, feature2 in itertools.combinations(df.columns.values.tolist(), 2):
        ignore_features = ["has bias"]
        if feature1 in ignore_features or feature2 in ignore_features:
            continue

        # Add binary comparison features
        X[f"{feature1} greater than {feature2}"] = (df[feature1] > df[feature2]).astype(int)
        X[f"{feature1} equals {feature2}"] = (df[feature1] == df[feature2]).astype(int)

    X["has bias"] = df["has bias"]
    X["is strided"] = ((df["stride height"] != 1) | (df["stride width"] != 1)).astype(int)
    X["is pointwise"] = ((df["filter height"] == 1) & (df["filter width"] == 1)).astype(int)

    if mode == "extended":
        X["is dilated"] = ((df["dilation height"] != 1) | (df["dilation width"] != 1)).astype(int)
        X["is grouped"] = (df["groups"] != 1).astype(int)
        X["is depthwise"] = (df["image channel"] == df["groups"]).astype(int)

    return X, y


def run_decision_tree(X_train, y_train, X_test, y_test, depth):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("DecisionTreeClassifier report:")
    print(classification_report(y_test, y_pred))

    rules = export_text(model, feature_names=list(X.columns))
    print("DecisionTreeClassifier rules:")
    print(rules)


def run_xgboost(X_train, y_train, X_test, y_test, depth):
    model = xgb.XGBClassifier(max_depth=depth, n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("XGBoost report:")
    print(classification_report(y_test, y_pred))

    feature_importance = model.get_booster().get_score(importance_type="weight")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("XGBoost most important features:")
    for feature, score in sorted_importance[:10]:
        print(f"- {feature}: {score}")
    print("")

    for idx, rule in enumerate(model.get_booster().get_dump(with_stats=False)):
        print(f"XGBoost rules from Tree {idx}:")
        print(rule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Learn a heuristic to predict the preferred method based on a speedup CSV."
    )
    parser.add_argument(
        "CSV_Results",
        type=str,
        help='Path to the output CSV file, it should contain a "conv_parameters" and "speedup" columns.',
    )
    parser.add_argument(
        "--zc-mode",
        type=str,
        help="Select the zero-copy mode to use as data. Extended is used for grouped and dilated convolution, normal is used otherwise. Default is normal.",
        choices=[
            "normal",
            "extended",
        ],
        default="normal",
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        help="Speedup threshold to consider a method as preferred. Default is 0.0.",
        default=0.0,
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        help="Maximum depth of the decision tree. Default is 1.",
        default=1,
    )
    parser.add_argument(
        "--split-train-test",
        action="store_true",
        help="Split input into train and test sets.",
    )
    parser.add_argument(
        "--xgboost",
        action="store_true",
        help="Use XGBoost instead of DecisionTreeClassifier.",
    )

    args = parser.parse_args()
    csv_results = Path(args.CSV_Results)
    mode = args.zc_mode
    speedup_threshold = args.speedup_threshold
    depth = args.tree_depth
    use_xgb = args.xgboost
    split_train_test = args.split_train_test

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    df = pd.read_csv(csv_results, header=0, index_col=False)

    df = get_data(df)
    X, y = get_X_y(df, mode, speedup_threshold)

    if split_train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    if not use_xgb:
        run_decision_tree(X_train, y_train, X_test, y_test, depth)
    else:
        run_xgboost(X_train, y_train, X_test, y_test, depth)
