#!/usr/bin/env python3

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from filter_csv import exclude_from_df, include_only_in_df, split_parameters


def get_data(df: pd.DataFrame):
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


# Define a scaling function for weights
def speedup_weights(speedup):
    scaled = np.abs(speedup)
    scaled = np.clip(scaled, 0, 5)
    scaled = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))
    return scaled


def remove_invariant_features(X: pd.DataFrame):
    # Save labels
    columns = X.columns

    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)

    # Add labels back
    selected_features_mask = selector.get_support()
    selected_feature_names = columns[selected_features_mask]
    X = pd.DataFrame(X, columns=selected_feature_names)

    return X


def reduce_dimensionality(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = None,
    max_depth: int = None,
    y_weights: np.array = None,
):
    # Save labels
    columns = X.columns

    # Reduce dimensionality using most important features from tree ensemble
    clf = ExtraTreesClassifier(max_depth=max_depth, n_estimators=100)
    clf = clf.fit(X, y, sample_weight=y_weights)
    selector = SelectFromModel(clf, prefit=True, max_features=max_features)
    selector.feature_names_in_ = columns
    X = selector.transform(X)

    # Add labels back
    selected_features_mask = selector.get_support()
    selected_feature_names = columns[selected_features_mask]
    X = pd.DataFrame(X, columns=selected_feature_names)

    return X


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
    y_weights = speedup_weights(df["speedup"])
    df = df.drop(columns=["speedup"])

    # Remove columns with the same values
    df = remove_invariant_features(df)

    # Features
    X = pd.DataFrame()

    # Only add binary comparison features
    for feature1, feature2 in itertools.combinations(df.columns.values.tolist(), 2):
        ignore_features = ["has bias"]
        if feature1 in ignore_features or feature2 in ignore_features:
            continue

        # Add binary comparison features
        X[f"{feature1} greater than {feature2}"] = (df[feature1] > df[feature2]).astype(int)
        X[f"{feature1} equals {feature2}"] = (df[feature1] == df[feature2]).astype(int)

    X["has bias"] = df["has bias"]
    X["has stride height"] = (df["stride height"] != 1).astype(int)
    X["has stride width"] = (df["stride width"] != 1).astype(int)
    X["is strided"] = ((df["stride height"] != 1) | (df["stride width"] != 1)).astype(int)
    X["is pointwise in height"] = (df["filter height"] == 1).astype(int)
    X["is pointwise in width"] = (df["filter width"] == 1).astype(int)
    X["is pointwise"] = ((df["filter height"] == 1) & (df["filter width"] == 1)).astype(int)
    X["has padding height"] = (df["padding height"] > 0).astype(int)
    X["has padding width"] = (df["padding width"] > 0).astype(int)
    X["has padding"] = ((df["padding height"] > 0) | (df["padding width"] > 0)).astype(int)
    X["has overlap in height"] = (df["filter height"] > df["stride height"]).astype(int)
    X["has overlap in width"] = (df["filter width"] > df["stride width"]).astype(int)
    X["has overlap"] = (
        (df["filter height"] > df["stride height"]) | (df["filter width"] > df["stride width"])
    ).astype(int)

    if mode == "extended":
        X["is dilated"] = ((df["dilation height"] != 1) | (df["dilation width"] != 1)).astype(int)
        X["is grouped"] = (df["groups"] != 1).astype(int)

    return X, y, y_weights


def run_decision_tree(
    X_train, y_train, X_test, y_test, max_depth=None, max_leaf_nodes=None, w_train=None
):
    model = DecisionTreeClassifier(
        max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)

    print("DecisionTreeClassifier report:")
    print(classification_report(y_test, y_pred))

    rules = export_text(model, feature_names=list(X.columns))
    print("DecisionTreeClassifier rules:")
    print(rules)


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
        "--max-depth",
        type=int,
        help="Maximum depth of the decision tree. Default is no limit.",
        default=None,
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        help="Maximum leaf nodes of the decision tree. Default is no limit.",
        default=None,
    )
    parser.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of features to use. By setting a limit, only the n most important features (decided by a tree ensemble during dimensionality reduction) are considered. Default is no limit.",
        default=None,
    )
    parser.add_argument(
        "--split-train-test",
        action="store_true",
        help="Split input into train and test sets.",
    )

    args = parser.parse_args()
    csv_results = Path(args.CSV_Results)
    mode = args.zc_mode
    speedup_threshold = args.speedup_threshold
    max_depth = args.max_depth
    max_features = args.max_features
    max_leaf_nodes = args.max_leaf_nodes
    split_train_test = args.split_train_test

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    # mute pandas warnings
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df = pd.read_csv(csv_results, header=0, index_col=False)

    df = get_data(df)
    X, y, y_weights = get_X_y(df, mode, speedup_threshold)

    X = reduce_dimensionality(X, y, max_features, max_depth, y_weights)

    if split_train_test:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, y_weights, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train, w_train = X, y, y_weights
        X_test, y_test, w_test = X, y, y_weights

    run_decision_tree(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, w_train)
