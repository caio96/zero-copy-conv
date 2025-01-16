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
from sklearn.metrics import classification_report, f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

from filter_csv import exclude_from_df, include_only_in_df, split_parameters


def get_data(df: pd.DataFrame):
    # Separate the parameters
    df = split_parameters(df)

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
def speedup_weights(y, speedup):
    df = pd.DataFrame()
    df["y"] = y
    df["speedup"] = speedup

    # Get raw weights clipped at the 99th percentile
    df["weight"] = np.abs(speedup)
    df["weight"] = np.clip(df["weight"], 0, df["weight"].quantile(0.99))

    # Normalize between 0 and 1
    df["weight"] = (df["weight"] - np.min(df["weight"])) / (
        np.max(df["weight"]) - np.min(df["weight"])
    )

    # Separate weights by class
    weights_pos = df.loc[df["y"] == 1, "weight"]
    weights_neg = df.loc[df["y"] == 0, "weight"]

    # Balance weights per class
    weights_pos = weights_pos / weights_pos.sum()
    weights_neg = weights_neg / weights_neg.sum()

    # Assign normalized weights back to the dataset
    df.loc[df["y"] == 1, "weight"] = weights_pos
    df.loc[df["y"] == 0, "weight"] = weights_neg

    # Normalize between 0 and 1
    df["weight"] = (df["weight"] - df["weight"].min()) / (df["weight"].max() - df["weight"].min())

    return df["weight"]


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


def get_X_y(df: pd.DataFrame, mode: str, occurrences_csv: Path = None, speedup_threshold: float = 0.0):

    if occurrences_csv:
        occurrences_df = pd.read_csv(occurrences_csv, header=0, index_col=False)
        df = df.merge(occurrences_df, on="conv_parameters", how="left")

    df.drop(columns=["conv_parameters"], inplace=True)

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
    y_weights = speedup_weights(y, df["speedup"])
    if occurrences_csv:
        y_weights = y_weights * df["occurrences"]
        df = df.drop(columns=["occurrences"])

    df = df.drop(columns=["speedup"])

    # Remove columns with the same values
    df = remove_invariant_features(df)

    # Features
    X = pd.DataFrame()

    # Add zeros and ones for the comparisons
    df["0"] = 0
    df["1"] = 1

    # Only add binary comparison features
    for feature1, feature2 in itertools.combinations(df.columns.values.tolist(), 2):
        ignore_features = ["has bias"]
        if feature1 in ignore_features or feature2 in ignore_features:
            continue

        # Add binary comparison features
        X[f"{feature1} greater than {feature2}"] = (df[feature1] > df[feature2]).astype(int)
        X[f"{feature1} equals {feature2}"] = (df[feature1] == df[feature2]).astype(int)

    X["has bias"] = df["has bias"]

    # Remove columns with the same values that may have been added
    X = remove_invariant_features(X)

    return X, y, y_weights


def run_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    max_depth=None,
    max_leaf_nodes=None,
    w_train=None,
    weight_balance=1.0,
    search=False,
):
    if weight_balance == 1.0 or weight_balance == -1.0:
        class_weights = {0: 1.0, 1: 1.0}
    elif weight_balance > 1.0:
        class_weights = {0: 1.0, 1: weight_balance}
    elif weight_balance < 1.0:
        class_weights = {0: -weight_balance, 1: 1.0}
    else:
        raise ValueError(f"Invalid weight balance {weight_balance}.")

    if not search:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
            class_weight=class_weights,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_test)

        print("DecisionTreeClassifier report:")
        print(classification_report(y_test, y_pred))

        return model

    # Custom scorer for precision for class 1
    # scorer = make_scorer(precision_score, pos_label=1)
    scorer = make_scorer(f1_score, pos_label=1)

    # Define parameter grid
    param_grid = {
        "criterion": ["gini"],  # Splitting criteria
        "max_depth": [1, 2, None],  # Maximum depth of the tree
        "max_leaf_nodes": [4, 5, 6],  # Maximum depth of the tree
        "max_features": [None, "sqrt", "log2"],  # Number of features to consider for best split
        "class_weight": [
            None,
            {0: 1.0, 1: 2.0},
            {0: 1.0, 1: 1.5},
            {0: 2.0, 1: 1.0},
            {0: 1.5, 1: 1.0},
        ],
    }

    # Perform grid search
    grid_search = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        scoring=scorer,
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return best_model


def prune_duplicate_leaves(mdl):

    def is_leaf(inner_tree, index):
        # Check whether node is leaf node
        return (
            inner_tree.children_left[index] == TREE_LEAF
            and inner_tree.children_right[index] == TREE_LEAF
        )

    def prune_index(inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not is_leaf(inner_tree, inner_tree.children_left[index]):
            prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not is_leaf(inner_tree, inner_tree.children_right[index]):
            prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (
            is_leaf(inner_tree, inner_tree.children_left[index])
            and is_leaf(inner_tree, inner_tree.children_right[index])
            and (decisions[index] == decisions[inner_tree.children_left[index]])
            and (decisions[index] == decisions[inner_tree.children_right[index]])
        ):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            inner_tree.feature[index] = TREE_UNDEFINED
            ##print("Pruned {}".format(index))

    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)


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
        "--occurrences-csv",
        type=str,
        help="Path to csv with 'conv_parameters' and 'occurences' columns. If passed, it is used to weight samples."
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
    parser.add_argument(
        "--weight-balance",
        type=float,
        default=1.0,
        help="Weight used to balance the classes. Default is 1.0 (equal weights), which is the same as -1.0. If greater than 1.0, class 1 is weighted more. If less than -1.0, class 0 is weighted more.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Use grid search to find the best hyperparameters.",
    )

    args = parser.parse_args()
    csv_results = Path(args.CSV_Results)
    mode = args.zc_mode
    speedup_threshold = args.speedup_threshold
    max_depth = args.max_depth
    max_features = args.max_features
    max_leaf_nodes = args.max_leaf_nodes
    split_train_test = args.split_train_test
    weight_balance = args.weight_balance
    search = args.search
    occurrences_csv = args.occurrences_csv

    # Check if csv file exists
    if (not csv_results.exists()) or (not csv_results.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    if occurrences_csv:
        occurrences_csv = Path(occurrences_csv)
        if (not occurrences_csv.exists()) or (not occurrences_csv.is_file()):
            print("CSV with occurrences not found.", file=sys.stderr)
            sys.exit(-1)

    # Mute pandas warnings
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df = pd.read_csv(csv_results, header=0, index_col=False)

    df = get_data(df)
    X, y, y_weights = get_X_y(df, mode, occurrences_csv, speedup_threshold)

    X = reduce_dimensionality(X, y, max_features, max_depth, y_weights)

    if split_train_test:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, y_weights, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train, w_train = X, y, y_weights
        X_test, y_test, w_test = X, y, y_weights

    model = run_decision_tree(
        X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, w_train, weight_balance, search
    )

    prune_duplicate_leaves(model)

    rules = export_text(model, feature_names=list(X.columns))
    print("DecisionTreeClassifier rules:")
    print(rules)
