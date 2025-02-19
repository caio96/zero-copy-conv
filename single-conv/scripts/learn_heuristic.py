#!/usr/bin/env python3

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from filter_csv import exclude_from_df, include_only_in_df, split_parameters, get_categories
from joblib import parallel_backend
from sklearn import set_config
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from matplotlib import pyplot as plt
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from summarize_performance import plot_speedup
from tabulate import tabulate


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
    df["dim m"] = df["output height"]
    df["dim k"] = df["filter width"] * df["image channel"] / df["groups"]
    df["dim n"] = df["output channel"] / df["groups"]
    df["A size"] = df["dim m"] * df["dim k"]
    df["B size"] = df["dim k"] * df["dim n"]
    df["C size"] = df["dim m"] * df["dim n"]

    # ZeroCopy2d_Ext
    df["image channel / groups"] = df["image channel"] / df["groups"]

    return df


# Define a scaling function for weights
def get_speedup_weights(y, speedup=None, occurrences=None):
    if speedup is None and occurrences is None:
        return None

    df = pd.DataFrame()
    df["y"] = y

    df["weight"] = 1.0

    # Get the absolute performance change
    if speedup is not None:
        # Clip outliers
        speedup = np.clip(speedup, speedup.quantile(0.01), speedup.quantile(0.99))
        # Get the absolute performance change
        speedup = np.abs(speedup)
        # Normalize between 0 and 1
        speedup = (speedup - speedup.min()) / (speedup.max() - speedup.min())
        df["weight"] = df["weight"] * speedup

    if occurrences is not None:
        df["weight"] = df["weight"] * occurrences

    # Separate weights by class
    weights_pos = df.loc[df["y"] == 1, "weight"]
    weights_neg = df.loc[df["y"] == 0, "weight"]

    # Balance weights per class
    weights_pos = weights_pos / weights_pos.sum()
    weights_neg = weights_neg / weights_neg.sum()

    # Assign normalized weights back to the dataset
    df.loc[df["y"] == 1, "weight"] = weights_pos
    df.loc[df["y"] == 0, "weight"] = weights_neg

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
    y_weights: np.array = None,
):
    # Save labels
    columns = X.columns

    # Reduce dimensionality using most important features from tree ensemble
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X, y, sample_weight=y_weights)
    selector = SelectFromModel(clf, prefit=True)
    selector.feature_names_in_ = columns
    X = selector.transform(X)

    # Add labels back
    selected_features_mask = selector.get_support()
    selected_feature_names = columns[selected_features_mask]
    X = pd.DataFrame(X, columns=selected_feature_names)

    return X


def separate_features(df: pd.DataFrame, speedup_threshold: float, use_speedup: bool, use_occurrences: bool, use_log2: bool):

    # Target column
    y = (df["speedup"] > speedup_threshold).astype(int)

    # Target weights
    speedup = None
    occurrences = None
    if use_speedup:
        if use_log2:
            speedup = df["log2_speedup"]
        else:
            speedup = df["speedup"]
    if use_occurrences:
        if use_log2:
            occurrences = np.log2(df["occurrences"])+1
        else:
            occurrences = df["occurrences"]

    y_weights = get_speedup_weights(y, speedup, occurrences)

    # Remove columns not used as features
    df = df.drop(columns=["conv_parameters", "speedup", "occurrences", "log2_speedup"])

    # Remove columns with the same values
    df = remove_invariant_features(df)

    # Add zeros and ones for the comparisons
    df["0"] = 0
    df["1"] = 1

    # Features
    X = pd.DataFrame()

    # Only add binary comparison features
    for feature1, feature2 in itertools.combinations(
        sorted(df.columns.values.tolist(), reverse=True), 2
    ):
        ignore_features = ["has bias"]
        if feature1 in ignore_features or feature2 in ignore_features:
            continue

        # Add binary comparison features
        X[f"{feature1} greater than {feature2}"] = (df[feature1] > df[feature2]).astype(int)
        X[f"{feature1} equals {feature2}"] = (df[feature1] == df[feature2]).astype(int)

    if "has bias" in df.columns:
        X["has bias"] = df["has bias"]

    # Remove columns with the same values that may have been added
    X = remove_invariant_features(X)

    return X, y, y_weights


def print_results(y_test, y_pred, w_test):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, sample_weight=w_test)
    _, _, _, support = precision_recall_fscore_support(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)

    print("\nWeighted classification report:")
    summary = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": support,
        "accuracy": [accuracy],
    }
    print(
        tabulate(
            summary,
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
            showindex=["class 0", "class 1"],
        )
    )

    print("\nConfusion matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    print(tabulate(conf_matrix_df, headers="keys", tablefmt="psql", floatfmt=".2f"))

    print("\nWeighted confusion matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    print(tabulate(conf_matrix_df, headers="keys", tablefmt="psql", floatfmt=".2f"))


def run_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    w_train=None,
    w_test=None,
    search_scoring="f1"
):
    set_config(enable_metadata_routing=True)

    # Custom scorer for class 1
    if search_scoring == "precision":
        scorer = make_scorer(precision_score).set_score_request(sample_weight=True)
    elif search_scoring == "f1":
        scorer = make_scorer(f1_score).set_score_request(sample_weight=True)
    elif search_scoring == "recall":
        scorer = make_scorer(recall_score).set_score_request(sample_weight=True)
    elif search_scoring == "accuracy":
        scorer = make_scorer(accuracy_score).set_score_request(sample_weight=True)
    else:
        print("Invalid scoring function.", file=sys.stderr)
        sys.exit(-1)

    # Define parameter grid
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],  # Splitting criteria
        "max_leaf_nodes": [2, 3, 4, 5, 6],  # Maximum number of leaf nodes
        "max_features": [None, "sqrt", "log2"],  # Number of features to consider for best split
        "class_weight": [
            None,
            {0: 1.0, 1: 3.0},
            {0: 3.0, 1: 1.0},
            {0: 1.0, 1: 2.0},
            {0: 2.0, 1: 1.0},
            {0: 1.0, 1: 1.5},
            {0: 1.5, 1: 1.0},
        ],
        "min_samples_leaf": [0.01, 0.05],
        "min_samples_split": [0.01, 0.05],
    }

    # Perform grid search
    grid_search = GridSearchCV(
        DecisionTreeClassifier().set_fit_request(sample_weight=True),
        param_grid,
        scoring=scorer,
        n_jobs=-1,
        cv=5,
        verbose=1,
    )
    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UndefinedMetricWarning)
            grid_search.fit(X_train, y_train, sample_weight=w_train)

    # Best parameters and model
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    print_results(y_test, y_pred, w_test)

    return best_model, y_pred


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
            # print("Pruned {}".format(index))

    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Learn a heuristic to predict the preferred method based on a speedup CSV."
    )
    parser.add_argument(
        "Speedup_CSV",
        type=str,
        help='Path to the output CSV file, it should contain a "conv_parameters" and "speedup" columns (generated by summarize_performance).',
    )
    parser.add_argument(
        "Output_Dir",
        type=str,
        help='Path to the output dir to save a plot of the decision tree.',
    )
    parser.add_argument(
        "--include-only-conv-types",
        nargs="+",
        type=str,
        help="Only include the specified convolution types",
        choices=get_categories(),
    )
    parser.add_argument(
        "--exclude-conv-types",
        nargs="+",
        type=str,
        help="List of convolution types to exclude",
        choices=get_categories(),
    )
    parser.add_argument(
        "--split-train-test",
        action="store_true",
        help="Split input into train and test sets. Otherwise cross validation is used and the model is evaluated on the whole dataset.",
    )
    parser.add_argument(
        "--speedup-threshold",
        type=float,
        help="Speedup threshold to consider a method as preferred. Default is 0.0.",
        default=0.0,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="f1",
        choices=["precision", "f1", "recall", "accuracy"],
        help="Scoring function used in the grid search. Default is f1.",
    )
    parser.add_argument(
        "--use-speedup",
        action="store_true",
        help="Use the speedup of each convolution to weight the samples. Can be combined with --use-occurrences.",
    )
    parser.add_argument(
        "--use-occurrences",
        action="store_true",
        help="Use the number of occurrences of each convolution to weight the samples. Can be combined with --use-speedup.",
    )
    parser.add_argument(
        "--use-log2",
        action="store_true",
        help="Use the log2 to scale weights.",
    )

    args = parser.parse_args()
    csv_input = Path(args.Speedup_CSV)
    plot_output_dir = Path(args.Output_Dir)
    include_only_conv_types = args.include_only_conv_types
    exclude_conv_types = args.exclude_conv_types
    split_train_test = args.split_train_test
    speedup_threshold = args.speedup_threshold
    scoring = args.scoring
    use_speedup = args.use_speedup
    use_occurrences = args.use_occurrences
    use_log2_weights = args.use_log2

    # Check if csv file exists
    if (not csv_input.exists()) or (not csv_input.is_file()):
        print("CSV with results not found.", file=sys.stderr)
        sys.exit(-1)

    if not Path(plot_output_dir).exists() or not Path(plot_output_dir).is_dir():
        print("Output directory for plot does not exist.", file=sys.stderr)
        sys.exit(-1)

    # Mute pandas warnings
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df = pd.read_csv(csv_input, header=0, index_col=False)

    # Filter convs if needed
    num_columns = len(df.columns)
    df = split_parameters(df)
    df = include_only_in_df(df, include_only_conv_types)
    df = exclude_from_df(df, exclude_conv_types)
    df = df.iloc[:, :num_columns]

    # Add more features
    df = get_data(df)

    # Separate X, y, and weights
    X, y, y_weights = separate_features(df, speedup_threshold, use_speedup, use_occurrences, use_log2_weights)
    X = reduce_dimensionality(X, y, y_weights)

    if split_train_test:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, y_weights, test_size=0.2
        )
    else:
        X_train, y_train, w_train = X, y, y_weights
        X_test, y_test, w_test = X, y, y_weights

    model, y_pred = run_decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
        w_train,
        w_test,
        scoring,
    )

    if not split_train_test:
        speedup_results = df[["conv_parameters", "occurrences", "speedup"]]
        print("\nOriginal speedup results:")
        plot_speedup(speedup_results, "old", "new", None, True)

        df["prediction"] = y_pred
        df = df.loc[df["prediction"] == 1]
        speedup_results = df[["conv_parameters", "occurrences", "speedup"]]
        print("\nHeuristic speedup results:")
        plot_speedup(speedup_results, "old", "new", None, True)

    prune_duplicate_leaves(model)

    rules = export_text(model, feature_names=list(X.columns))
    print("\nDecisionTreeClassifier rules:")
    print(rules)

    if plot_output_dir:
        plot_tree(model, proportion=True, filled=True, feature_names=list(X.columns), class_names=["class 0", "class 1"])
        plt.savefig(
            plot_output_dir / "decision_tree.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
