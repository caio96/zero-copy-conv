#!/usr/bin/env python3
"""
Parse memory_raw.csv from run_memory.sh and produce formatted result tables.

Usage: python summarize_memory.py RAW_CSV [--out PATH]
  RAW_CSV : path to memory_raw.csv
  --out   : output CSV path (default: memory_results.csv next to this script)

Outputs:
  - memory_results.csv : mean RSS (MB) and 95% CI for each (executable, layer)
  - Printed tables matching the paper's memory table format (3 comparisons)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

LAYER_PARAMS = [
    (1, "1 32 112 112 64 3 3 1 1 1 1 1 1 1 1 1 0 0"),
    (2, "1 64 56 56 64 3 3 1 1 1 1 1 1 1 1 1 0 0"),
    (3, "1 224 7 7 224 3 3 1 1 1 1 1 1 1 1 1 0 0"),
    (4, "1 320 14 14 320 2 2 0 0 0 0 2 2 1 1 1 0 1"),
    (5, "1 384 14 14 384 3 3 1 1 1 1 1 1 1 1 384 0 1"),
    (6, "1 960 33 33 256 3 3 12 12 12 12 1 1 12 12 1 0 0"),
]
PARAMS_TO_LAYER = {p: n for n, p in LAYER_PARAMS}

EXE_TO_METHOD = {
    "benchmark_im2col": "Im2col",
    "benchmark_zero_copy": "ZConv",
    "benchmark_zero_copy_blis": "ZConv-BLIS",
    "benchmark_yaconv": "Yaconv",
    "benchmark_libtorch": "LibTorch",
    "benchmark_libtorch_zerocopy": "LibTorch-ZConv",
}

# Yaconv does not support groups or dilation; its RSS for Layers 5-6 is
# only the process overhead (SkipWithError). Exclude those rows.
YACONV_UNSUPPORTED_LAYERS = {5, 6}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_rss_kb"] = pd.to_numeric(df["max_rss_kb"], errors="coerce")
    df = df.dropna(subset=["max_rss_kb"])
    df["max_rss_mb"] = df["max_rss_kb"] / 1024.0

    agg = (
        df.groupby(["method", "layer"])["max_rss_mb"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    ci = st.norm.interval(0.95, loc=agg["mean"],
                          scale=agg["sem"].clip(lower=1e-12))
    agg["ci95"] = (ci[1] - ci[0]) / 2
    agg["significant"] = (agg["ci95"] / agg["mean"]) < 0.5  # sanity check

    return agg.rename(columns={"mean": "mean_mb", "sem": "sem_mb"})


def print_comparison(agg: pd.DataFrame, method_a: str, method_b: str,
                     ratio_label: str = None) -> pd.DataFrame:
    if ratio_label is None:
        ratio_label = f"{method_b} / {method_a}"

    a = agg[agg["method"] == method_a].set_index("layer")[["mean_mb", "ci95"]]
    b = agg[agg["method"] == method_b].set_index("layer")[["mean_mb", "ci95"]]

    if a.empty or b.empty:
        print(f"  (no data for {method_a} or {method_b})")
        return pd.DataFrame()

    layers = sorted(set(a.index) & set(b.index))
    rows = []
    for layer_id, params in LAYER_PARAMS:
        if layer_id not in layers:
            continue
        ma = a.loc[layer_id, "mean_mb"]
        mb = b.loc[layer_id, "mean_mb"]
        rows.append({
            "Layer": layer_id,
            "Params": params,
            f"{method_a} (MB)": round(ma, 2),
            f"{method_b} (MB)": round(mb, 2),
            ratio_label: round(mb / ma, 2),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        print(result.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path,
                        help="Path to memory_raw.csv from run_memory.sh")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV (default: memory_results.csv next to this script)")
    args = parser.parse_args()

    out_path = args.out or (Path(__file__).parent / "memory_results.csv")

    df = pd.read_csv(args.raw_csv)
    df["params"] = df["params"].str.strip()
    df["method"] = df["executable"].map(EXE_TO_METHOD)
    df["layer"] = df["params"].map(PARAMS_TO_LAYER)

    unknown_exe = df[df["method"].isna()]["executable"].unique()
    if len(unknown_exe) > 0:
        print(f"Warning: unknown executables (skipping): {list(unknown_exe)}")
    df = df.dropna(subset=["method", "layer"])
    df["layer"] = df["layer"].astype(int)

    # Drop Yaconv results for unsupported layers
    mask_yaconv_bad = (df["method"] == "Yaconv") & (df["layer"].isin(YACONV_UNSUPPORTED_LAYERS))
    df = df[~mask_yaconv_bad]

    agg = aggregate(df)
    agg.to_csv(out_path, index=False)
    print(f"Aggregated results saved to: {out_path}\n")

    print("=== Im2col vs ZConv ===")
    print_comparison(agg, "Im2col", "ZConv", "ZConv / Im2col")

    print("\n=== Yaconv vs ZConv-BLIS ===")
    print_comparison(agg, "Yaconv", "ZConv-BLIS", "ZConv-BLIS / Yaconv")

    print("\n=== LibTorch vs LibTorch-ZConv ===")
    print_comparison(agg, "LibTorch", "LibTorch-ZConv", "LibTorch-ZConv / LibTorch")


if __name__ == "__main__":
    main()
