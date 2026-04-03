import os
import re
import pandas as pd
import numpy as np

log_dir = "benchmark_results"

records = []

# --- Parse logs ---
for filename in os.listdir(log_dir):
    if not filename.endswith(".log"):
        continue

    with open(os.path.join(log_dir, filename), "r") as f:
        content = f.read()

    runs = content.split("BEGIN_RUN")

    for run in runs:
        if "END_RUN" not in run:
            continue

        exe_match = re.search(r"EXECUTABLE:\s*(\S+)", run)
        params_match = re.search(r"PARAMS:\s*(.*)", run)
        rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", run)

        if not (exe_match and params_match and rss_match):
            continue

        records.append({
            "params": params_match.group(1).strip(),
            "executable": exe_match.group(1),
            "rss_kb": int(rss_match.group(1)),
        })

df = pd.DataFrame(records)

# --- Rename methods ---
rename_map = {
    "benchmark_im2col": "Im2col",
    "benchmark_yaconv": "Yaconv",
    "benchmark_zero_copy": "ZConv",
    "benchmark_zero_copy_blis": "ZConv-BLIS",
    "benchmark_libtorch": "LibTorch",
    "benchmark_libtorch_zerocopy": "LibTorch-ZConv",
}
df["method"] = df["executable"].map(rename_map)

# --- Convert to MB ---
df["rss_mb"] = df["rss_kb"] / 1024.0

# --- Aggregate: mean + CI ---
grouped = df.groupby(["params", "method"])["rss_mb"]
stats = grouped.agg(["mean", "std", "count"]).reset_index()

stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])

# --- Pivot ---
mean_table = stats.pivot(index="params", columns="method", values="mean")
ci_table = stats.pivot(index="params", columns="method", values="ci95")

# --- Significance ---
def compute_significance(base, new):
    base_mean = mean_table[base]
    base_ci = ci_table[base]

    new_mean = mean_table[new]
    new_ci = ci_table[new]

    base_low = base_mean - base_ci
    base_high = base_mean + base_ci

    new_low = new_mean - new_ci
    new_high = new_mean + new_ci

    return (base_low > new_high) | (new_low > base_high)

# --- Build tables ---
pd.options.display.float_format = "{:.3f}".format

def build_table(methods, baseline):
    table = pd.DataFrame(index=mean_table.index)

    # Means
    for m in methods:
        table[m] = mean_table[m]

    # CI
    for m in methods:
        table[f"{m}_ci"] = ci_table[m]

    # Relative (only for non-baseline)
    for m in methods:
        if m == baseline:
            continue
        table[f"{m}_rel"] = mean_table[m] / mean_table[baseline]

    # Significance
    for m in methods:
        if m == baseline:
            continue
        table[f"{m}_significant"] = compute_significance(baseline, m)

    return table

# --- Tables ---
table_im2col = build_table(["Im2col", "ZConv"], "Im2col")
table_yaconv = build_table(["Yaconv", "ZConv-BLIS"], "Yaconv")
table_libtorch = build_table(["LibTorch", "LibTorch-ZConv"], "LibTorch")

# --- Print ---
print("\n=== Memory Table (MB) — Im2col ===\n")
print(table_im2col.to_string())

print("\n=== Memory Table (MB) — Yaconv ===\n")
print(table_yaconv.to_string())

print("\n=== Memory Table (MB) — LibTorch ===\n")
print(table_libtorch.to_string())
