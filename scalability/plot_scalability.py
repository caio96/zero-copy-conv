#!/usr/bin/env python3
"""
Generate the scalability figure for the paper.

Layout: 3 rows x 2 cols so each panel matches the size of the original
2-panel figure when scaled to columnwidth. The bottom-right slot holds
the legend instead of a data panel.

Usage:
    python plot_scalability.py [--data PATH] [--output PATH]

Defaults:
    --data   : data.csv next to this script
    --output : scalability.png next to this script
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

CORES = [1, 2, 4, 8]

METHODS = [
    ("im2col",   "Im2col",         "#ca0020", "o", "-"),
    ("libtorch", "LibTorch",       "#555555", "s", "--"),
    ("lt_zconv", "LibTorch-ZConv", "#f4a582", "^", "-."),
    ("zconv",    "ZConv",          "#0571b0", "D", "-"),
]

PANELS = [
    (1, r"Layer~\#1 (regular)"),
    (2, r"Layer~\#2 (regular)"),
    (3, r"Layer~\#3 (small $IH$)"),
    (4, r"Layer~\#4 (strided)"),
    (5, r"Layer~\#5 (depthwise)"),
]

N_COLS = 2
N_ROWS = (len(PANELS) + N_COLS - 1) // N_COLS  # ceil(5/2) = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to data.csv (default: data.csv next to this script)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: scalability.png next to this script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path = args.data or (script_dir / "data.csv")
    output_path = args.output or (script_dir / "scalability.png")

    rc("font", **{"family": "serif", "serif": ["Libertine"]})
    rc("text", usetex=True)
    rc("text.latex", preamble="\n".join([
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{libertine}",
        r"\usepackage{newtxtext,newtxmath}",
        r"\usepackage{amsmath}",
    ]))
    plt.rcParams.update({"font.size": 14, "legend.fontsize": 12})

    data = pd.read_csv(data_path, comment="#")

    panel_w, panel_h = 7.5 / 2, 3.0
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(panel_w * N_COLS, panel_h * N_ROWS),
        sharey=True,
    )

    # Unused bottom-right slot holds the legend; hide any slots beyond that
    for i in range(len(PANELS) + 1, N_ROWS * N_COLS):
        axes.flat[i].set_visible(False)

    ideal_x = np.array(CORES)
    ideal_y = ideal_x / ideal_x[0]

    for idx, (layer_id, title) in enumerate(PANELS):
        ax = axes.flat[idx]
        df = data[data["layer"] == layer_id].sort_values("cores")

        ax.plot(CORES, ideal_y, color="lightgray", linestyle="--",
                linewidth=1.2, label="Ideal", zorder=1)

        for col, label, color, marker, linestyle in METHODS:
            if col not in df.columns:
                continue
            ax.plot(CORES, df[col].values, label=label, color=color,
                    marker=marker, markersize=6, linestyle=linestyle,
                    linewidth=1.8, zorder=2)

        ax.set_xticks(CORES)
        ax.set_xticklabels([str(c) for c in CORES])
        ax.set_xlabel("Cores")
        ax.set_title(title, pad=5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_ylim(bottom=0)

        if idx % N_COLS == 0:
            ax.set_ylabel("Speedup over 1 core")

    # Place legend in the empty bottom-right slot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax = axes.flat[len(PANELS)]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", frameon=True,
                     framealpha=1, edgecolor="black", handlelength=1.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
