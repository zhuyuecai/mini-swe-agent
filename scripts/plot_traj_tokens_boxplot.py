#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path


def total_tokens_in_value(value):
    if isinstance(value, dict):
        return sum(
            item if key == "total_tokens" and isinstance(item, int) else total_tokens_in_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(total_tokens_in_value(item) for item in value)
    return 0


def trajectory_token_sums(folder):
    traj_files = sorted(folder.glob("**/*.traj.json"))
    if not traj_files:
        raise ValueError(f"No *.traj.json files found in {folder}")
    return [total_tokens_in_value(json.loads(path.read_text())) for path in traj_files]

def get_labels_from_foldername(folder_name):
    if "vanilla" in folder_name:
        return "Vanilla"
    if "pkg" in folder_name:
        return "PKG"
    return "Developer Profile"

def plot_boxplot(token_sums, labels, output):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="font-cache-"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required to create the box plot. Install it with `python -m pip install matplotlib`."
        raise RuntimeError(msg) from e

    fig_width = max(8, len(labels) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    try:
        ax.boxplot(token_sums, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(token_sums, labels=labels, showmeans=True)
    ax.set_title("Total Tokens per Trajectory")
    ax.set_ylabel("Sum of total_tokens")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a box plot of per-trajectory total_tokens sums for one or more run folders."
    )
    parser.add_argument("folders", nargs="+", type=Path, help="Folders containing *.traj.json files.")
    parser.add_argument("-o", "--output", type=Path, default=Path("trajectory-token-boxplot.png"), help="Output image path.")
    return parser.parse_args()


def main():
    args = parse_args()
    token_sums = []
    labels = []
    for folder in args.folders:
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")
        sums = trajectory_token_sums(folder)
        token_sums.append(sums)
        labels.append(get_labels_from_foldername(folder.name))
        print(
            f"{folder}: {len(sums)} trajectories, total={sum(sums)}, "
            f"median={statistics.median(sums)}, min={min(sums)}, max={max(sums)}"
        )
    plot_boxplot(token_sums, labels, args.output)
    print(f"Saved box plot to {args.output}")


if __name__ == "__main__":
    main()
