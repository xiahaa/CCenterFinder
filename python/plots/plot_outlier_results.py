"""
Load outlier benchmark CSVs from results/ and draw comparison figures.

Expected inputs created by src/benchmark_outlier.cpp:
- results/outlier_prob_XX_pcl_results.txt with header: center_error,radius_error,success
- results/outlier_prob_XX_cga_results.txt with same columns
- results/outlier_summary.csv with columns: prob,method,mean_center_error
"""

import os
import glob
import argparse
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['font.size'] = 12


def load_method_file(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception:
        # Fallback if files are whitespace separated by accident
        return np.loadtxt(path, skiprows=1)


def discover_results(results_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    data: Dict[str, Dict[str, np.ndarray]] = {}
    pcl_files = glob.glob(os.path.join(results_dir, "outlier_prob_*_pcl_results.txt"))
    for pcl_path in sorted(pcl_files):
        token = os.path.basename(pcl_path).replace("outlier_prob_", "").replace("_pcl_results.txt", "")
        cga_path = os.path.join(results_dir, f"outlier_prob_{token}_cga_results.txt")
        if not os.path.exists(cga_path):
            continue
        data.setdefault(token, {})
        data[token]["pcl"] = load_method_file(pcl_path)
        data[token]["cga"] = load_method_file(cga_path)
    return data


def compute_stats(arr: np.ndarray) -> Tuple[float, float, float, float, float]:
    # columns: center_error, radius_error, success
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, 0.0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mask = arr[:, 2] == 1
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan, 0.0
    ce = arr[mask, 0]
    re = arr[mask, 1]
    return float(np.mean(ce)), float(np.std(ce)), float(np.mean(re)), float(np.std(re)), float(np.mean(mask))


def plot_bars(results: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    tokens = sorted(results.keys(), key=lambda s: float(s))
    probs = [float(t) for t in tokens]

    pcl_mean = []
    cga_mean = []
    pcl_std = []
    cga_std = []

    for t in tokens:
        m_ce, m_cstd, _, _, _ = compute_stats(results[t]["pcl"]) if "pcl" in results[t] else (np.nan, np.nan, np.nan, np.nan, np.nan)
        c_ce, c_cstd, _, _, _ = compute_stats(results[t]["cga"]) if "cga" in results[t] else (np.nan, np.nan, np.nan, np.nan, np.nan)
        pcl_mean.append(m_ce)
        cga_mean.append(c_ce)
        pcl_std.append(m_cstd)
        cga_std.append(c_cstd)

    x = np.arange(len(probs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    # Use lighter colors for the bars
    ax.bar(x - width/2, pcl_mean, width, label="PCL", color="#AFC6E9")   # light blue
    ax.bar(x + width/2, cga_mean, width, label="CGA", color="#FFD7B1")   # light orange
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in probs])
    ax.set_xlabel("Outlier probability")
    ax.set_ylabel("Mean center error")
    ax.set_title("Center error vs. outlier probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outlier_center_error_bars.png"), dpi=600, bbox_inches="tight")

    # Box plots per probability for more distribution detail
    # fig2, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    # pcl_boxes = [results[t]["pcl"][results[t]["pcl"][:, 2] == 1, 0] for t in tokens]
    # cga_boxes = [results[t]["cga"][results[t]["cga"][:, 2] == 1, 0] for t in tokens]

    # axs[0].boxplot(pcl_boxes, labels=[f"{p:.2f}" for p in probs], showfliers=False)
    # axs[0].set_title("PCL center error distribution")
    # axs[0].set_xlabel("Outlier probability")
    # axs[0].set_ylabel("Center error")
    # axs[0].grid(True, alpha=0.3)

    # axs[1].boxplot(cga_boxes, labels=[f"{p:.2f}" for p in probs], showfliers=False)
    # axs[1].set_title("CGA center error distribution")
    # axs[1].set_xlabel("Outlier probability")
    # axs[1].grid(True, alpha=0.3)

    # fig2.tight_layout()
    # fig2.savefig(os.path.join(output_dir, "outlier_center_error_boxplots.png"), dpi=600, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="analysis_output")
    args = parser.parse_args()

    results = discover_results(args.results_dir)
    if not results:
        print(f"No outlier results found in {args.results_dir}. Run the C++ benchmark first.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    plot_bars(results, args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
