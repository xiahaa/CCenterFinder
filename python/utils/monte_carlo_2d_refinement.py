import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reuse the experiment harness
from experiment_2d_refinement import monte_carlo_experiment


def _single_trial(args):
    points_per_trial, search_ratio, seed = args
    rng = np.random.default_rng(seed)
    sr = max(0.1, float(search_ratio + 0.05 * (rng.random() - 0.5)))
    avg_err, Rres, tres, R, t, r_error, t_error = monte_carlo_experiment(num=points_per_trial, search_ratio=sr)
    return avg_err, r_error, t_error


def run_trials(num_trials: int, points_per_trial: int, search_ratio: float, seed: int = 42, workers: int = None):
    rng = np.random.default_rng(seed)

    methods = ["projected", "ellipse", "gt1", "gt2"]
    reproj_errors = {m: [] for m in methods}
    rot_errors = {m: [] for m in methods}
    trans_errors = {m: [] for m in methods}

    trial_args = [(points_per_trial, search_ratio, int(rng.integers(0, 2**31 - 1))) for _ in range(num_trials)]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_single_trial, a) for a in trial_args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Monte Carlo (2D refinement)"):
            try:
                avg_err, r_err, t_err = f.result()
                for m in methods:
                    if m in avg_err:
                        reproj_errors[m].append(float(avg_err[m]))
                    if m in r_err:
                        rot_errors[m].append(float(r_err[m]))
                    if m in t_err:
                        trans_errors[m].append(float(t_err[m]))
            except Exception:
                continue

    return reproj_errors, rot_errors, trans_errors


def save_csv(metrics: dict, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, filename)
    methods = sorted(metrics.keys())
    # Pad lists to same length
    max_len = max(len(v) for v in metrics.values()) if methods else 0
    padded = {m: (metrics[m] + [np.nan] * (max_len - len(metrics[m]))) for m in methods}
    arr = np.vstack([padded[m] for m in methods]).T
    header = ",".join(methods)
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    return csv_path


def plot_results(reproj_errors: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    methods = sorted(reproj_errors.keys())
    means = [np.nanmean(reproj_errors[m]) if len(reproj_errors[m]) > 0 else np.nan for m in methods]

    # Bar chart of mean reprojection error
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(methods, means, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("Mean reprojection error (px)")
    ax.set_title("2D refinement: mean reprojection error across trials")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mc2d_reproj_mean_bars.png"), dpi=200, bbox_inches="tight")

    # Box plot distribution
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.boxplot([reproj_errors[m] for m in methods], labels=methods, showfliers=False)
    ax2.set_ylabel("Reprojection error (px)")
    ax2.set_title("2D refinement: reprojection error distribution")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "mc2d_reproj_boxplot.png"), dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo experiment for 2D ellipse center refinement")
    parser.add_argument("--trials", type=int, default=100, help="Number of Monte Carlo trials")
    parser.add_argument("--points", type=int, default=20, help="Number of circles per trial")
    parser.add_argument("--search_ratio", type=float, default=0.3, help="Search ellipse scaling ratio")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Where to save results")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes")
    args = parser.parse_args()

    reproj_errors, rot_errors, trans_errors = run_trials(args.trials, args.points, args.search_ratio, args.seed, args.workers)
    csv_reproj = save_csv(reproj_errors, args.output_dir, "mc2d_reprojection_errors.csv")
    csv_rot = save_csv(rot_errors, args.output_dir, "mc2d_rotation_errors.csv")
    csv_trans = save_csv(trans_errors, args.output_dir, "mc2d_translation_errors.csv")
    plot_results(reproj_errors, args.output_dir)

    print(f"Saved CSVs: {csv_reproj}, {csv_rot}, {csv_trans}")
    print(f"Saved figures into: {args.output_dir}")


if __name__ == "__main__":
    main()
