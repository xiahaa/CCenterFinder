import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Import the comparison functions from the existing script
sys.path.append(os.path.dirname(__file__))
from compare_decoupled_vs_joint import (
    circle_fitting_decoupled,
    _simulate_case_offplane_noise, _simulate_case_limited_arc_nonuniform,
    _simulate_case_sparse_nonuniform, _simulate_case_two_side_different_intervals,
    _normalize, _pick_u_from_normal
)
from cga_joint_fitting import cga_robust_circle_fitting_joint

# Global font settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['font.size'] = 12

def compute_metrics(points: np.ndarray, true_center: np.ndarray, true_radius: float,
                   true_normal: np.ndarray, method_name: str) -> Dict[str, float]:
    """Compute error metrics for a fitting method."""
    try:
        if method_name == 'Classical':
            center, radius, normal, _ = circle_fitting_decoupled(points)
        elif method_name == 'Proposed':
            center, radius, normal = cga_robust_circle_fitting_joint(points)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Center error
        center_error = np.linalg.norm(center - true_center)

        # Radius error
        radius_error = abs(radius - true_radius)

        return {
            'center_error': center_error,
            'radius_error': radius_error,
            'success': True
        }
    except Exception as e:
        print(f"Error in {method_name}: {e}")
        return {
            'center_error': np.nan,
            'radius_error': np.nan,
            'success': False
        }

def run_monte_carlo_experiment(scenario_name: str, scenario_func, num_experiments: int = 1000,
                              seed: int = 42) -> Dict[str, List[Dict[str, float]]]:
    """Run Monte Carlo experiments for a given scenario."""
    rng = np.random.default_rng(seed)

    classical_results = []
    proposed_results = []

    print(f"Running Monte Carlo experiments for: {scenario_name}")

    for i in tqdm(range(num_experiments), desc=scenario_name):
        # Generate random ground truth
        true_center = rng.uniform(-2, 2, size=3)
        true_radius = rng.uniform(1.0, 5.0)
        true_normal = _normalize(rng.normal(size=3))

        # Generate scenario-specific points
        points = scenario_func(true_center, true_radius, true_normal, seed=rng.integers(0, 2**31))

        # Test both methods
        classical_metrics = compute_metrics(points, true_center, true_radius, true_normal, 'Classical')
        proposed_metrics = compute_metrics(points, true_center, true_radius, true_normal, 'Proposed')

        classical_results.append(classical_metrics)
        proposed_results.append(proposed_metrics)

    return {
        'Classical': classical_results,
        'Proposed': proposed_results
    }

def create_scenario_functions():
    """Create scenario-specific point generation functions."""
    def scenario1_func(center, radius, normal, seed=42):
        return _simulate_case_offplane_noise(center, radius, normal, num=100, noise_std=0.2, seed=seed)

    def scenario2_func(center, radius, normal, seed=42):
        return _simulate_case_limited_arc_nonuniform(center, radius, normal, num=100, arc_deg=70.0, noise_std=0.2, seed=seed)

    def scenario3_func(center, radius, normal, seed=42):
        return _simulate_case_sparse_nonuniform(center, radius, normal, num=12, noise_std=0.2, seed=seed)

    def scenario4_func(center, radius, normal, seed=42):
        return _simulate_case_two_side_different_intervals(center, radius, normal, num_side1=20, arc1_deg=200.0, noise_std=0.2, seed=seed)

    return {
        'Isotropic 3D noise': scenario1_func,
        'Limited span': scenario2_func,
        'Sparse non-uniform': scenario3_func,
        'Symmetric distribution': scenario4_func
    }

def plot_bar_charts(all_results: Dict[str, Dict[str, List[Dict[str, float]]]],
                   output_dir: str):
    """Create bar charts comparing mean errors across scenarios."""
    metrics = ['center_error', 'radius_error']
    metric_labels = ['Center Error', 'Radius Error']

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]

        scenarios = list(all_results.keys())
        classical_means = []
        proposed_means = []
        classical_stds = []
        proposed_stds = []

        for scenario in scenarios:
            classical_data = [r[metric] for r in all_results[scenario]['Classical'] if r['success']]
            proposed_data = [r[metric] for r in all_results[scenario]['Proposed'] if r['success']]

            classical_means.append(np.mean(classical_data))
            proposed_means.append(np.mean(proposed_data))
            classical_stds.append(np.std(classical_data))
            proposed_stds.append(np.std(proposed_data))

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax.bar(x - width/2, classical_means, width, label='Classical',
                      color='#48c9b0', alpha=0.8, yerr=classical_stds, capsize=5)
        bars2 = ax.bar(x + width/2, proposed_means, width, label='Proposed',
                      color='#f8c471', alpha=0.8, yerr=proposed_stds, capsize=5)

        ax.set_xlabel('Scenario')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monte_carlo_bar_charts.png'), dpi=200, bbox_inches='tight')
    plt.show()

def plot_box_plots(all_results: Dict[str, Dict[str, List[Dict[str, float]]]],
                  output_dir: str):
    """Create box plots comparing error distributions across scenarios."""
    metrics = ['center_error', 'radius_error']
    metric_labels = ['Center Error', 'Radius Error']

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]

        classical_data = []
        proposed_data = []
        scenario_labels = []

        for scenario in all_results.keys():
            classical_vals = [r[metric] for r in all_results[scenario]['Classical'] if r['success']]
            proposed_vals = [r[metric] for r in all_results[scenario]['Proposed'] if r['success']]

            classical_data.extend(classical_vals)
            proposed_data.extend(proposed_vals)
            scenario_labels.extend([f'{scenario}\n(Classical)'] * len(classical_vals))
            scenario_labels.extend([f'{scenario}\n(Proposed)'] * len(proposed_vals))

        # Create box plot data
        all_data = classical_data + proposed_data
        positions = []
        labels = []

        pos = 0
        for scenario in all_results.keys():
            classical_vals = [r[metric] for r in all_results[scenario]['Classical'] if r['success']]
            proposed_vals = [r[metric] for r in all_results[scenario]['Proposed'] if r['success']]

            if classical_vals:
                positions.append(pos)
                labels.append(f'{scenario}\nClassical')
                pos += 1

            if proposed_vals:
                positions.append(pos)
                labels.append(f'{scenario}\nProposed')
                pos += 1

        # Plot boxes
        data_by_pos = {}
        pos_idx = 0
        for scenario in all_results.keys():
            classical_vals = [r[metric] for r in all_results[scenario]['Classical'] if r['success']]
            proposed_vals = [r[metric] for r in all_results[scenario]['Proposed'] if r['success']]

            if classical_vals:
                data_by_pos[pos_idx] = classical_vals
                pos_idx += 1

            if proposed_vals:
                data_by_pos[pos_idx] = proposed_vals
                pos_idx += 1

        if data_by_pos:
            bp = ax.boxplot([data_by_pos[pos] for pos in sorted(data_by_pos.keys())],
                           positions=sorted(data_by_pos.keys()), patch_artist=True)

            # Color boxes
            for i, patch in enumerate(bp['boxes']):
                if i % 2 == 0:  # Classical
                    patch.set_facecolor('#48c9b0')
                    patch.set_alpha(0.7)
                else:  # Proposed
                    patch.set_facecolor('#f8c471')
                    patch.set_alpha(0.7)

        ax.set_xlabel('Scenario and Method')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monte_carlo_box_plots.png'), dpi=200, bbox_inches='tight')
    plt.show()

def print_statistics(all_results: Dict[str, Dict[str, List[Dict[str, float]]]]):
    """Print detailed statistics for each scenario and method."""
    metrics = ['center_error', 'radius_error']
    metric_labels = ['Center Error', 'Radius Error']

    print("\n" + "="*80)
    print("MONTE CARLO BENCHMARKING RESULTS")
    print("="*80)

    for scenario, results in all_results.items():
        print(f"\n{scenario.upper()}:")
        print("-" * 50)

        for method in ['Classical', 'Proposed']:
            method_results = [r for r in results[method] if r['success']]
            success_rate = len(method_results) / len(results[method]) * 100

            print(f"\n{method} Method (Success Rate: {success_rate:.1f}%):")

            for metric, label in zip(metrics, metric_labels):
                values = [r[metric] for r in method_results]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    median_val = np.median(values)
                    q25 = np.percentile(values, 25)
                    q75 = np.percentile(values, 75)

                    print(f"  {label}:")
                    print(f"    Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
                    print(f"    Median: {median_val:.4f}")
                    print(f"    IQR: [{q25:.4f}, {q75:.4f}]")

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo benchmarking of circle fitting methods")
    parser.add_argument('--num_experiments', type=int, default=1000, help='Number of experiments per scenario')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get scenario functions
    scenario_functions = create_scenario_functions()

    # Run Monte Carlo experiments
    all_results = {}
    for scenario_name, scenario_func in scenario_functions.items():
        results = run_monte_carlo_experiment(scenario_name, scenario_func,
                                           args.num_experiments, args.seed)
        all_results[scenario_name] = results

    # Generate visualizations
    print("\nGenerating bar charts...")
    plot_bar_charts(all_results, args.output_dir)

    print("Generating box plots...")
    plot_box_plots(all_results, args.output_dir)

    # Print statistics
    print_statistics(all_results)

    # Save raw results
    results_file = os.path.join(args.output_dir, 'monte_carlo_results.npz')
    np.savez(results_file, **{f"{scenario}_{method}":
                             np.array([[r['center_error'], r['radius_error']]
                                     for r in results[method] if r['success']])
                             for scenario, results in all_results.items()
                             for method in ['Classical', 'Proposed']})

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Raw data saved to: {results_file}")

if __name__ == "__main__":
    main()
