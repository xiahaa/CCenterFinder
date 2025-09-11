"""
Python script to analyze and visualize results from C++ Monte Carlo benchmark.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
from typing import Dict, List, Tuple
import argparse

# Global font settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['font.size'] = 12

def load_results(results_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load results from C++ benchmark output files.

    Parameters
    ----------
    results_dir : str
        Directory containing the result files

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Nested dictionary: {scenario: {method: data}}
        data is array with columns: [center_error, radius_error, success]
    """
    results = {}

    # Find all result files
    pattern = os.path.join(results_dir, "*_results.txt")
    files = glob.glob(pattern)

    for file_path in files:
        filename = os.path.basename(file_path)
        # Parse filename: scenario_method_results.txt
        parts = filename.replace("_results.txt", "").split("_")
        if len(parts) >= 2:
            method = parts[-1]  # Last part is method (pcl, cga, classical)
            scenario = "_".join(parts[:-1])  # Everything else is scenario

            if scenario not in results:
                results[scenario] = {}

            # Load data
            try:
                data = np.loadtxt(file_path, skiprows=1)  # Skip header
                results[scenario][method] = data
                print(f"Loaded {filename}: {data.shape[0]} experiments")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return results

def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a method's results.

    Parameters
    ----------
    data : np.ndarray
        Array with columns: [center_error, radius_error, success]

    Returns
    -------
    Dict[str, float]
        Dictionary with statistics
    """
    # Filter successful experiments
    success_mask = data[:, 2] == 1
    successful_data = data[success_mask]

    if len(successful_data) == 0:
        return {
            'success_rate': 0.0,
            'mean_center_error': np.nan,
            'std_center_error': np.nan,
            'mean_radius_error': np.nan,
            'std_radius_error': np.nan,
            'num_successful': 0,
            'num_total': len(data)
        }

    center_errors = successful_data[:, 0]
    radius_errors = successful_data[:, 1]

    return {
        'success_rate': len(successful_data) / len(data),
        'mean_center_error': np.mean(center_errors),
        'std_center_error': np.std(center_errors),
        'mean_radius_error': np.mean(radius_errors),
        'std_radius_error': np.std(radius_errors),
        'num_successful': len(successful_data),
        'num_total': len(data)
    }

def plot_bar_charts(results: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Create bar charts comparing mean errors across scenarios."""
    scenarios = list(results.keys())
    methods = ['pcl', 'cga']
    method_labels = ['PCL', 'CGA']
    colors = ['#e74c3c', '#f8c471']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Center error plot
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.25

    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        stds = []

        for scenario in scenarios:
            if method in results[scenario]:
                stats = compute_statistics(results[scenario][method])
                means.append(stats['mean_center_error'])
                stds.append(stats['std_center_error'])
            else:
                means.append(0)
                stds.append(0)

        ax1.bar(x + i * width, means, width, label=label, color=color, alpha=0.8,
                yerr=stds, capsize=5)

    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Center Error')
    ax1.set_title('Center Error Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Radius error plot
    ax2 = axes[1]

    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = []
        stds = []

        for scenario in scenarios:
            if method in results[scenario]:
                stats = compute_statistics(results[scenario][method])
                means.append(stats['mean_radius_error'])
                stds.append(stats['std_radius_error'])
            else:
                means.append(0)
                stds.append(0)

        ax2.bar(x + i * width, means, width, label=label, color=color, alpha=0.8,
                yerr=stds, capsize=5)

    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Radius Error')
    ax2.set_title('Radius Error Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_charts.png'), dpi=200, bbox_inches='tight')
    plt.show()

def plot_box_plots(results: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Create box plots for detailed error distribution analysis."""
    scenarios = list(results.keys())
    methods = ['pcl', 'cga']
    method_labels = ['PCL', 'CGA']
    colors = ['#e74c3c', '#f8c471']

    fig, axes = plt.subplots(2, len(scenarios), figsize=(4 * len(scenarios), 8))
    if len(scenarios) == 1:
        axes = axes.reshape(2, 1)

    for i, scenario in enumerate(scenarios):
        # Center error box plot
        ax1 = axes[0, i]
        data_to_plot = []
        labels = []

        for method, label, color in zip(methods, method_labels, colors):
            if method in results[scenario]:
                data = results[scenario][method]
                success_mask = data[:, 2] == 1
                successful_data = data[success_mask]

                if len(successful_data) > 0:
                    data_to_plot.append(successful_data[:, 0])  # center errors
                    labels.append(label)

        if data_to_plot:
            bp1 = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp1['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax1.set_title(f'{scenario.replace("_", " ").title()}\nCenter Error')
        ax1.set_ylabel('Center Error')
        ax1.grid(True, alpha=0.3)

        # Radius error box plot
        ax2 = axes[1, i]
        data_to_plot = []
        labels = []

        for method, label, color in zip(methods, method_labels, colors):
            if method in results[scenario]:
                data = results[scenario][method]
                success_mask = data[:, 2] == 1
                successful_data = data[success_mask]

                if len(successful_data) > 0:
                    data_to_plot.append(successful_data[:, 1])  # radius errors
                    labels.append(label)

        if data_to_plot:
            bp2 = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_title(f'{scenario.replace("_", " ").title()}\nRadius Error')
        ax2.set_ylabel('Radius Error')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_plots.png'), dpi=200, bbox_inches='tight')
    plt.show()

def plot_success_rates(results: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """Create bar chart showing success rates for each method and scenario."""
    scenarios = list(results.keys())
    methods = ['pcl', 'cga']
    method_labels = ['PCL', 'CGA']
    colors = ['#e74c3c', '#f8c471']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenarios))
    width = 0.25

    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        success_rates = []

        for scenario in scenarios:
            if method in results[scenario]:
                stats = compute_statistics(results[scenario][method])
                success_rates.append(stats['success_rate'])
            else:
                success_rates.append(0)

        ax.bar(x + i * width, success_rates, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        for j, scenario in enumerate(scenarios):
            if method in results[scenario]:
                stats = compute_statistics(results[scenario][method])
                height = stats['success_rate']
                ax.text(x[j] + i * width, height + 0.01, f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates.png'), dpi=200, bbox_inches='tight')
    plt.show()

def print_statistics(results: Dict[str, Dict[str, np.ndarray]]):
    """Print detailed statistics for all methods and scenarios."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    scenarios = list(results.keys())
    methods = ['pcl', 'cga']
    method_labels = ['PCL', 'CGA']

    for scenario in scenarios:
        print(f"\nScenario: {scenario.replace('_', ' ').title()}")
        print("-" * 50)

        for method, label in zip(methods, method_labels):
            if method in results[scenario]:
                stats = compute_statistics(results[scenario][method])
                print(f"{label:>10}: Success={stats['success_rate']:.3f} "
                      f"({stats['num_successful']}/{stats['num_total']}) | "
                      f"Center Error: {stats['mean_center_error']:.4f}±{stats['std_center_error']:.4f} | "
                      f"Radius Error: {stats['mean_radius_error']:.4f}±{stats['std_radius_error']:.4f}")
            else:
                print(f"{label:>10}: No data available")

def main():
    """Main function to analyze C++ benchmark results."""
    parser = argparse.ArgumentParser(description="Analyze C++ Monte Carlo benchmark results")
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading C++ benchmark results...")
    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        print("Make sure to run the C++ benchmark first:")
        print("  mkdir build && cd build")
        print("  cmake .. && make monte_carlo_benchmark")
        print("  ./monte_carlo_benchmark 1000")
        return

    print(f"Loaded results for {len(results)} scenarios")

    # Print statistics
    print_statistics(results)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_bar_charts(results, args.output_dir)
    plot_box_plots(results, args.output_dir)
    plot_success_rates(results, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("Generated files:")
    print("  - bar_charts.png: Mean error comparison")
    print("  - box_plots.png: Error distribution analysis")
    print("  - success_rates.png: Success rate comparison")

if __name__ == "__main__":
    main()
