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
    colors = ['#85c1e9', '#f9e79f']  # light blue, light yellow

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
                yerr=None, capsize=5)

    ax1.set_ylabel('Center Error')
    ax1.set_title('Center Error Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    # ax1.legend()
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
                yerr=None, capsize=5)

    ax2.set_ylabel('Radius Error')
    ax2.set_title('Radius Error Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_charts.png'), dpi=200, bbox_inches='tight')
    plt.show()

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

    # Create visualizations
    print("\nCreating visualizations...")
    plot_bar_charts(results, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("Generated files:")
    print("  - bar_charts.png: Mean error comparison")

if __name__ == "__main__":
    main()
