#!/usr/bin/env python3
"""
Script to run the complete C++ Monte Carlo benchmark and analyze results.
"""

import os
import subprocess
import sys
import argparse

def run_cpp_benchmark(num_experiments=1000, results_dir="results"):
    """Run the C++ Monte Carlo benchmark."""
    print("Building C++ benchmark...")

    # Create build directory
    if not os.path.exists("build"):
        os.makedirs("build")

    # Build the benchmark
    try:
        # Configure with CMake
        subprocess.run(["cmake", ".."], cwd="build", check=True)

        # Build
        subprocess.run(["make", "monte_carlo_benchmark"], cwd="build", check=True)

        print("✓ Build successful")

    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        return False

    # Run the benchmark
    print(f"Running Monte Carlo benchmark with {num_experiments} experiments...")
    try:
        cmd = ["./monte_carlo_benchmark", str(num_experiments), results_dir]
        subprocess.run(cmd, cwd="build", check=True)

        print("✓ Benchmark completed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def run_analysis(results_dir="results", output_dir="analysis_output"):
    """Run the Python analysis script."""
    print("Analyzing results...")

    try:
        cmd = [sys.executable, "python/utils/plot_monte_carlo_results.py.py",
               "--results_dir", results_dir, "--output_dir", output_dir]
        subprocess.run(cmd, check=True)

        print("✓ Analysis completed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed: {e}")
        return False

def main():
    """Main function to run the complete benchmark pipeline."""
    parser = argparse.ArgumentParser(description="Run C++ Monte Carlo benchmark and analyze results")
    parser.add_argument('--num_experiments', type=int, default=1000,
                       help='Number of experiments per scenario')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save C++ results')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis plots')
    parser.add_argument('--skip_build', action='store_true',
                       help='Skip building and just run analysis')
    args = parser.parse_args()

    print("C++ Monte Carlo Benchmark Pipeline")
    print("=" * 40)

    success = True

    # Run C++ benchmark
    if not args.skip_build:
        success = run_cpp_benchmark(args.num_experiments, args.results_dir)
        if not success:
            print("Failed to run C++ benchmark")
            return False

    # Run analysis
    success = run_analysis(args.results_dir, args.output_dir)
    if not success:
        print("Failed to run analysis")
        return False

    print("\n🎉 Complete pipeline finished successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Analysis plots saved to: {args.output_dir}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
