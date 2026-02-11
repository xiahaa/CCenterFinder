# CCenterFinder

[![CMake Build](https://github.com/xiahaa/CCenterFinder/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/xiahaa/CCenterFinder/actions/workflows/cmake-multi-platform.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Accurate and robust localization of 3D and 2D circle centers using Conformal Geometric Algebra (CGA).

## Features

- **3D Circle Fitting**: High-precision 3D circle fitting via Conformal Geometric Algebra (CGA)
- **Robust Methods**: Improved numerical stability through normalization and stable eigenvalue selection
- **RANSAC Support**: Outlier-robust fitting with customizable RANSAC implementation
- **Multiple Baselines**: Compare with PCL's Circle3D RANSAC implementation
- **Comprehensive Testing**: Unified test program with multiple modes for easy validation
- **Performance Benchmarks**: Monte Carlo simulations across various noise and outlier scenarios

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Executables](#executables)
- [Algorithms](#algorithms)
- [Benchmarks](#benchmarks)
- [Python Tools](#python-tools)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Dependencies

### Required C++ Libraries

- **Eigen3** (≥3.3): Linear algebra library
- **OpenCV** (≥3.0): Computer vision library for I/O and examples
- **PCL** (Point Cloud Library): For PCL RANSAC baseline comparison
- **Boost**: Filesystem library for file I/O
- **OpenMP**: Parallel computing support

### Optional Python Dependencies (for analysis and demos)

```bash
pip install numpy matplotlib opencv-python
```

### Installation of Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y libeigen3-dev libopencv-dev libboost-all-dev libpcl-dev
```

#### macOS (using Homebrew)
```bash
brew install eigen opencv boost pcl
```

#### Windows (using vcpkg)
```powershell
vcpkg install eigen3 opencv boost-filesystem pcl
```

### RANSAC Templates

Code in `include/rtl/` adapted from:
- [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC)
- [RansacLib](https://github.com/tsattler/RansacLib)

## Installation

Clone the repository:

```bash
git clone https://github.com/xiahaa/CCenterFinder.git
cd CCenterFinder
```

## Build Instructions

### Linux

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all targets
make -j$(nproc)

# Or build specific targets
make simple_test
make monte_carlo_benchmark
make benchmark_outlier
make benchmark3d
```

### Windows (PowerShell)

```powershell
# Create build directory
mkdir build
cd build

# Configure with Visual Studio
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build in Release mode
cmake --build . --config Release

# Or build specific targets
cmake --build . --config Release --target simple_test
cmake --build . --config Release --target monte_carlo_benchmark
```

### macOS

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all targets
make -j$(sysctl -n hw.ncpu)
```

## Usage

## Executables

### simple_test

Unified test program with three modes for easy testing and validation.

**Available Modes:**
1. **Generated** (default): Creates synthetic 3D circle data and fits it
2. **Given**: Uses predefined test data from the code
3. **RANSAC**: Creates noisy data with outliers and uses RANSAC for robust fitting

**Usage:**
```bash
# Run with default (generated) mode
./build/simple_test

# Use predefined test data
./build/simple_test given

# RANSAC mode with custom parameters
./build/simple_test ransac --threshold 0.1 --iterations 300 --noise 0.01 --outliers 30

# Show help
./build/simple_test --help
```

**RANSAC Parameters:**
- `--threshold <value>`: RANSAC distance threshold (default: 0.1)
- `--iterations <value>`: Number of RANSAC iterations (default: 300)
- `--noise <value>`: Gaussian noise standard deviation (default: 0.01)
- `--outliers <value>`: Number of outlier points to add (default: 30)

### monte_carlo_benchmark

Monte Carlo comparison across multiple scenarios: isotropic noise, limited arcs, sparse non-uniform sampling, and symmetric non-uniform sampling.

**Usage:**
```bash
# Run 1000 experiments, save results to 'results' directory
./build/monte_carlo_benchmark 1000 results
```

**Output:**
- Per-scenario text results for statistical analysis
- Performance metrics (accuracy, precision, execution time)

### benchmark_outlier

Outlier-robustness benchmark comparing CGA with RANSAC vs PCL's Circle3D implementation.

**Usage:**
```bash
./build/benchmark_outlier
```

**Features:**
- CGA path uses RANSAC with Robust CGA estimator
- PCL path uses `pcl::SACSegmentation` with `SACMODEL_CIRCLE3D`
- Configurable RANSAC thresholds and iterations

**Tips:**
- Tune RANSAC threshold (typically 0.1–0.2) based on your data scale
- Increase iterations for high outlier ratios (>50%)

### benchmark3d

Simple noise benchmark for evaluating fitting accuracy under various noise conditions.

**Usage:**
```bash
./build/benchmark3d
```

## Algorithms

### Original CGA (C++)

**Implementation:** `include/Fit3DCircle.hpp`

The baseline CGA method without pre-normalization. Uses conformal geometric algebra to solve for 3D circle parameters through eigenvalue decomposition.

**Method:** `ConformalFit3DCircle::Fit(points, center, radius)`

### Robust CGA (C++)

**Implementation:** `include/RobustFit3DCircle.hpp`

Improves numerical stability through several enhancements:

1. **Centering and Scaling**: Center points and scale so RMS ≈ √2 before forming the CGA system
2. **Robust Eigenvalue Selection**: Select two smallest positive eigenvalues; fallback to two smallest
3. **Parameter Recovery**: Recover circle parameters; then unscale and uncenter outputs

**Usage:**
```cpp
#include "RobustFit3DCircle.hpp"

Eigen::Vector3d center;
double radius;
Eigen::Vector3d normal;

// Fit circle to 3D points
int result = robust_cga::RobustFit3DCircle::Fit(points, center, radius, &normal);

if (result == 0) {
    std::cout << "Center: " << center.transpose() << std::endl;
    std::cout << "Radius: " << radius << std::endl;
    std::cout << "Normal: " << normal.transpose() << std::endl;
}
```

### RANSAC Implementation

The project uses a custom RANSAC implementation based on the RTL (RANSAC Template Library) framework:

**Components:**
- **RTL Framework**: Located in `include/rtl/` - generic RANSAC templates
- **Circle3D Estimator**: Implements the RTL::Estimator interface for 3D circle fitting
- **Robust CGA Integration**: Uses Robust CGA for model computation within RANSAC
- **Distance Metric**: 3D point-to-circle distance considering both plane and edge distances

**Key Features:**
- Configurable threshold for inlier classification
- Adjustable iteration count
- Support for different RANSAC variants (MSAC, MLESAC, LMedS)

## Benchmarks

### Outlier Benchmark

**Source:** `src/benchmark_outlier.cpp`

Compares the robustness of different fitting methods under varying outlier ratios:

- **CGA with RANSAC**: Uses the RTL RANSAC framework with Robust CGA estimator, then refines on inliers
- **PCL RANSAC**: Uses `pcl::SACSegmentation<pcl::PointXYZ>` with `SACMODEL_CIRCLE3D`

**Running the Benchmark:**

```bash
# Linux/macOS
./build/benchmark_outlier

# Windows
.\build\Release\benchmark_outlier.exe
```

**Configuration Tips:**
- Adjust RANSAC threshold (default ~0.1–0.2) based on your data scale
- Increase iterations for datasets with high outlier ratios (>50%)
- Monitor inlier counts to verify successful convergence

### Monte Carlo Benchmark

**Source:** `src/monte_carlo_benchmark.cpp`

Comprehensive statistical evaluation across four scenarios:

1. **Isotropic Noise**: Gaussian noise added uniformly in all directions
2. **Limited Arcs**: Only partial circle coverage available
3. **Sparse Non-Uniform**: Irregular point spacing
4. **Symmetric Non-Uniform**: Symmetric but non-uniform sampling

**Running the Benchmark:**

```bash
# Run with 1000 experiments per scenario
./build/monte_carlo_benchmark 1000 results

# Results are saved to the 'results' directory
# Output format: per-scenario text files with statistics
```

**Output Metrics:**
- Center error (Euclidean distance)
- Radius error (absolute difference)
- Execution time
- Success rate

### Performance Characteristics

**Typical Performance:**
- **Simple Fitting**: <1ms for 100 points on modern hardware
- **RANSAC Fitting**: 5-50ms depending on iterations and point count
- **Monte Carlo (1000 runs)**: ~10-60 seconds per scenario

**Accuracy:**
- **Low Noise** (<0.01 units): Sub-millimeter precision
- **Medium Noise** (0.01-0.1 units): Millimeter-level precision
- **High Outlier Ratios** (>50%): Robust fitting maintains accuracy with RANSAC

## Python Tools

### Optional Analysis Scripts

The `python/` directory contains utilities for visualization and analysis:

**Installation:**
```bash
pip install numpy matplotlib opencv-python
```

**Available Tools:**
- Animation utilities in `python/animation/`
- Plotting and visualization in `python/plots/`
- Data analysis utilities in `python/utils/`
- Jupyter notebooks for interactive exploration: `python/demo.ipynb`

**Running Benchmarks with Python:**

```bash
# Run complete benchmark pipeline (C++ + Python analysis)
python run_benchmark.py --num_experiments 1000 --results_dir results --output_dir analysis_output

# Skip C++ build and just analyze existing results
python run_benchmark.py --skip_build --results_dir results --output_dir analysis_output
```

## Troubleshooting

### Build Issues

**Problem:** CMake cannot find Eigen3, OpenCV, or PCL

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev libopencv-dev libpcl-dev

# macOS
brew install eigen opencv pcl

# Windows - specify paths manually
cmake -DEIGEN3_INCLUDE_DIR="C:/path/to/eigen3" \
      -DOpenCV_DIR="C:/path/to/opencv/build" \
      -DPCL_DIR="C:/path/to/pcl/build" ..
```

**Problem:** OpenMP not found

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS (using Homebrew's llvm)
brew install libomp
export OpenMP_ROOT=$(brew --prefix libomp)
```

### Runtime Issues

**Problem:** Fitting fails with radius = 0 or returns -1

**Causes:**
- Insufficient points (need at least 3 points)
- Collinear or nearly-collinear points
- Numerical instability with poorly-conditioned data

**Solutions:**
- Use Robust CGA (`RobustFit3DCircle`) instead of baseline CGA
- Ensure points are well-distributed on the circle
- Check input data for NaN or infinite values
- Scale input data to reasonable magnitudes (avoid very large or very small coordinates)

**Problem:** RANSAC not finding inliers

**Solutions:**
- Increase iteration count (try 500-1000 for high outlier ratios)
- Adjust threshold to match your data scale
- Verify that true inliers exist in the dataset
- Check that minimum sample size (5 points) can form valid circles

### Performance Issues

**Problem:** RANSAC is too slow

**Solutions:**
- Reduce iteration count (balance between speed and accuracy)
- Downsample point cloud before fitting
- Use adaptive RANSAC termination criteria
- Consider parallel RANSAC implementation

### CI/CD Issues

**Problem:** GitHub Actions workflow fails with network connectivity errors

**Solution:**

The workflow has been updated to properly install dependencies. If you encounter network issues:

1. Check that the runner has access to package repositories
2. Verify no firewall blocks apt/brew/vcpkg repositories
3. For corporate environments, configure proxy settings:
   ```yaml
   env:
     HTTP_PROXY: http://proxy.example.com:8080
     HTTPS_PROXY: http://proxy.example.com:8080
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines

1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all benchmarks pass before submitting

### Code Style

- Use meaningful variable names
- Add comments for complex algorithms
- Follow C++11 or later standards
- Use const-correctness
- Prefer stack allocation over heap when possible

## Acknowledgements

This project builds upon excellent work from the open-source community:

- [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC) - Generic RANSAC implementation
- [RansacLib](https://github.com/tsattler/RansacLib) - Modern RANSAC library
- [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D) - Python RANSAC for 3D geometry

Special thanks to all contributors who have helped improve this project.

## References

### Academic Papers

- Förstner, W., & Wrobel, B. P. (2016). *Photogrammetric Computer Vision*. Springer.
- Li, H., & Hartley, R. (2006). "Five-Point Motion Estimation Made Easy." *ICPR*.
- Chernov, N., & Lesort, C. (2005). "Least Squares Fitting of Circles." *Journal of Mathematical Imaging and Vision*.

### Online Resources

- [Decoupled solution of 3D circle fitting](https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/)
- [Conformal Geometric Algebra Tutorial](http://www.geometricalgebra.net/)
- [PCL Documentation](https://pointclouds.org/documentation/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

1. **levinson** (xiahaa) - Lead Developer
2. **jun** (jjj) - Contributor
3. **lwc** (leovinchen) - Contributor

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ccenterfinder,
  author = {Levinson and Jun and Lwc},
  title = {CCenterFinder: Accurate 3D Circle Center Localization using CGA},
  year = {2023},
  url = {https://github.com/xiahaa/CCenterFinder}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Project Status:** Active Development

**Last Updated:** February 2026
