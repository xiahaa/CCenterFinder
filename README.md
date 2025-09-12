# CCenterFinder

Accurate localization of 3D and 2D circle centers.

- 3D circle fitting via Conformal Geometric Algebra (CGA)
- Robust CGA (normalization + stable eigen selection) and RANSAC baselines
- PCL Circle3D (RANSAC) baseline for comparison
- 2D exact circle center via search (for reference)
- Unified test program (`simple_test`) with multiple modes for easy testing

## Dependencies

### C++
- Eigen3
- OpenCV (examples/IO)
- PCL (PCL RANSAC baseline)

### Python (optional, analysis/demos)
- NumPy, Matplotlib, OpenCV

### RANSAC templates
Code in `include/rtl/` adapted from: [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC) and [RansacLib](https://github.com/tsattler/RansacLib)

## Build (CMake)

### Windows (PowerShell)
```powershell
cd D:\data\CCenterFinder
mkdir build; cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### Linux
```bash
cd /path/to/CCenterFinder
mkdir build && cd build
cmake ..
make -j
```

## Executables

- `simple_test`: Unified test program with three modes (generated data, given data, RANSAC)
- `monte_carlo_benchmark`: Monte Carlo comparison across scenarios (noise, arcs, sparsity, symmetric non-uniform)
- `benchmark_circle_fitting_noise`: simple noise benchmark
- `benchmark_outlier`: outlier-robustness benchmark (CGA vs PCL)
- `test_fit3d`: quick sanity check (original vs robust CGA on synthetic data)

Build targets (after configure):
```powershell
cmake --build build --config Release --target simple_test
cmake --build build --config Release --target monte_carlo_benchmark
cmake --build build --config Release --target benchmark_outlier
cmake --build build --config Release --target test_fit3d
```

## How we compute fits

### Original CGA (C++)
`include/Fit3DCircle.hpp` implements the baseline CGA method (no pre-normalization).

### Robust CGA (C++)
`include/RobustFit3DCircle.hpp` improves numerical stability:
- Center points and scale so RMS ≈ √2 before forming the CGA system
- Select two smallest positive eigenvalues; fallback to two smallest
- Recover circle parameters; then unscale and uncenter outputs

Use:
```cpp
Eigen::Vector3d center; double radius; Eigen::Vector3d normal;
robust_cga::RobustFit3DCircle::Fit(points, center, radius, &normal);
```

## RANSAC Implementation

The project uses a custom RANSAC implementation based on the RTL (RANSAC Template Library) framework:

- **RTL Framework**: Located in `include/rtl/` - adapted from [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC) and [RansacLib](https://github.com/tsattler/RansacLib)
- **Circle3D Estimator**: Implements the RTL::Estimator interface for 3D circle fitting
- **Robust CGA Integration**: Uses Robust CGA for model computation within RANSAC
- **Distance Metric**: 3D point-to-circle distance considering both plane and edge distances

## Outlier benchmark

- Source: `src/benchmark_outlier.cpp`
- CGA path uses RANSAC (`include/rtl`) with an estimator powered by Robust CGA, then refits on inliers.
- PCL path uses `pcl::SACSegmentation<pcl::PointXYZ>` with `SACMODEL_CIRCLE3D`.

Run (example):
```powershell
cmake --build build --config Release --target benchmark_outlier
.\n+build\Release\benchmark_outlier.exe
```

Tips:
- Tune RANSAC thresholds (`find_circle_cga` uses tolerance ~0.1–0.2) based on your data scale.
- Increase iterations for high outlier ratios.

## Monte Carlo benchmark

Source: `src/monte_carlo_benchmark.cpp`

Runs 4 scenarios (isotropic noise; limited arcs; sparse non-uniform; symmetric non-uniform). Outputs per-scenario text results for analysis.

Run:
```powershell
cmake --build build --config Release --target monte_carlo_benchmark
build\Release\monte_carlo_benchmark.exe 1000 results
```

## Simple Test Program

`src/simple_test.cpp` provides a unified test program with three modes:

### Modes
1. **Generated** (default): Creates synthetic 3D circle data and fits it
2. **Given**: Uses predefined test data from the code
3. **RANSAC**: Creates noisy data with outliers and uses RANSAC for robust fitting

### Usage
```powershell
# Build the program
cmake --build build --config Release --target simple_test

# Run with different modes
build\Release\simple_test.exe                    # Generated mode (default)
build\Release\simple_test.exe generated          # Explicit generated mode
build\Release\simple_test.exe given              # Use predefined test data
build\Release\simple_test.exe ransac             # RANSAC mode

# RANSAC mode with custom parameters
build\Release\simple_test.exe ransac --threshold 0.1 --iterations 300 --noise 0.01 --outliers 30

# Show help
build\Release\simple_test.exe --help
```

### RANSAC Parameters
- `--threshold <value>`: RANSAC distance threshold (default: 0.1)
- `--iterations <value>`: Number of RANSAC iterations (default: 300)
- `--noise <value>`: Gaussian noise standard deviation (default: 0.01)
- `--outliers <value>`: Number of outlier points to add (default: 30)

## Quick test

`src/test_fit3d` prints original vs robust CGA center/radius errors on synthetic data.

```powershell
cmake --build build --config Release --target test_fit3d
build\Release\test_fit3d.exe
```

## Python demos (optional)

Some utilities live under `python/`. Install basics:
```bash
pip install numpy matplotlib opencv-python
```

## Acknowledgement

- [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC)
- [RansacLib](https://github.com/tsattler/RansacLib)
- [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)

## Reference
- [Decoupled solution of 3D circle fitting](https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/)

## Contributors
1. levinson (xiahaa)
2. jun (jjj)
3. lwc (leovinchen)
