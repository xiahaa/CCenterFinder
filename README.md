# TODO:
1. ~~find relevant data;~~
2. ~~finish C++ ransac version;~~
3. ~~add python version;~~

# CCenterFinder

This repository contains our implementation for accurately localizing 3D and 2D circular centers, including:

1. A 3D circular center finding algorithm based on Conformal Algebra. We also provide a RANSAC variant to handle outliers.
2. An exact 2D circular center finding algorithm that uses grid search. To solve the two-fold ambiguity problem, we provide a feasible solution given two coplanar circles.
3. Demonstrations in automation and robotics applications showing how the proposed method can help improve accuracy.

## Dependencies

### For the C++ version:
1. Eigen3
2. OpenCV (REQUIRED for examples)
3. PCL (REQUIRED for examples)

### For the Python version:
1. NumPy
2. OpenCV (optional)
3. Matplotlib (optional)

### For RANSAC:
We borrow template code from: [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC/tree/master) and [Ransaclib](https://github.com/tsattler/RansacLib/tree/master)

## Data

To generate Monte-Carlo experimental data, modify the code to match your folder structure, e.g., `os.makedirs('/mnt/d/data/IROS/data/3d_experiment', exist_ok=True)`, and run:
```bash
python main_carlo_pcl_3d.py --output_folder /path/to/output_folder
```

## Usage

### Running the C++ version
Compile and run the benchmark:
```bash
g++ -o benchmark_circle_fitting_noise src/benchmark_circle_fitting_noise.cpp -I/path/to/eigen -I/path/to/opencv -I/path/to/pcl -L/path/to/libs -lopencv_core -lopencv_imgproc -lopencv_highgui -lpcl_common -lpcl_io -lpcl_segmentation
./benchmark_circle_fitting_noise [benchmark_choice]
```
`benchmark_choice` can be `0` for noise benchmark or `1` for span benchmark.

### Running the Python version
Generate data and run experiments:
```bash
python python/main_carlo_pcl_3d.py --output_folder /path/to/output_folder
```

### Demo

Here is a demo of using the RANSAC algorithm with `circle_cga.py`:

![Circle CGA RANSAC Demo](python/animation/pyransac3d/circle_cga.gif)

## Citation

If you use this code, please cite our work.

## Acknowledgement

We acknowledge the use of the following libraries:
- [GRANSAC](https://github.com/drsrinathsridhar/GRANSAC/tree/master)
- [Ransaclib](https://github.com/tsattler/RansacLib/tree/master)
- [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D)

## Contributors
1. levinson (xiahaa)
2. jun (jjj)
3. lwc (leovinchen)
