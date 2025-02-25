# TODO:
1. ~~find relevant data;~~
2. ~~finish C++ ransac version;~~
3. ~~add python version;~~

# CCenterFinder
This repository contains our implementation on accurately localizing 3D and 2D circular centers, including:

1. A 3D circular center finding algorithm based on Conformal Algebra. We also provide a RANSAC variant so that it can handle  outliers.
2. An exact 2D circular center finding algorithm that uses grid search. To solve the two-fold ambiguity problem, we give a feasible solution given two coplanar circles.
3. We show in some automation and robotics application user cases that the proposed method can help users improve accuracy.

# Dependencies
For the C++ version:
1. Eigen3
2. OpenCV (REQUIRED for examples)
3. PCL (REQUIRED for examples)

For the Python version:
1. NumPy
2. OpenCV (optional)
3. Matplotlib (optional)

# Data
To generate Monte-Carlo experimental data, change codes like `os.makedirs('/mnt/d/data/IROS/data/3d_experiment',exist_ok=True)` according to your folder structure, run
```bash
python main_carlo_pcl_3d.py 
```
todo: add argparser to specify output folder for generated data.


# Usage

# Citation

# Acknowlegement 

[GRANSAC](https://github.com/drsrinathsridhar/GRANSAC/tree/master)

[Ransaclib](https://github.com/tsattler/RansacLib/tree/master)

# Contributors
1. levinson (xiahaa)
2. jun (jjj)
3. lwc (leovinchen)
