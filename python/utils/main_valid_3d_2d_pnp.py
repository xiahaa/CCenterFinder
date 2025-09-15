import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

from ellipse_center_refinement import  *
from experiment_2d_refinement import  *

from tqdm import tqdm, trange
#%matplotlib qt

num_experiments=100
r_error=np.zeros((num_experiments,4))
t_error=np.zeros((num_experiments,4))
reprojection_error = np.zeros((num_experiments,4))

import sophuspy as sp

pbar=tqdm(total=num_experiments)
i=0
while i < num_experiments:
    try:
        avg_err, Rres, tres, R, t = monte_carlo_experiment(num = 20, search_ratio=0.3)
        pbar.update(1)
    except:
        i+=1
        continue

    for j, key in enumerate(Rres.keys()):
        errR=sp.SO3(R.T@Rres[key])
        r_error[i,j]=np.linalg.norm(errR.log())
        t_error[i,j]=np.linalg.norm(tres[key]-t)
        reprojection_error[i,j]=avg_err[key]
    i+=1

print(Rres.keys())
print("r_error:")
print(np.mean(r_error[:0]))
print(np.mean(r_error[:1]))
print(np.mean(r_error[:2]))
print(np.mean(r_error[:3]))
print("t_error:")
print(np.mean(t_error[:0]))
print(np.mean(t_error[:1]))
print(np.mean(t_error[:2]))
print(np.mean(t_error[:3]))
print("rp_error:")
print(np.mean(reprojection_error[:0]))
print(np.mean(reprojection_error[:1]))
print(np.mean(reprojection_error[:2]))
print(np.mean(reprojection_error[:3]))
