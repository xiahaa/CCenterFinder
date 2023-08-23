import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

from ellipse_center_refinement import  *
from experiment_extrinsic_finding import  *

from tqdm import tqdm, trange
#%matplotlib qt

num_experiments=20
r_error=np.zeros((num_experiments,4))
t_error=np.zeros((num_experiments,4))
reprojection_error = np.zeros((num_experiments,4))

import sophus as sp

pbar=tqdm(total=num_experiments)
i=0
while i < num_experiments:
    try:
        Rinit, tinit, Rres, tres, R, t = monte_carlo_experiment()
        pbar.update(1)
    except:
        continue

    t=t.squeeze()

    errR = sp.SO3(R.T @ Rinit)
    r_error[i, 0] = np.linalg.norm(errR.log())
    t_error[i, 0] = np.linalg.norm(tinit - t)
    errR = sp.SO3(R.T @ Rres)
    r_error[i, 1] = np.linalg.norm(errR.log())
    t_error[i, 1] = np.linalg.norm(tres - t)

    i+=1

print("r_error:")
print(np.mean(r_error[:0]))
print(np.mean(r_error[:1]))
print("t_error:")
print(np.mean(t_error[:0]))
print(np.mean(t_error[:1]))


