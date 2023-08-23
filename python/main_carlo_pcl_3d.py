import numpy as np
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
# import mql_toolkits.mplot3d.axes3d as axes3d
# %matplotlib qt

from experiment_3d_pcl import  *

from tqdm import tqdm, trange

num_experiments=1000
center_error = np.zeros((num_experiments,2))

pbar=tqdm(total=num_experiments)
i=0
while i < num_experiments:
    centers,_=monte_carlo_experiment(noise_level=0.01,threshold=0.01,do_refinement=True)
    center_real = centers['real']
    center_pcl=centers['pcl']
    center_lsq=centers['lsq']
    pbar.update(1)
    center_error[i,0]=np.linalg.norm(center_real-center_pcl)
    center_error[i,1]=np.linalg.norm(center_real-center_lsq)
    i+=1

print('center error')
print(np.mean(center_error[:,0]))
print(np.mean(center_error[:,1]))


