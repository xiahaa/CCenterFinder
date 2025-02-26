import numpy as np
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
# import mql_toolkits.mplot3d.axes3d as axes3d
# %matplotlib qt

from experiment_3d_pcl import  *

from tqdm import tqdm, trange


def simu_1():
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


def simu_2():
    # simu 2
    num_experiments=500
    noise_levels = np.array([1e-4,1e-3,1e-2,1e-1,1])

    # use tqdm
    pbar=tqdm(total=num_experiments*len(noise_levels))
    for noise_level in noise_levels:
        i=0
        while i < num_experiments:
            centers,pcn=monte_carlo_experiment(noise_level=noise_level,threshold=0.01,do_refinement=True,data_gen=True)

            # write out pcn to a file, the output folder is /mnt/d/data/IROS/data/3d_experiment
            import os
            os.makedirs('/mnt/d/data/IROS/data/3d_experiment',exist_ok=True)
            # write point line by line
            # output file name is composed of noise level and experiment id

            filename = '/mnt/d/data/IROS/data/3d_experiment/pcn_{:.6f}_{:04d}.txt'.format(noise_level,i)

            with open(filename,'w') as f:
                for j in range(pcn.shape[1]):
                    f.write(f'{pcn[0,j]} {pcn[1,j]} {pcn[2,j]}\n')

            # write out also the result to a file, line by line
            filename = '/mnt/d/data/IROS/data/3d_experiment/centers_{:.6f}_{:04d}.txt'.format(noise_level,i)
            np.savetxt(filename,np.array([centers['real'],centers['pcl'],centers['lsq']]))
            pbar.update(1)
            i+=1

def simu_3():

    def gen_data_span(span_angle, noise_level):
        r=2+np.random.rand(1)*3
        n=50
        theta=np.linspace(0,np.pi*span_angle/180.0,n).squeeze()
        p = np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros((1, n))))
        R = np.eye(3)
        dR = cv.Rodrigues(np.random.randn(3, 1))[0]
        R = R @ dR
        t = np.random.rand(3, 1) * 5
        pc = R @ p + t
        pcn = pc + np.random.randn(*pc.shape) * noise_level

        centers=dict()
        centers['pcl']=t.squeeze()
        centers['lsq']=t.squeeze()
        centers['real']=t.squeeze()
        return (centers,pcn)

    # simu 2
    num_experiments=500
    noise = 1e-1
    span_angle = np.array([90, 135, 180, 225, 270, 315, 360])
    # use tqdm
    pbar=tqdm(total=num_experiments*len(span_angle))
    for span in span_angle:
        i=0
        while i < num_experiments:
            centers,pcn=gen_data_span(span, noise)
            # write out pcn to a file, the output folder is /mnt/d/data/IROS/data/3d_experiment
            import os
            base_dir = '/mnt/d/data/IROS/data/3d_experiment_span'
            os.makedirs(base_dir,exist_ok=True)
            # write point line by line
            # output file name is composed of noise level and experiment id
            filename = os.path.join(base_dir, 'pcn_{:03d}_{:.6f}_{:04d}.txt'.format(span, noise, i))
            with open(filename,'w') as f:
                for j in range(pcn.shape[1]):
                    f.write(f'{pcn[0,j]} {pcn[1,j]} {pcn[2,j]}\n')

            # write out also the result to a file, line by line
            filename = os.path.join(base_dir, 'centers_{:03d}_{:.6f}_{:04d}.txt'.format(span, noise, i))
            np.savetxt(filename,np.array([centers['real'],centers['pcl'],centers['lsq']]))
            pbar.update(1)
            i+=1

if __name__=='__main__':
    simu_3()