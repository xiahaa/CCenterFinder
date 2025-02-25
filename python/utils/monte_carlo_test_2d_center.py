import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

from ellipse_center_refinement import  *
# from experiment_2d_refinement import  *
import find_rectify_homography as frh
from tqdm import tqdm, trange
#%matplotlib qt

N=1
res=np.zeros((N,),dtype='uint8')
res2=np.zeros((N,),dtype='uint8')

for i in tqdm(range(N),total=N):
    r=2
    theta = np.linspace(0, np.pi * 2, 200)
    P = np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros((1, 200))))
    P2 = np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros((1, 200))))
    P2 = P2+np.array([[4],[4],[0]])
    R = np.eye(3)
    dR = cv.Rodrigues(np.random.randn(3, 1) * 0.5)[0]
    R = R @ dR
    t = np.array([[1], [0], [8]])
    Pc = R @ P + t
    Pc2 = R @ P2 + t

    fx = 600
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')

    Pcn = Pc / Pc[-1, :]
    Pcn2 = Pc2 / Pc2[-1, :]

    uvraw = K @ Pcn
    uvraw2 = K @ Pcn2

    uv = uvraw + np.vstack((np.random.randn(*uvraw[2, :].shape), np.zeros((1, uvraw.shape[1]))))
    uv2 = uvraw2 + np.vstack((np.random.randn(*uvraw[2, :].shape), np.zeros((1, uvraw.shape[1]))))

    uv = uv[:2, :].astype(dtype='float32')
    uv2 = uv2[:2, :].astype(dtype='float32')
    ell = cv.fitEllipse(uv.T)
    ell2 = cv.fitEllipse(uv2.T)

    ex = ell[0][0]
    ey = ell[0][1]
    ea = ell[1][0] * 0.5
    eb = ell[1][1] * 0.5
    etheta = ell[2] * np.pi / 180.0

    # theta = np.linspace(0, np.pi * 2, 200)
    Re = np.array([[np.cos(etheta), -np.sin(etheta)], [np.sin(etheta), np.cos(etheta)]])
    ep = Re @ np.vstack((ea * np.cos(theta), eb * np.sin(theta))) + np.array([[ex], [ey]])

    ex2 = ell2[0][0]
    ey2 = ell2[0][1]
    ea2 = ell2[1][0] * 0.5
    eb2 = ell2[1][1] * 0.5
    etheta2 = ell2[2] * np.pi / 180.0

    # theta = np.linspace(0, np.pi * 2, 200)
    Re2 = np.array([[np.cos(etheta2), -np.sin(etheta2)], [np.sin(etheta2), np.cos(etheta2)]])
    ep2 = Re2 @ np.vstack((ea * np.cos(theta), eb * np.sin(theta))) + np.array([[ex2], [ey2]])

    #
    circular_center = R@np.array([[0],[0],[0]])+t
    circular_center = circular_center/circular_center[-1,:]
    uv_circular_center = K@circular_center

    poly = get_ellipse_polynomial_coeff(ell)
    center = np.array(err[0])

    new_center, centers = get_distance_with_gradient_descent(poly,center,1e-6,1e-3,1e-10,0,K,r*2)
    error = np.linalg.norm(uv_circular_center.squeeze()[:2]-new_center)

    found_real=False
    if error<2:
        found_real=True
        res[i]=1

    H,Q,T=frh.findHomography(ell, new_center.squeeze()[:2])
    eph=np.vstack((ep,np.ones((1,ep.shape[1]))))
    ep2h=np.vstack((ep2,np.ones((1,ep.shape[1]))))
    # transform
    epth=H@T@eph
    ep2th=H@T@ep2h
    epth=epth/epth[-1,:]
    ep2th = ep2th / ep2th[-1, :]

    ##
    cc=np.mean(epth,axis=1).reshape((3,1))
    dist=np.linalg.norm(epth-cc,axis=0)
    mean_redius1 = np.mean(dist)

    ##
    cc2 = np.mean(ep2th,axis=1).reshape((3,1))
    dist = np.linalg.norm(ep2th - cc2, axis=0)
    mean_redius2 = np.mean(dist)

    ratio = mean_redius1 / (mean_redius2+1e-8)

    if abs(ratio-1)>0.05:
        pass
    else:
        res2[i]=1

print("successful rate:{}".format(np.nonzero(res==res2)[0].shape[0]/N))
tp = np.nonzero(np.bitwise_and(res==1,res2==1))[0].shape[0]
fp = np.nonzero(np.bitwise_and(res==0,res2==1))[0].shape[0]
tn = np.nonzero(np.bitwise_and(res==0,res2==0))[0].shape[0]
fn = np.nonzero(np.bitwise_and(res==1,res2==0))[0].shape[0]
print("precision:{}, recall:{}".format((tp/(tp+fp),tp/(tp+fn))))
print("accuracy:{}".format((tp+tn)/(tp+tn+fp+fn)))

