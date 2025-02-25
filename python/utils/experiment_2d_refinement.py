import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

from ellipse_center_refinement import *
from tqdm import tqdm, trange

def generate_camera_pose():
    R = np.eye(3)
    dR = cv.Rodrigues(np.random.randn(3,1)*0.4)[0]
    R = R@dR
    t = np.array([[1],[0],[0]])

    return R,t

def get_center_with_grid_search(outer, centers, K, marker_diameter):
    fs = []
    for c in centers:
        fc = eval_distance_f0(outer, c, K, marker_diameter)
        fs.append(fc)

    fs = np.asarray(fs)
    ids = np.argsort(fs)
    minimums = centers[ids[:2],:]
    scores = fs[ids[:2]]
    return centers, scores

def generate_points(r):
    theta = np.linspace(0,np.pi*2,200)
    P = np.vstack((r*np.cos(theta),r*np.sin(theta),np.zeros((1,200))))
    c3d = np.random.rand(3,1)*5
    P = P+c3d
    return P, c3d

def fit_ellipse(uv):
    ell = cv.fitEllipse(uv.T)
    ex=ell[0][0]
    ey=ell[0][1]
    ea=ell[1][0]*0.5
    eb=ell[1][1]*0.5
    etheta=ell[2]*np.pi/180.0

    theta = np.linspace(0, np.pi * 2, 200)
    R = np.array([[np.cos(etheta),-np.sin(etheta)],[np.sin(etheta),np.cos(etheta)]])
    ep = R@np.vstack((ea * np.cos(theta), eb * np.sin(theta))) + np.array([[ex],[ey]])

    poly = get_ellipse_polynomial_coeff(ell)
    return ex, ey, ea, eb, etheta, ep, poly

def compute_reprojection_error(P, R, t, K, uv):
    Pc=R@P+t
    Pc=Pc/Pc[-1,:]
    Pproj=K@Pc
    uvproj=Pproj[:2,:]
    err = uvproj-uv
    err = np.sqrt(err[0,:]**2+err[1,:]**2)
    avgerr = np.mean(err)
    return err, avgerr

def recover_pose(p3d, p2d, K, dist=np.zeros(5)):
    success, rvec, tvec, inliers = cv.solvePnPRansac(p3d,p2d,K,dist,flags=cv.SOLVEPNP_ITERATIVE, \
                                                     useExtrinsicGuess=False, iterationsCount=100,\
                                                     reprojectionError=8,confidence=0.99)
    R=cv.Rodrigues(rvec)[0]
    return R, tvec, inliers

def monte_carlo_experiment(search_ratio=0.1):
    fx=600
    K = np.array([[fx,0,640],[0,fx,480],[0,0,1]],dtype='float32')
    num=50
    R,t=generate_camera_pose()
    radius=2

    p3d=np.zeros((num,3),dtype='float32')
    p2d={}
    p2d['real']=np.zeros((num,2),dtype='float32')
    p2d['image']=np.zeros((num,2),dtype='float32')
    p2d['gt1']=np.zeros((num,2),dtype='float32')
    p2d['gt2']=np.zeros((num,2),dtype='float32')

    debug=False

    Rres={}
    tres={}
    inliers={}
    for i in range(num):
        P,c3d=generate_points(radius)
        Pc=R@P+t
        Pcn=Pc/Pc[-1,:]
        uvraw=K@Pcn
        uv=uvraw+np.vstack((np.random.randn(*uvraw[2,:].shape),np.zeros((1,uvraw.shape[1]))))

        uv=uv[:2,:].astype(dtype='float32')
        ex,ey,ea,eb,etheta,ep,poly=fit_ellipse(uv)

        img=np.zeros((960,1280,1),dtype='uint8')
        mask=cv.ellipse(img,(int(ex),int(ey)),(int(ea),int(eb)),etheta*180/np.pi,0,360,(255,255,255),-1)

        mm=cv.moments(mask)
        mmcx = mm['m10'] / mm['m00']
        mmcy = mm['m01'] / mm['m00']

        circular_center = R@c3d+t
        circular_center=circular_center/circular_center[-1,:]
        uv_circular_center = K@circular_center

        if debug:
            plt.figure(figsize=(10,10))
            plt.imshow(mask,cmap='gray')
            plt.xlim(0,1280)
            plt.ylim(0,960)
            plt.plot(uv[0,:],uv[1,:],'r.',label='input')
            plt.plot(ep[0, :], ep[1, :], 'r.', label='fitted')
            plt.plot(ex, ey, 'bo', label='fitted ellipse center')
            plt.plot(uv_circular_center[0, :], uv_circular_center[1, :], 'mx', label='projected circular center')
            plt.plot(mmcx, mmcy, 'yd', label='center of mass')
            plt.legend()
            plt.axis('off')
            plt.show()

        img = np.zeros((960,1280,1),dtype='uint8')
        mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),-1)
        if debug:
            plt.figure(figsize=(10,10))
            plt.imshow(mask,cmap='gray')
            plt.xlim(0,1280)
            plt.ylim(0,960)
            plt.plot(uv[0,:],uv[1,:],'r.',label='input')
            plt.plot(ex, ey, 'bo', label='fitted ellipse center')
            plt.plot(uv_circular_center[0, :], uv_circular_center[1, :], 'mx', label='projected circular center')
            plt.plot(mmcx, mmcy, 'yd', label='center of mass')
            plt.legend()
            plt.axis('off')
            plt.show()


        v_nz, u_nz, _=mask2.nonzero()
        if debug:
            score_map=np.zeros(mask2.shape,dtype='float32')
            for u, v in zip(u_nz, v_nz):
                score=eval_distance_f0(poly,np.array([u,v]),K,2*radius,4)
                score_map[v,u,:]=score
            min_score=score_map.min()
            max_score=score_map.max()
            score_map=(score_map-min_score)/(max_score-min_score)*255
            plt.imshow(score_map,cmap='jet')
            plt.plot(ex, ey, 'bo', label='fitted ellipse center')
            plt.plot(uv_circular_center[0, :], uv_circular_center[1, :], 'mx', label='projected circular center')
            plt.plot(mmcx, mmcy, 'yd', label='center of mass')
            plt.xlim(0, 1280)
            plt.ylim(0, 960)
            plt.colorbar(fraction=0.03,pad=0.05)
            plt.axis('off')
            plt.show()

        centers=np.vstack((u_nz,v_nz)).T

        minimums, scores=get_center_with_grid_search(poly,centers,K,2*radius)

        found_mc=None
        another_mc=None
        errs=[]
        for mc in minimums:
            err = (mc[0]-uv_circular_center[0])**2+(mc[1]-uv_circular_center[1])**2
            errs.append(err)
        if errs[0] > errs[1]:
            found_mc=minimums[1,:].reshape((2,))
            another_mc=minimums[0,:].reshape((2,))
        else:
            found_mc = minimums[0, :].reshape((2,))
            another_mc = minimums[1, :].reshape((2,))

        p3d[i,:]=c3d.T
        p2d['image'][i,:]=np.array([mmcx,mmcy])
        p2d['real'][i, :] = np.array([uv_circular_center[0,0], uv_circular_center[1,0]])
        p2d['gt1'][i, :] = np.array([found_mc[0], found_mc[1]])
        p2d['gt2'][i, :] = np.array([another_mc[0], another_mc[1]])

    Rres['real'], tres['real'], inliers['real']=recover_pose(p3d,p2d['real'],K)
    Rres['image'], tres['image'], inliers['image'] = recover_pose(p3d, p2d['image'], K)
    Rres['gt1'], tres['gt1'], inliers['gt1'] = recover_pose(p3d, p2d['gt1'], K)
    Rres['gt2'], tres['gt2'], inliers['gt2'] = recover_pose(p3d, p2d['gt2'], K)

    errs = {}
    avg_err={}
    for key in Rres.keys():
        errs[key], avg_err[key] = compute_reprojection_error(p3d[inliers[key],:].squeeze().T,
                                                             Rres[key],tres[key],p2d[inliers[key],:].squeeze().T)

    return avg_err, Rres, tres, R, t
