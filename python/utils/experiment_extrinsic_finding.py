import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

from ellipse_center_refinement import *
from tqdm import tqdm,trange

from joblib import Parallel, delayed
import time

# disable warnings
import warnings
warnings.filterwarnings("ignore")

def generate_camera_pose():
    R = np.eye(3)
    dR = cv.Rodrigues(np.random.randn(3, 1)*0.4)[0]
    R = R @ dR
    t = np.array([[1],[0],[8]])

    return R,t

def generate_points(r):
    # r = 2 + np.random.rand(1) * 3
    n = 200
    theta = np.linspace(0, np.pi * 2, n).squeeze()
    p = np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros((1, n))))
    c3d=np.random.rand(3,1)*5
    p=p+c3d
    return p,c3d

def fit_ellipse(uv):
    ell = cv.fitEllipse(uv.T)
    ex = ell[0][0]
    ey = ell[0][1]
    ea = ell[1][0] * 0.5
    eb = ell[1][1] * 0.5
    etheta = ell[2] * np.pi / 180.0

    theta = np.linspace(0, np.pi * 2, 200)
    R = np.array([[np.cos(etheta), -np.sin(etheta)], [np.sin(etheta), np.cos(etheta)]])
    ep = R @ np.vstack((ea * np.cos(etheta), eb * np.sin(etheta))) + np.array([[ex], [ey]])

    poly = get_ellipse_polynomial_coeff(ell)
    return ex, ey, ea, eb, etheta, ep, poly

def recover_pose(p3d, p2d, K, dist=np.zeros(5)):
    success, rvec, tvec, inliers = cv.solvePnPRansac(p3d,p2d,K,dist,flags=cv.SOLVEPNP_ITERATIVE, \
                                                     useExtrinsicGuess=False, iterationsCount=100,\
                                                     reprojectionError=8,confidence=0.99)
    R=cv.Rodrigues(rvec)[0]
    return R, tvec, inliers

def evaluate_cost(outers, centers, K, marker_diameter):
    fs=[]
    for outer, c in zip(outers,centers):
        c = c.reshape((3,))

        fc=eval_distance_f0(outer,c,K,marker_diameter)
        fs.append(fc)
    fs=np.array(fs)
    cost=np.sum(fs)
    return cost

def get_center_with_grid_search(polys, p3d, rpyt_samples, K, marker_diameter):
    costs=[]
    for rpyt in rpyt_samples:
        rt=RpytToRt(rpyt[:3],rpyt[3:6])
        R=rt[:3,:3]
        t=rt[:3,3]
        centers=[]
        for c3d in p3d:
            circular_center=R@c3d.reshape((3,1))+t.reshape((3,1))
            circular_center=circular_center/circular_center[-1,:]
            uv_circular_center=K@circular_center
            centers.append(uv_circular_center)
        centers=np.array(centers)
        cost=evaluate_cost(polys,centers,K,marker_diameter)
        costs.append(cost)
    costs=np.array(costs)
    ids=np.argsort(costs)
    minimums=rpyt_samples[ids[:2],:]
    scores=costs[ids[:2]]
    return minimums, scores

def func_single(polys, p3d, rpyt, K, marker_diameter):
    rt = RpytToRt(rpyt[:3], rpyt[3:6])
    R = rt[:3, :3]
    t = rt[:3, 3]
    centers = []
    for c3d in p3d:
        circular_center = R @ c3d.reshape((3, 1)) + t.reshape((3, 1))
        circular_center = circular_center / circular_center[-1, :]
        uv_circular_center = K @ circular_center
        centers.append(uv_circular_center)
    centers = np.array(centers)
    cost = evaluate_cost(polys, centers, K, marker_diameter)
    return cost

def get_centers_with_grid_search_parallel(polys, p3d, rpyt_samples, K, marker_diameter):
    # show progress
    print("Searching for centers with grid search...")
    results = Parallel(n_jobs=-1)(delayed(func_single)(polys,p3d,rpyt,K,marker_diameter) for rpyt in tqdm(rpyt_samples))
    costs=np.array(results)
    ids = np.argsort(costs)
    minimums = rpyt_samples[ids[:2], :]
    scores = costs[ids[:2]]
    return minimums, scores

def compute_reprojection_error(P, R, t, K, uv):
    Pc=R@P+t
    Pc=Pc/Pc[-1,:]
    Pproj=K@Pc
    uvproj=Pproj[:2,:]
    err = uvproj-uv
    err = np.sqrt(err[0,:]**2+err[1,:]**2)
    avgerr = np.mean(err)
    return err, avgerr

def RtToRpyt(rt):
    n = rt[:,0]
    o = rt[:,1]
    a = rt[:,2]
    y = np.arctan2(n[1],n[0])
    p = np.arctan2(-n[2], n[0]*np.cos(y)+n[1]*np.sin(y))
    r = np.arctan2(a[0]*np.sin(y)-a[1]*np.cos(y),-o[0]*np.sin(y)+o[1]*np.cos(y))
    rpy = [r*180.0/np.pi,p*180.0/np.pi,y*180.0/np.pi]
    rpy=np.array(rpy).reshape((3,1))
    t=np.zeros((3,1))
    t[0]=rt[0,3]
    t[1] = rt[1, 3]
    t[2] = rt[2, 3]
    return rpy, t

def RpytToRt(rpy, t):
    rpy = rpy.reshape((3,))
    t = t.reshape((3,))
    rpy = rpy * 180.0 / np.pi
    Rz = np.array([np.cos(rpy[2]), -np.sin(rpy[2]), 0, np.sin(rpy[2]), np.cos(rpy[2]), 0, 0,0,1]).reshape((3,3))
    Ry = np.array([np.cos(rpy[1]), 0, np.sin(rpy[1]), 0,1,0, -np.sin(rpy[1]), 0, np.cos(rpy[1])]).reshape((3,3))
    Rx = np.array([1,0,0, 0, np.cos(rpy[0]), -np.sin(rpy[0]), 0, np.sin(rpy[0]), np.cos(rpy[0])]).reshape((3,3))
    R = Rz@Ry@Rx
    rt=np.eye(4)
    rt[:3,:3]=R
    rt[:3,3]=t.reshape((3,))
    return rt

def func(polys, p3d, x, K, marker_diameter):
    rt=RpytToRt(x[:3],x[3:6])
    R=rt[:3,:3]
    t=rt[:3,3]
    centers=[]
    for c3d in p3d:
        circular_center = R @ c3d.reshape((3, 1)) + t.reshape((3, 1))
        circular_center = circular_center / circular_center[-1, :]
        uv_circular_center = K @ circular_center
        centers.append(uv_circular_center)
    centers = np.array(centers)
    cost = evaluate_cost(polys, centers, K, marker_diameter)
    return cost

def get_rt_with_gradient_descent(polys, p3d, x0, K, marker_diameter, step, plambda, tolx, tolfun, verbose=True):
    x = x0.copy()
    it = 1
    xs = []
    xs.append(x0)
    while True:
        g = np.zeros((6,))
        for i in range(6):
            xplus = x.copy()
            xmius = x.copy()
            xplus[i]+=step
            xmius[i]-=step
            #
            fplus=func(polys,p3d,xplus,K,marker_diameter)
            fplus=func(polys,p3d,xmius,K,marker_diameter)

            g[i] = (fplus - fmius) / 2 / step

        f = func(polys, p3d, x, K, marker_diameter)

        is_reduce = False
        step_down = False

        while (plambda * np.linalg.norm(g) > tolx and step_down is False):
            newx = x - plambda * g
            newf = func(polys, p3d, newx, K, marker_diameter)
            if newf < f:
                x = newx.copy()
                f = newf.copy()
                if is_reduce is False:
                    plambda *= 2
                step_down = True
            else:
                plambda /= 2
                is_reduce = True

        if step_down is False:
            if verbose:
                logging.INFO("plambda * norm(g): {}".format(plambda * np.linalg.norm(g)))
                logging.INFO("cannot reduce!")
            break

        if abs(f - newf) < tolfun:
            if verbose:
                logging.INFO("exit because of tolfunc")
            break
        xs.append(x)
        it += 1

        if it > 1e5:
            logging.INFO("exit because of max iteration!")
            break
    if verbose:
        logging.INFO("after {} iterations!".format(it))
    return (x, xs)

def monte_carlo_experiment(search_range=3,search_step=1):
    fx = 600
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')
    num = 50
    R, t = generate_camera_pose()
    radius = 2

    p3d = np.zeros((num, 3), dtype='float32')
    polys=np.zeros((num,6), dtype='float32')

    debug=False

    Rres = {}
    tres = {}
    inliers = {}
    for i in range(num):
        P, c3d = generate_points(radius)
        Pc = R @ P + t
        Pcn = Pc / Pc[-1, :]
        uvraw = K @ Pcn
        uv = uvraw + np.vstack((np.random.randn(*uvraw[:2, :].shape), np.zeros((1, uvraw.shape[1]))))

        uv = uv[:2, :].astype(dtype='float32')
        ex, ey, ea, eb, etheta, ep, poly = fit_ellipse(uv)

        p3d[i,:]=c3d.T
        polys[i,:]=poly.reshape((1,6))

    rt = np.block([[R,t],[0,0,0,1]])
    real_rpy, real_t=RtToRpyt(rt)

    init_rpy = real_rpy
    init_t = real_t

    init_rpy[0]+=2
    init_rpy[1] += 2
    init_rpy[2] += -1
    init_t[0] += 1
    init_t[1] += -1
    init_t[2] += 2

    rpy_samples=[]
    for r_search in np.arange(-search_range,search_range,search_step):
        for p_search in np.arange(-search_range, search_range, search_step):
            for y_search in np.arange(-search_range, search_range, search_step):
                for x_search in np.arange(-search_range, search_range, search_step):
                    for y_search in np.arange(-search_range, search_range, search_step):
                        for z_search in np.arange(-search_range, search_range, search_step):
                            rpy_samples.append(np.array([init_rpy[0]+r_search,\
                                                        init_rpy[1]+p_search,\
                                                        init_rpy[2]+y_search,\
                                                        init_t[0]+x_search,\
                                                        init_t[1]+y_search,\
                                                        init_t[2]+z_search]))

    rpy_samples=np.array(rpy_samples)

    tic=time.time()
    minimums, scores = get_centers_with_grid_search_parallel(polys,p3d,rpy_samples,K,2*radius)
    toc=time.time()
    print("time spent:{}".format(toc-tic))

    found_mc = None
    another_mc = None
    errs = []
    for mc in minimums:
        rtres = RpytToRt(mc[:3],mc[3:6])
        Rres=rtres[:3,:3]
        tres=rtres[:3,3]
        break

    rt_init=RpytToRt(init_rpy,init_t)

    import sophuspy as sp
    Rinit = rt_init[:3,:3]
    tinit = rt_init[:3,3]
    errR1 = sp.SO3(R.T@Rres)
    rotation_error_init = np.linalg.norm(errR1.log())

    errR2 = sp.SO3(R.T@Rres)
    rotation_error_refine = np.linalg.norm(errR2.log())

    print("rotation_error_init:")
    print(rotation_error_init)
    print("rotation_error_refine:")
    print(rotation_error_refine)

    return Rinit, tinit, Rres, tres, R, t


if __name__=='__main__':
    monte_carlo_experiment()
