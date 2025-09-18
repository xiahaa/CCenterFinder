import numpy as np
import cv2 as cv
import sophuspy as sp
from ellipse_center_refinement import get_ellipse_polynomial_coeff

def generate_camera_pose():
    """
    Generate a camera pose that produces a visibly tilted ellipse projection.

    Strategy:
    - Apply a significant rotation about X and Y (tilt) so the circle plane
      is not parallel to the image plane (avoids near-circular appearance).
    - Keep Z forward and translation such that the projection is within frame.
    """
    # Fixed, strong tilt about X and Y to ensure ellipse (not circle)
    rx, ry, rz = np.deg2rad([35.0, -25.0, 10.0])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    R = Rz @ Ry @ Rx

    # Translation: shift right and down a bit, and forward in Z
    # Push ellipse towards image edge and closer to camera so part may fall outside
    t = np.array([[1.6],
                  [-0.9],
                  [3.5]])

    return R, t

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

def recover_pose(p3d, p2d, K, dist=np.zeros(5), use_ransac=True):
    if use_ransac:
        success, rvec, tvec, inliers = cv.solvePnPRansac(p3d,p2d,K,dist,flags=cv.SOLVEPNP_ITERATIVE, \
                                                         useExtrinsicGuess=False, iterationsCount=100,\
                                                         reprojectionError=2,confidence=0.99)
        R=cv.Rodrigues(rvec)[0]
    else:
        success, rvec, tvec = cv.solvePnP(p3d,p2d,K,dist,flags=cv.SOLVEPNP_EPNP , \
                                         useExtrinsicGuess=False)
        R=cv.Rodrigues(rvec)[0]
        # set inliers to all indices
        inliers = np.arange(p3d.shape[0])
    return R, tvec, inliers

def compute_pose_error(Rres, tres, R, t):
    r_error = {}
    t_error = {}
    for _, key in enumerate(Rres.keys()):
        res_key = key
        errR=sp.SO3(R.T@Rres[key])
        r_error[res_key]=np.linalg.norm(errR.log())
        t_error[res_key]=np.linalg.norm(tres[key]-t)

    return r_error, t_error