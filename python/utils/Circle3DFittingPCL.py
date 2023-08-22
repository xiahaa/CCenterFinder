import numpy as np
from scipy.optimize import least_squares

def compute_model_coeff(pcd):
    if pcd.shape[0] < pcd.shape[1]:
        pcd=pcd.T

    p0=pcd[:,0]
    p1=pcd[:,1]
    p2=pcd[:,2]

    helper_v01 = p0-p1
    helper_v02 = p0-p2
    helper_v10 = p1-p0
    helper_v12 = p1-p2
    helper_v20 = p2-p0
    helper_v21 = p2-p1

    common_helper_vec=np.cross(helper_v01,helper_v12)

    common_dividend=2*np.linalg.norm(common_helper_vec)**2

    alpha = (np.linalg.norm(helper_v12) ** 2 * helper_v01.dot(helper_v02)) / common_dividend
    beta = (np.linalg.norm(helper_v02) ** 2 * helper_v10.dot(helper_v12)) / common_dividend
    gamma = (np.linalg.norm(helper_v01) ** 2 * helper_v20.dot(helper_v21)) / common_dividend

    circle_center = alpha*p0+beta*p1+gamma*p2

    circle_radius_vec = circle_center-p0
    circle_radius = np.linalg.norm(circle_radius_vec)
    circlr_norm=common_helper_vec/np.linalg.norm(common_helper_vec)

    model_coeff=np.zeros((7,))

    model_coeff[:3]=circle_center.squeeze()
    model_coeff[3] = circle_radius
    model_coeff[4:] = circlr_norm.squeeze()

    return model_coeff

def get_distance_to_model(pcd, model_coeff):
    C=model_coeff[:3].copy()
    N=model_coeff[4:].copy()
    r=model_coeff[3]

    distances=[]
    for i, p in enumerate(pcd):
        helper_vec=p-C
        clambda=helper_vec.dot(N)/np.linalg.norm(N)**2
        p_proj = p+clambda*N
        helper_vec_2 = p_proj-C
        K=C+r*helper_vec_2/np.linalg.norm(helper_vec_2)
        distance_vec=p-K
        distances.append(np.linalg.norm(distance_vec))

    distances=np.array(distances).squeeze()

    return distances

def cost_func(x,pcd):
    C=np.array([x[0],x[1],x[2]]).squeeze()
    N=np.array([x[4],x[5],x[6]]).squeeze()
    r=x[3]
    fvec=np.zeros((pcd.shape[1],))
    for i,pt in enumerate(pcd.T):
        P=pt-C
        helper_vec = P-C
        plambda = -helper_vec.dot(N) / np.linalg.norm(N) ** 2
        p_proj = P + plambda * N
        helper_vec_2 = p_proj - C
        K = C + r * helper_vec_2 / np.linalg.norm(helper_vec_2)
        distance_vec = P - K
        fvec[i] = (np.linalg.norm(distance_vec))

    return fvec

def optimize_model_coeff(pcd, inliers, model_coeff):
    optimized_coeff = model_coeff.copy()
    res=least_squares(cost_func,optimized_coeff,method='lm',args=(pcd[:,inliers],))

    return res.x if res.success else model_coeff

if __name__=='__main__':
    import cv2 as cv
    r=4
    n=50
    theta=np.linspace(0,np.pi*2,n)
    p=np.vstack((r*np.cos(theta),r*np.sin(theta),np.zeros((1,n))))
    R=np.eye(3)
    dR=cv.Rodrigues(np.random.randn(3,1))[0]
    R=R@dR
    t=np.random.rand(3,1)*5
    pc=R@p+t
    pcn=pc+np.random.randn(*pc.shape)*0.5

    randomid = np.arange(0,pcn.shape[1])
    np.random.shuffle(randomid)

    model_coeff=compute_model_coeff(pcn[:,randomid[:3]])
    distances = get_distance_to_model(pcn,model_coeff)
    inlier_id=np.nonzero(distances<1)[0]

    optimize_model_coeff = optimize_model_coeff(pcn, inlier_id, model_coeff)

    import lsq_3d_circle as lsqfit

    res_lsq = lsqfit.lsq_fit_3d_circle(pcn)
    center = res_lsq['center']

    # some drawing function


