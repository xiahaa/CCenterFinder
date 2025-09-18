import numpy as np
import os
import sys
from utility import recover_pose, compute_reprojection_error

def ransac_validation(p3: np.ndarray, p2: np.ndarray, p2_alt: np.ndarray, K: np.ndarray):
    # p3: nx3 array
    # p2: nx2 array
    # p2_alt: nx2 array
    # K: 3x3 array
    # return the inliers and the outliers
    p3 = p3.T
    p2 = p2.T
    p2_alt = p2_alt.T

    max_iter = 1000
    num_points = p3.shape[1]
    inlier_threshold = 8
    min_sample_size = 4

    # inliers = []
    # outliers = []
    # best_inliers = []
    best_R = None
    best_t = None
    best_reprojection_error = np.inf
    for i in range(max_iter):
        # first pick 4 index from 0:num_points-1
        sample = np.random.choice(num_points, min_sample_size, replace=False)
        p3_sample = p3[:,sample]
        # for p2_sample, we toss to randomly select from p2 or p2_alt
        p2_sample = np.zeros((2,min_sample_size))
        for j in range(min_sample_size):
            if np.random.rand() < 0.5:
                p2_sample[:,j] = p2[:,sample[j]]
            else:
                p2_sample[:,j] = p2_alt[:,sample[j]]
        # fit a model
        R, t, _=recover_pose(p3_sample.T, p2_sample.T, K, use_ransac=False)
        # compute the inliers
        p3cam = R@p3+t
        p3cam = p3cam/p3cam[-1,:]
        p2cam = K@p3cam
        p2cam = p2cam[:2,:]
        # compute the inliers
        reprojection_error = 0
        for j in range(num_points):
            error1 = np.linalg.norm(p2cam[:,j]-p2[:,j])
            error2 = np.linalg.norm(p2cam[:,j]-p2_alt[:,j])
            if error1 < error2:
                error = error1
            else:
                error = error2
            reprojection_error += error

        reprojection_error = reprojection_error/num_points

        if reprojection_error < best_reprojection_error:
            best_R = R
            best_t = t
            best_reprojection_error = reprojection_error

    if best_R is not None:
        # use best_R and best_t to compute the inliers
        p3cam = best_R@p3+best_t
        p3cam = p3cam/p3cam[-1,:]
        p2cam = K@p3cam
        p2cam = p2cam[:2,:]
        p2inliers = np.zeros((2,num_points))
        for j in range(num_points):
            error1 = np.linalg.norm(p2cam[:,j]-p2[:,j])
            error2 = np.linalg.norm(p2cam[:,j]-p2_alt[:,j])
            if error1 < error2:
                p2inliers[:,j] = p2[:,j]
            else:
                p2inliers[:,j] = p2_alt[:,j]

        # refine the pose
        Rres, tres, inliers = recover_pose(p3.T, p2inliers.T, K, use_ransac=True, ransac_threshold=inlier_threshold)

        valid3d = p3[:,inliers.squeeze()].T
        valid2d = p2inliers[:,inliers.squeeze()].T

        errs, avg_err = compute_reprojection_error(valid3d.T,
                                                             Rres, tres, K, valid2d.T)

        return Rres, tres, avg_err, p2inliers

    else:
        return None, None, None, None
