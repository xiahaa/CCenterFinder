import dis
import numpy as np
import os
import sys
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ellipse_center_refinement import *
from typing import Optional
import find_rectify_homography as frh

def select_real_mc_using_homograph(ell, poly, ep, ep_alt, K, radius, nominal_ratio=1.0):
    # do grid search on ell to find out two candidates
    found_mc, another_mc = grid_search_refinement(ell, poly, K, radius*2, search_ratio=0.5)

    min_ratio_error = np.inf
    real_mc = None

    ex,ey,ea,eb,etheta = ell
    ell_cv = [[ex,ey],[ea*2,eb*2],etheta*180.0/np.pi]
    for mc in [found_mc, another_mc]:
        H,Q,T=frh.findHomography(ell_cv, mc.squeeze()[:2])
        # to homogeneous coordinates
        eph=np.vstack((ep,np.ones((1,ep.shape[1]))))
        eph_alt=np.vstack((ep_alt,np.ones((1,ep_alt.shape[1]))))
        # transform
        epth=H@T@eph
        epth_alt=H@T@eph_alt
        # to non-homogeneous coordinates
        epth=epth/epth[-1,:]
        epth_alt = epth_alt / epth_alt[-1, :]

        ##
        cc=np.mean(epth,axis=1).reshape((3,1))
        dist=np.linalg.norm(epth-cc,axis=0)
        mean_redius = np.mean(dist)

        ##
        cc_alt = np.mean(epth_alt,axis=1).reshape((3,1))
        dist = np.linalg.norm(epth_alt - cc_alt, axis=0)
        mean_redius_alt = np.mean(dist)

        ratio = mean_redius / (mean_redius_alt+1e-8)

        if abs(ratio-nominal_ratio)<min_ratio_error:
            min_ratio_error = abs(ratio-nominal_ratio)
            real_mc = mc

    # return also the false_found_mc as the one further to the real_mc
    false_found_mc = None
    if np.linalg.norm(found_mc-real_mc)>np.linalg.norm(another_mc-real_mc):
        false_found_mc = found_mc
    else:
        false_found_mc = another_mc

    return real_mc, false_found_mc

def generate_masked_points(ell: np.ndarray, search_ratio: float, K: Optional[np.ndarray] = None) -> np.ndarray:
    ex, ey, ea, eb, etheta = ell

    a = ea*search_ratio
    b = eb*search_ratio

    # generate a grid of points inside the bbox spanned by ea*search_ratio and eb*search_ratio
    x = np.arange(-a, a, 0.1)
    y = np.arange(-b, b, 0.1)
    xx, yy = np.meshgrid(x, y)

    # find out the points inside the ellipse
    inside = (xx**2/a**2 + yy**2/b**2) <= 1
    final_points = np.vstack((xx[inside], yy[inside])).T

    # rotate the points by etheta
    R = np.array([[np.cos(etheta), -np.sin(etheta)], [np.sin(etheta), np.cos(etheta)]])
    final_points = R @ final_points.T
    final_points = final_points.T

    # translate the points by ex and ey
    final_points = final_points + np.array([ex, ey])

    # to integer and remove duplicates
    final_points = np.round(final_points).astype(int)
    final_points = np.unique(final_points, axis=0)

    return final_points

def grid_search_refinement_cv(ell: np.ndarray, poly: np.ndarray, K:np.ndarray, radius:float, search_ratio:float=0.5):
    ex, ey, ea, eb, etheta = ell
    width = int(K[0,2]*2)
    height = int(K[1,2]*2)
    img=np.zeros((height,width,1),dtype='uint8')
    mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),-1)
    v_nz, u_nz, _=mask2.nonzero()
    # score map will be computed in the combined 1x2 figure below when debug
    candidates =np.vstack((u_nz,v_nz)).T
    scores = []
    for c in candidates:
        fc = eval_distance_f0(poly, c, K, 2*radius)
        scores.append(fc)

    scores = np.asarray(scores)

    all_centers=candidates
    all_scores=scores

    picked = select_minima_with_suppression(all_centers, all_scores, k=2, suppress_radius=15)
    # Fallback if suppression returns less than 2
    if len(picked) < 2:
        ids = np.argsort(np.asarray(all_scores).reshape(-1))[:2]
        picked = [all_centers[i] for i in ids]
    minimums = np.vstack(picked)

    found_mc = minimums[0, :].reshape((2,))
    another_mc = minimums[1, :].reshape((2,))

    return found_mc, another_mc

def check_point_generation(ell: np.ndarray, search_ratio: float, K: np.ndarray, ep: np.ndarray):
    ex, ey, ea, eb, etheta = ell
    width = int(K[0,2]*2)
    height = int(K[1,2]*2)
    img=np.zeros((height,width,1),dtype='uint8')
    mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),-1)
    v_nz, u_nz, _=mask2.nonzero()
    # score map will be computed in the combined 1x2 figure below when debug
    points_gen_1 =np.vstack((u_nz,v_nz)).T

    points_gen_2=generate_masked_points(ell, search_ratio)

    # plot both points_gen_1 and points_gen_2 in the image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    # mask3 = cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),1)
    # draw the ep
    plt.scatter(ep[0,:], ep[1,:], color='green', marker='.', label='ep', alpha=0.6)
    # plt.scatter(points_gen_1[:,0], points_gen_1[:,1], color='blue', marker='o', label='cv2 ellipse mask', alpha=0.6)
    plt.scatter(points_gen_2[:,0], points_gen_2[:,1], color='red', marker='+', label='generate_masked_points', alpha=0.6)
    # plt.legend(loc='upper right')
    plt.title("Blue: cv2 ellipse mask, Red: generate_masked_points")
    plt.show()

    distances = []
    for p in points_gen_2:
        # find the closest point in points_gen_1
        closest_point = points_gen_1[np.argmin(np.linalg.norm(points_gen_1 - p, axis=1))]
        # compute the distance between the two points
        distance = np.linalg.norm(closest_point - p)
        distances.append(distance)
    print(np.max(distances))

    # should have the same points
    # assert np.allclose(points_gen_1, points_gen_2)

def select_minima_with_suppression(centers: np.ndarray, scores: np.ndarray, k: int = 2, suppress_radius: float = 15) -> list:
    """
    Select up to k minima with non-maximum suppression to avoid nearby picks.

    Args:
        centers: Candidate centers as array of shape (N, 2)
        scores: Cost values for each center, lower is better (shape (N,))
        k: Number of minima to select
        suppress_radius: Minimum Euclidean distance (in pixels) between picks

    Returns:
        List of selected centers (each shape (2,)) in order of increasing score
    """
    if centers is None or len(centers) == 0:
        return []
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(scores)
    selected = []
    for idx in order:
        c = centers[idx]
        if not np.isfinite(scores[idx]):
            continue
        too_close = False
        for s in selected:
            if np.hypot(c[0] - s[0], c[1] - s[1]) < suppress_radius:
                too_close = True
                break
        if too_close:
            continue
        selected.append(c)
        if len(selected) >= k:
            break
    return selected

def grid_search_refinement(ell: np.ndarray, poly: np.ndarray, K:np.ndarray, radius:float, search_ratio:float=0.5) -> tuple[np.ndarray, np.ndarray]:
    # print("ell:{}".format(ell))
    candidates=generate_masked_points(ell, search_ratio)
    scores = []
    for c in candidates:
        fc = eval_distance_f0(poly, c, K, 2*radius)
        scores.append(fc)

    scores = np.asarray(scores)

    all_centers=candidates
    all_scores=scores

    picked = select_minima_with_suppression(all_centers, all_scores, k=2, suppress_radius=10)
    # Fallback if suppression returns less than 2
    if len(picked) < 2:
        ids = np.argsort(np.asarray(all_scores).reshape(-1))[:2]
        picked = [all_centers[i] for i in ids]
    minimums = np.vstack(picked)

    found_mc = minimums[0, :].reshape((2,))
    another_mc = minimums[1, :].reshape((2,))

    return found_mc, another_mc


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    theta = np.linspace(0, np.pi * 2, 200)

    radius = 2
    # generate a 3D point of a circle
    P = np.vstack((radius * np.cos(theta), radius * np.sin(theta), np.zeros((1, 200))))
    # random pose
    R = np.eye(3)
    dR = cv.Rodrigues(rng.standard_normal((3, 1)) * 0.5)[0]
    R = R @ dR
    t = np.array([[1], [0], [5]]) + rng.standard_normal((3, 1)) * 1

    Pc = R @ P + t
    fx = 600
    # intrinsics
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')

    # normalize
    Pcn = Pc / Pc[-1, :]

    # project
    uvraw = K @ Pcn

    # add image noise (only on xy)
    uv = uvraw + np.vstack((rng.standard_normal(uvraw[:2, :].shape), np.zeros((1, uvraw.shape[1]))))

    uv = uv[:2, :].astype(dtype='float32')

    from utility import fit_ellipse
    ex,ey,ea,eb,etheta,ep,poly=fit_ellipse(uv)

    ell = [ex,ey,ea,eb,etheta]
    check_point_generation(ell, 0.5, K, ep)

    # K = 600
    # K = np.array([[K, 0, 640], [0, K, 480], [0, 0, 1]])
    # ell = [[640,480],[200,100],30]

    found_mc, another_mc = grid_search_refinement(ell, poly, K, radius, 0.3)
    found_mc1, another_mc1 = grid_search_refinement_cv(ell, poly, K, radius, 0.3)
    print("found_mc:{}".format(found_mc))
    print("another_mc:{}".format(another_mc))
    print("found_mc1:{}".format(found_mc1))
    print("another_mc1:{}".format(another_mc1))


    # ground truth center (projected)
    circular_center = R@np.array([[0],[0],[0]])+t
    circular_center = circular_center/circular_center[-1,:]
    uv_circular_center = K@circular_center
    real_mc = uv_circular_center.squeeze()[:2]  # Ground truth center
    print("real_mc:{}".format(real_mc))
    print("distance:{}".format(np.linalg.norm(found_mc - real_mc)))
    print("distance:{}".format(np.linalg.norm(another_mc - real_mc)))
    print("distance:{}".format(np.linalg.norm(found_mc1 - real_mc)))
    print("distance:{}".format(np.linalg.norm(another_mc1 - real_mc)))