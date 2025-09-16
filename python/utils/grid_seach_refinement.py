import dis
import numpy as np
import os
import sys
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ellipse_center_refinement import *


def generate_masked_points(ell, search_ratio, K):
    ex, ey, ea, eb, etheta = ell
    width = int(K[0,2]*2)
    height = int(K[1,2]*2)
    img=np.zeros((height,width,1),dtype='uint8')
    mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),-1)
    v_nz, u_nz, _=mask2.nonzero()
    # score map will be computed in the combined 1x2 figure below when debug
    final_points =np.vstack((u_nz,v_nz)).T
    return final_points

def check_point_generation(ell, search_ratio):
    ex, ey, ea, eb, etheta = ell
    print("ell:{}".format(ell))
    print("search_ratio:{}".format(search_ratio))
    print("etheta:{}".format(etheta))
    print("ea:{}".format(ea))
    print("eb:{}".format(eb))
    print("ex:{}".format(ex))
    print("ey:{}".format(ey))
    img=np.zeros((960,1280,1),dtype='uint8')
    mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta,0,360,(255,255,255),-1)
    # do grid search on the scaled ellipse, we need to sample all masked pixels and compute the score to generate the score map
    v_nz, u_nz, _=mask2.nonzero()
    # score map will be computed in the combined 1x2 figure below when debug
    points_gen_1=np.vstack((u_nz,v_nz)).T

    points_gen_2=generate_masked_points(ell, search_ratio)

    # plot both points_gen_1 and points_gen_2 in the image
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,10))
    # plt.imshow(img, cmap='gray')
    # plt.scatter(points_gen_1[:,0], points_gen_1[:,1], color='blue', marker='o', label='cv2 ellipse mask')
    # plt.scatter(points_gen_2[:,0], points_gen_2[:,1], color='red', marker='x', label='generate_masked_points')
    # plt.legend(loc='upper right')
    # plt.title("Blue: cv2 ellipse mask, Red: generate_masked_points")
    # plt.show()

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

def select_minima_with_suppression(centers, scores, k=2, suppress_radius=15):
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

def grid_search_refinement(ell: np.ndarray, poly: np.ndarray, K:np.ndarray, radius:float, search_ratio:float=0.5):
    # print("ell:{}".format(ell))
    candidates=generate_masked_points(ell, search_ratio, K)
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
    # first check the point generation
    # Generate a meaningful ellipse in the image for point generation check
    # Example: center (640, 480), axes (200, 100), angle 30 degrees
    ellipse_params = np.array([640, 480, 200, 100, 45])  # (cx, cy, a, b, theta_deg)
    check_point_generation(ellipse_params, 0.5)

    # K = 600
    # K = np.array([[K, 0, 640], [0, K, 480], [0, 0, 1]])
    # ell = [[640,480],[200,100],30]

    # found_mc, another_mc = grid_search_refinement(ellipse_params, K, 0.3)
