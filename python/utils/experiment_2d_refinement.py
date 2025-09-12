import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, lru_cache
import multiprocessing as mp

from ellipse_center_refinement import *
from tqdm import tqdm, trange
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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



@lru_cache(maxsize=1000)
def _cached_eval_distance_f0(outer_tuple, center_tuple, K_tuple, marker_diameter, N=4):
    """
    Cached version of eval_distance_f0 for repeated computations.

    Note: This requires converting numpy arrays to tuples for hashing.
    """
    outer = np.array(outer_tuple)
    center = np.array(center_tuple)
    K = np.array(K_tuple)
    try:
        return eval_distance_f0(outer, center, K, marker_diameter, N)
    except Exception as e:
        return np.inf

def _evaluate_single_center(outer, center, K, marker_diameter, N=4):
    """
    Evaluate cost function for a single center point.

    This is a helper function for parallel processing.
    """
    try:
        return eval_distance_f0(outer, center, K, marker_diameter, N)
    except Exception as e:
        return np.inf

def _evaluate_single_center_cached(outer, center, K, marker_diameter, N=4):
    """
    Evaluate cost function for a single center point with caching.

    This is a helper function for parallel processing with caching.
    """
    try:
        # Convert numpy arrays to tuples for caching
        outer_tuple = tuple(outer.flatten())
        center_tuple = tuple(center.flatten())
        K_tuple = tuple(K.flatten())
        return _cached_eval_distance_f0(outer_tuple, center_tuple, K_tuple, marker_diameter, N)
    except Exception as e:
        return np.inf


def get_center_with_grid_search_cached(outer, centers, K, marker_diameter, N=4, max_workers=None):
    """
    Parallel version of grid search with caching for repeated computations.

    This function is optimized for scenarios where similar computations are repeated.
    """
    if len(centers) == 0:
        return centers, np.array([])

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(centers))

    # Use ThreadPoolExecutor for better performance with NumPy operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed arguments
        evaluate_func = partial(_evaluate_single_center_cached, outer, K=K, marker_diameter=marker_diameter, N=N)

        # Submit all tasks
        futures = [executor.submit(evaluate_func, center) for center in centers]

        # Collect results with progress bar
        scores = []
        for future in tqdm(futures, desc="Evaluating centers (cached)", leave=False):
            scores.append(future.result())

        scores = np.array(scores)

    # Sort by cost (ascending) to find minima
    # ids = np.argsort(scores)

    # Return top 2 candidates as potential local minima
    # minimums = centers[ids[:2], :]
    # best_scores = scores[ids[:2]]

    return centers, scores

def get_center_with_grid_search(outer, centers, K, marker_diameter):
    fs = []
    for c in centers:
        fc = eval_distance_f0(outer, c, K, marker_diameter)
        fs.append(fc)

    scores = np.asarray(fs)
    # ids = np.argsort(fs)
    # minimums = centers[ids[:2],:]
    # scores = fs[ids[:2]]
    return centers, scores

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

def monte_carlo_experiment(num:int=50, search_ratio:float=0.1):
    fx=600
    K = np.array([[fx,0,640],[0,fx,480],[0,0,1]],dtype='float32')
    R,t=generate_camera_pose()
    radius=3

    p3d=np.zeros((num,3),dtype='float32')
    p2d={}
    p2d['real']=np.zeros((num,2),dtype='float32')
    p2d['ellipse']=np.zeros((num,2),dtype='float32')
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
        uv=uvraw+np.vstack((np.random.randn(*(uvraw[:2,:].shape)),np.zeros((1,uvraw.shape[1]))))

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

        # if debug:
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(mask,cmap='gray')
        #     plt.xlim(0,1280)
        #     # plt.ylim(0,960)
        #     # plt.plot(uv[0,:],uv[1,:],'r.',label='input')
        #     # # plt.plot(ep[0, :], ep[1, :], 'r.', label='fitted')
        #     # plt.plot(ex, ey, 'bo', label='fitted ellipse center')
        #     # plt.plot(uv_circular_center[0, :], uv_circular_center[1, :], 'mx', label='projected circular center')
        #     # plt.plot(mmcx, mmcy, 'yd', label='center of mass')
        #     # plt.legend()
        #     # plt.axis('off')
        #     # plt.show()

        # img = np.zeros((960,1280,1),dtype='uint8')

        # to avoid search too large, we need to scale the ellipse by search_ratio, as we know that the center should be not very far from the projected circular center
        img=np.zeros((960,1280,1),dtype='uint8')
        mask2=cv.ellipse(img,(int(ex),int(ey)),(int(ea*search_ratio),int(eb*search_ratio)),etheta*180/np.pi,0,360,(255,255,255),-1)
        # defer plotting to combined 1x2 figure later

        # do grid search on the scaled ellipse, we need to sample all masked pixels and compute the score to generate the score map
        v_nz, u_nz, _=mask2.nonzero()
        # score map will be computed in the combined 1x2 figure below when debug

        centers=np.vstack((u_nz,v_nz)).T

        # ideally, we have two local minima; compute scores then select with suppression
        all_centers, all_scores = get_center_with_grid_search(poly, centers, K, 2*radius)
        picked = select_minima_with_suppression(all_centers, all_scores, k=2, suppress_radius=15)
        # Fallback if suppression returns less than 2
        if len(picked) < 2:
            ids = np.argsort(np.asarray(all_scores).reshape(-1))[:2]
            picked = [all_centers[i] for i in ids]
        minimums = np.vstack(picked)

        # Sort minima by distance to projected circular center (closest first)
        dists = np.sum((minimums - np.array([uv_circular_center[0,0], uv_circular_center[1,0]]))**2, axis=1)
        order = np.argsort(dists)
        minimums = minimums[order]

        found_mc = minimums[0, :].reshape((2,))
        another_mc = minimums[1, :].reshape((2,))

        # 1x2 figure: left shows ellipse and centers; right shows score map
        if debug:
            # Global font settings
            plt.rcParams.update({
                'font.family': 'Arial',
                'font.size': 12,
                'axes.titlesize': 12,
                'axes.labelsize': 12,
                'legend.fontsize': 12
            })
            # Use a screen-friendly DPI; keep high DPI only for saving
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=120)
            fig.subplots_adjust(wspace=0.05)
            # Set light gray figure background
            fig.patch.set_facecolor('#eeeeee')
            ax0, ax1 = axes
            # Set light gray axes backgrounds
            ax0.set_facecolor('#f5f5f5')
            ax1.set_facecolor('#f5f5f5')

            # Compute a zoomed view window around the fitted ellipse
            zoom_scale = 1.8  # increase to zoom out, decrease to zoom in
            xmin = max(0, ex - zoom_scale * ea)
            xmax = min(1280, ex + zoom_scale * ea)
            ymin = max(0, ey - zoom_scale * eb)
            ymax = min(960, ey + zoom_scale * eb)

            # Left panel
            # Render mask with 0-valued background as light gray instead of black
            from matplotlib.colors import LinearSegmentedColormap
            mask_norm = (mask.squeeze().astype('float32'))/255.0
            bg_cmap = LinearSegmentedColormap.from_list('mask_bg', ['#dddddd', '#ffffff'], N=256)
            ax0.imshow(mask_norm, cmap=bg_cmap, vmin=0.0, vmax=1.0, extent=[0,1280,960,0], interpolation='nearest')
            ax0.scatter(uv[0,:], uv[1,:], s=6, c='#1f77b4', alpha=0.5, edgecolors='none', label='Input Points')
            ax0.plot(ep[0, :], ep[1, :], color='#00bfa6', linewidth=2, label='Fitted Ellipse')
            ax0.scatter([ex], [ey], s=70, facecolors='none', edgecolors='#ffcc00', linewidths=2, label='Ellipse Center')
            ax0.scatter([uv_circular_center[0, 0]], [uv_circular_center[1, 0]], s=140, marker='x', color='#e41a1c', linewidths=2, label='Projected Circular Center')
            ax0.scatter([mmcx], [mmcy], s=60, marker='D', color='#984ea3', edgecolors='white', linewidths=0.8, label='Center of Mass')
            ax0.set_xlim(xmin, xmax)
            ax0.set_ylim(ymin, ymax)
            ax0.set_aspect('equal')
            ax0.set_xticks([]); ax0.set_yticks([])
            # Defer legend to figure-level legend below
            ax0.set_title('Ellipse and Centers')

            # Right panel: score map restricted to scaled ellipse region
            score_map = np.full(mask2.shape, np.nan, dtype='float32')
            for u, v in zip(u_nz, v_nz):
                score = eval_distance_f0(poly, np.array([u, v]), K, 2*radius, 4)
                score_map[v, u, :] = score
            # Normalize only valid (masked) scores; unmasked (NaN) will be shown as gray background
            min_score = np.nanmin(score_map)
            max_score = np.nanmax(score_map)
            score_norm = (score_map - min_score) / (max_score - min_score + 1e-12)
            # Draw gray background first
            ax1.imshow(np.ones_like(score_map.squeeze())*0.85, cmap='gray', vmin=0, vmax=1, extent=[0,1280,960,0], interpolation='nearest')
            # Overlay valid cost surface
            import matplotlib.pyplot as _plt
            cmap = _plt.get_cmap('magma')
            valid_cost = np.ma.masked_invalid(score_norm.squeeze())
            im1 = ax1.imshow(valid_cost, cmap=cmap, vmin=0, vmax=1,
                             extent=[0,1280,960,0], interpolation='nearest')
            # Cost contours over the heatmap
            H, W = valid_cost.shape
            X, Y = np.meshgrid(np.linspace(0, 1280, W), np.linspace(0, 960, H))
            ax1.contour(X, Y, valid_cost, levels=10, colors='white', linewidths=0.6, alpha=0.8)
            ax1.scatter([ex], [ey], s=50, facecolors='none', edgecolors='#ffcc00', linewidths=1.8, label='Ellipse Center')
            ax1.scatter([uv_circular_center[0, 0]], [uv_circular_center[1, 0]], s=140, marker='x', color='#e41a1c', linewidths=1.8, label='Projected Circular Center')
            ax1.scatter([mmcx], [mmcy], s=50, marker='D', color='#984ea3', edgecolors='white', linewidths=0.8, label='Center of Mass')
            ax1.plot(found_mc[0], found_mc[1], 'g*', markersize=10, label='local minimum 1')
            ax1.plot(another_mc[0], another_mc[1], 'g^', markersize=8, label='local minimum 2')
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_aspect('equal')
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title('Score Map (Scaled Ellipse)')
            # Inset colorbar to keep panel widths equal
            cax = inset_axes(ax1, width="3%", height="40%", loc='lower left',
                             bbox_to_anchor=(1.02, 0.1, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.tick_params(labelsize=9)
            # Figure-level legend combining labels from both subplots
            h0, l0 = ax0.get_legend_handles_labels()
            h1, l1 = ax1.get_legend_handles_labels()
            # Keep order and remove duplicates while preserving first occurrence
            seen = set()
            handles = []
            labels = []
            for h, l in list(zip(h0, l0)) + list(zip(h1, l1)):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
            # Shorten legend labels to avoid folding
            label_map = {
                'Input Points': 'Input',
                'Fitted Ellipse': 'Ellipse',
                'Ellipse Center': 'Ellipse Ctr',
                'Projected Circular Center': 'Proj Ctr',
                'Center of Mass': 'Mass Ctr',
                'local minimum 1': 'Local Min 1',
                'local minimum 2': 'Local Min 2'
            }
            labels_short = [label_map.get(l, l) for l in labels]
            # Set columns equal to number of entries to keep legend in one row
            ncols = max(1, len(labels_short))
            fig.legend(handles, labels_short,
                       loc='lower center', ncol=ncols, frameon=True, framealpha=0.95,
                       bbox_to_anchor=(0.5, -0.02), fancybox=True, borderpad=0.8,
                       handlelength=2.2, handletextpad=0.8, labelspacing=0.3)

            # Leave space at bottom for the figure legend
            plt.tight_layout(rect=[0, 0.10, 1, 1], pad=0.6)
            # Save figure to result folder (export at high DPI)
            os.makedirs('result', exist_ok=True)
            out_path = os.path.join('result', 'ellipse_score.png')
            fig.savefig(out_path, dpi=600, bbox_inches='tight')
            plt.show()


        p3d[i,:]=c3d.T
        p2d['ellipse'][i,:]=np.array([ex,ey])
        p2d['real'][i, :] = np.array([uv_circular_center[0,0], uv_circular_center[1,0]])
        p2d['gt1'][i, :] = np.array([found_mc[0], found_mc[1]])
        p2d['gt2'][i, :] = np.array([another_mc[0], another_mc[1]])

        print(f"p2d['ellipse'][i,:]: {p2d['ellipse'][i,:]}")
        print(f"p2d['real'][i, :]: {p2d['real'][i, :]}")
        print(f"p2d['gt1'][i, :]: {p2d['gt1'][i, :]}")
        print(f"p2d['gt2'][i, :]: {p2d['gt2'][i, :]}")

    Rres['real'], tres['real'], inliers['real']=recover_pose(p3d,p2d['real'],K)
    Rres['ellipse'], tres['ellipse'], inliers['ellipse'] = recover_pose(p3d, p2d['ellipse'], K)
    Rres['gt1'], tres['gt1'], inliers['gt1'] = recover_pose(p3d, p2d['gt1'], K)
    Rres['gt2'], tres['gt2'], inliers['gt2'] = recover_pose(p3d, p2d['gt2'], K)

    errs = {}
    avg_err={}
    for key in Rres.keys():
        valid3d = p3d[inliers[key].squeeze(), :]
        valid2d = p2d[key][inliers[key].squeeze(), :]
        errs[key], avg_err[key] = compute_reprojection_error(valid3d.T,
                                                             Rres[key], tres[key], K, valid2d.T)

    return avg_err, Rres, tres, R, t

def analyze_results(avg_err, Rres, tres, R, t):
    """
    Analyze and display experiment results.

    Args:
        avg_err: Average reprojection errors for each method
        Rres: Recovered rotation matrices
        tres: Recovered translation vectors
        R: True rotation matrix
        t: True translation vector
    """
    print("\\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)

    # Display reprojection errors
    print("\\nReprojection Errors (pixels):")
    print("-" * 40)
    for method, error in avg_err.items():
        if error < np.inf:
            print(f"{method:12}: {error:8.3f}")
        else:
            print(f"{method:12}: Failed")

    # Display pose accuracy (if available)
    print("\\nPose Recovery Success:")
    print("-" * 40)
    for method in avg_err.keys():
        if Rres[method] is not None:
            print(f"{method:12}: Success")
        else:
            print(f"{method:12}: Failed")

    # Method comparison
    print("\\nMethod Ranking (by accuracy):")
    print("-" * 40)
    valid_methods = [(method, error) for method, error in avg_err.items()
                    if error < np.inf]
    valid_methods.sort(key=lambda x: x[1])

    for i, (method, error) in enumerate(valid_methods, 1):
        print(f"{i}. {method:12}: {error:8.3f} pixels")

    print("\\n" + "="*60)

if __name__ == '__main__':
    """
    Main execution block for the ellipse center refinement experiment.

    This script runs a Monte Carlo experiment comparing different methods
    for estimating the center of a projected 3D circle from 2D ellipse observations.
    """
    print("Starting Ellipse Center Refinement Experiment")
    print("=" * 50)
    # Run the main experiment
    avg_err, Rres, tres, R, t = monte_carlo_experiment(num=10, search_ratio=0.5)

    # Analyze and display results
    analyze_results(avg_err, Rres, tres, R, t)

    # Optional: Run parameter optimization
    # optimal_ratio = optimize_search_parameters()
    # print(f"\\nRecommended search ratio: {optimal_ratio}")