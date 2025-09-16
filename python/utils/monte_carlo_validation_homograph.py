import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv
from grid_seach_refinement import *
import find_rectify_homography as frh
from experiment_2d_refinement import fit_ellipse
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
#%matplotlib qt
# disable warnings
import warnings
warnings.filterwarnings("ignore")

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

def compute_ratio_error(ell, ep, ep_alt, mc):
    ex,ey,ea,eb,etheta = ell
    ell_cv = [[ex,ey],[ea*2,eb*2],etheta*180.0/np.pi]

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

    return ratio

def prediction_using_ratio_test(ell, ep, ep_alt, mc, nominal_ratio = 1.0, ratio_threshold = 0.05):
    ratio = compute_ratio_error(ell, ep, ep_alt, mc)
    print("ratio:{}".format(ratio))
    print("nominal_ratio:{}".format(nominal_ratio))
    if abs(ratio-nominal_ratio)<ratio_threshold:
        return True
    else:
        return False

def validation_using_homograph(ell, poly, ep, ep_alt, K, radius, real_mc, nominal_ratio = 1.0, ratio_threshold = 0.05, true_threshold = 3.0):
    # do grid search on ell to find out two candidates
    found_mc, another_mc = grid_search_refinement(ell, poly, K, radius*2, search_ratio=0.5)

    # make a toss to decide which mc to use
    toss = np.random.rand()
    if toss < 0.5:
        mc = found_mc
    else:
        mc = another_mc

    print("mc:{}".format(found_mc))
    print("another_mc:{}".format(another_mc))
    print("real_mc:{}".format(real_mc))
    print(np.linalg.norm(mc-real_mc))

    if np.linalg.norm(mc-real_mc)<true_threshold:
        prediction = prediction_using_ratio_test(ell, ep, ep_alt, mc, nominal_ratio, ratio_threshold)
        if prediction:
            # true positive
            return 'tp', found_mc, another_mc
        else:
            # false positive
            return 'fp', found_mc, another_mc
    else:
        #
        prediction = prediction_using_ratio_test(ell, ep, ep_alt, mc, nominal_ratio, ratio_threshold)
        if prediction:
            # false negative
            return 'fn', found_mc, another_mc
        else:
            # true negative
            return 'tn', found_mc, another_mc

    return None




def _validation_single_trial(seed: int):
    """Single trial for validation_test - extracted for parallelization"""
    rng = np.random.default_rng(seed)

    # generate a camera pose
    radius=2
    theta = np.linspace(0, np.pi * 2, 200)
    # generate a 3D point of a circle
    P = np.vstack((radius * np.cos(theta), radius * np.sin(theta), np.zeros((1, 200))))

    # a co-planar non-co-centric circle
    radius_alt=2
    P_alt = np.vstack((radius_alt * np.cos(theta), radius_alt * np.sin(theta), np.zeros((1, 200))))
    P_alt = P_alt+np.array([[4],[4],[0]])
    # generate a rotation matrix
    R = np.eye(3)
    dR = cv.Rodrigues(rng.standard_normal((3, 1)) * 0.6)[0]
    R = R @ dR
    # generate a translation vector
    t = np.array([[1], [0], [8]]) + rng.standard_normal((3, 1)) * 1
    Pc = R @ P + t
    Pc_alt = R @ P_alt + t

    # generate a K matrix
    fx = 600
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')

    # normalize the 3D points
    Pcn = Pc / Pc[-1, :]
    Pcn_alt = Pc_alt / Pc_alt[-1, :]

    # project the 3D points to the image plane
    uvraw = K @ Pcn
    uvraw_alt = K @ Pcn_alt

    # add noise to the image points
    uv = uvraw + np.vstack((rng.standard_normal(uvraw[:2, :].shape), np.zeros((1, uvraw.shape[1]))))
    uv_alt = uvraw_alt + np.vstack((rng.standard_normal(uvraw_alt[:2, :].shape), np.zeros((1, uvraw_alt.shape[1]))))

    # convert the image points to float32
    uv = uv[:2, :].astype(dtype='float32')
    uv_alt = uv_alt[:2, :].astype(dtype='float32')
    # fit an ellipse to the image points
    ex,ey,ea,eb,etheta,ep,poly=fit_ellipse(uv)
    ex_alt,ey_alt,ea_alt,eb_alt,etheta_alt,ep_alt,poly_alt=fit_ellipse(uv_alt)
    ellparams = np.array([ex,ey,ea,eb,etheta])
    ellparams_alt = np.array([ex_alt,ey_alt,ea_alt,eb_alt,etheta_alt])

    # ground truth center
    circular_center = R@np.array([[0],[0],[0]])+t
    circular_center = circular_center/circular_center[-1,:]
    uv_circular_center = K@circular_center
    real_mc = uv_circular_center.squeeze()[:2]  # Ground truth center

    nominal_ratio = radius / radius_alt

    # make a numpy of ell
    true_mc, false_found_mc = select_real_mc_using_homograph(ellparams, poly, ep, ep_alt, K, radius, nominal_ratio)

    # Compute error between true_mc (selected by homography) and real_mc (ground truth)
    error_vector = np.asarray(real_mc).reshape(-1) - np.asarray(true_mc).reshape(-1)  # [dx, dy]

    # also return the ellipse center
    ellipse_center = np.array([ellparams[0], ellparams[1]])

    return error_vector, real_mc, true_mc, false_found_mc, ellipse_center

def validation_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--radius_alt', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Directory to save FP/FN logs')
    args = parser.parse_args()

    # Limit OpenCV internal threads to avoid oversubscription
    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31-1)) for _ in range(args.trials)]

    error_vectors = []
    true_centers = []
    real_centers = []
    false_found_centers = []
    ellipse_centers = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_validation_single_trial, s) for s in seeds]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Monte Carlo (homograph validation)'):
            try:
                result = f.result()
                error_vec, real_mc, true_mc, false_found_mc, ellipse_center = result
                error_vectors.append(error_vec)
                true_centers.append(true_mc)
                false_found_centers.append(false_found_mc)
                real_centers.append(real_mc)
                ellipse_centers.append(ellipse_center)
            except Exception:
                continue

    print("error_vectors:{}".format(error_vectors))

    # Save error data
    if error_vectors:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        # Save error vectors
        error_data = np.array(error_vectors)
        np.savetxt(os.path.join(args.output_dir, 'validation_error_vectors.csv'),
                  error_data, delimiter=',',
                  header='dx,dy', comments='')

        # Save centers
        true_data = np.array(true_centers)
        real_data = np.array(real_centers)
        np.savetxt(os.path.join(args.output_dir, 'validation_true_centers.csv'),
                  true_data, delimiter=',',
                  header='true_x,true_y', comments='')
        np.savetxt(os.path.join(args.output_dir, 'validation_real_centers.csv'),
                  real_data, delimiter=',',
                  header='real_x,real_y', comments='')
        false_found_data = np.array(false_found_centers)
        np.savetxt(os.path.join(args.output_dir, 'validation_false_found_centers.csv'),
                  false_found_data, delimiter=',',
                  header='false_found_x,false_found_y', comments='')

        # Save ellipse centers
        ellipse_data = np.array(ellipse_centers)
        np.savetxt(os.path.join(args.output_dir, 'validation_ellipse_centers.csv'),
                  ellipse_data, delimiter=',',
                  header='ellipse_x,ellipse_y', comments='')

        # compute the error vector between the ellipse center and the real_mc
        ell_error_data = np.array(ellipse_centers) - np.array(real_centers)

        # Plot error distribution
        plot_error_distribution(error_data, ell_error_data, args.output_dir)

        print("Saved error data and plots to {}".format(args.output_dir))

        # also list the case where the error is greater than 3 pixels, print the error vector, true_mc, false_found_mc, real_mc
        for i in range(len(error_data)):
            if np.linalg.norm(error_data[i]) > 3:
                print("error_vector:{}".format(error_data[i]))
                print("true_mc:{}".format(true_centers[i]))
                print("false_found_mc:{}".format(false_found_centers[i]))
                print("real_mc:{}".format(real_centers[i]))
                print("ellipse_center:{}".format(ellipse_centers[i]))
                print("--------------------------------")

def plot_error_distribution(error_data, ell_error_data, output_dir):
    """Plot 2D error distribution between true_mc and real_mc"""
    dx = error_data[:, 0]
    dy = error_data[:, 1]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Distribution: Homography-selected vs Ground Truth Centers', fontsize=14)

    # 1. Scatter plot of errors
    ax1 = axes[0, 0]
    scatter = ax1.scatter(dx, dy, alpha=0.6, s=20, c=np.linalg.norm(error_data, axis=1),
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Error in X (pixels)')
    ax1.set_ylabel('Error in Y (pixels)')
    ax1.set_title('2D Error Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='Error Magnitude (pixels)')

    # 2. Histogram of X errors
    ax2 = axes[0, 1]
    ax2.hist(dx, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Error in X (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('X Error Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=np.mean(dx), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(dx):.2f}')
    ax2.legend()

    # 3. Histogram of Y errors
    ax3 = axes[1, 0]
    ax3.hist(dy, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Error in Y (pixels)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Y Error Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=np.mean(dy), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(dy):.2f}')
    ax3.legend()

    # 4. Error magnitude distribution
    ax4 = axes[1, 1]
    error_magnitudes = np.linalg.norm(error_data, axis=1)
    ax4.hist(error_magnitudes, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Error Magnitude (pixels)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Magnitude Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=np.mean(error_magnitudes), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(error_magnitudes):.2f}')
    ax4.axvline(x=np.median(error_magnitudes), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(error_magnitudes):.2f}')
    ax4.legend()

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'validation_error_distribution.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\nError Statistics:")
    print(f"Mean X error: {np.mean(dx):.3f} ± {np.std(dx):.3f} pixels")
    print(f"Mean Y error: {np.mean(dy):.3f} ± {np.std(dy):.3f} pixels")
    print(f"Mean magnitude: {np.mean(error_magnitudes):.3f} ± {np.std(error_magnitudes):.3f} pixels")
    print(f"Median magnitude: {np.median(error_magnitudes):.3f} pixels")
    print(f"Max magnitude: {np.max(error_magnitudes):.3f} pixels")
    print(f"95th percentile: {np.percentile(error_magnitudes, 95):.3f} pixels")

def _simulate_once(seed: int, fx: float, radius: float, radius_alt: float):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi * 2, 200)

    # generate a 3D point of a circle
    P = np.vstack((radius * np.cos(theta), radius * np.sin(theta), np.zeros((1, 200))))

    # a co-planar non-co-centric circle
    P_alt = np.vstack((radius_alt * np.cos(theta), radius_alt * np.sin(theta), np.zeros((1, 200))))
    P_alt = P_alt+np.array([[4],[4],[0]])

    # random pose
    R = np.eye(3)
    dR = cv.Rodrigues(rng.standard_normal((3, 1)) * 0.5)[0]
    R = R @ dR
    t = np.array([[1], [0], [8]]) + rng.standard_normal((3, 1)) * 1

    Pc = R @ P + t
    Pc_alt = R @ P_alt + t

    # intrinsics
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')

    # normalize
    Pcn = Pc / Pc[-1, :]
    Pcn_alt = Pc_alt / Pc_alt[-1, :]

    # project
    uvraw = K @ Pcn
    uvraw_alt = K @ Pcn_alt

    # add image noise (only on xy)
    uv = uvraw + np.vstack((rng.standard_normal(uvraw[:2, :].shape), np.zeros((1, uvraw.shape[1]))))
    uv_alt = uvraw_alt + np.vstack((rng.standard_normal(uvraw_alt[:2, :].shape), np.zeros((1, uvraw_alt.shape[1]))))

    uv = uv[:2, :].astype(dtype='float32')
    uv_alt = uv_alt[:2, :].astype(dtype='float32')

    ex,ey,ea,eb,etheta,ep,poly=fit_ellipse(uv)
    ex_alt,ey_alt,ea_alt,eb_alt,etheta_alt,ep_alt,poly_alt=fit_ellipse(uv_alt)
    ellparams = np.array([ex,ey,ea,eb,etheta])
    ellparams_alt = np.array([ex_alt,ey_alt,ea_alt,eb_alt,etheta_alt])

    # ground truth center (projected)
    circular_center = R@np.array([[0],[0],[0]])+t
    circular_center = circular_center/circular_center[-1,:]
    uv_circular_center = K@circular_center

    nominal_ratio = radius / radius_alt
    ratio_threshold = 0.1

    label, found_mc, another_mc = validation_using_homograph(ellparams, poly, ep, ep_alt, K, radius, uv_circular_center[:2,:].squeeze().tolist(), nominal_ratio, ratio_threshold)
    return label, np.asarray(found_mc).reshape(-1).tolist(), np.asarray(another_mc).reshape(-1).tolist(), np.asarray(uv_circular_center[:2,:].squeeze()).reshape(-1).tolist()

def validdation_test_precision_recall():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--radius_alt', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Directory to save FP/FN logs')
    args = parser.parse_args()

    # Limit OpenCV internal threads to avoid oversubscription
    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31-1)) for _ in range(args.trials)]
    results = []
    fp_logs = []
    fn_logs = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_simulate_once, s, args.fx, args.radius, args.radius_alt) for s in seeds]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Monte Carlo (homograph validation)'):
            try:
                label, found_mc, another_mc, real_mc = f.result()
                results.append(label)
                if label == 'fp':
                    fp_logs.append([*found_mc, *another_mc, *real_mc])
                elif label == 'fn':
                    fn_logs.append([*found_mc, *another_mc, *real_mc])
            except Exception:
                continue

    print("results:{}".format(results))

    # now we have tp, fp, fn, tn, compute the metrics
    arr = np.array(results)
    cnt_tp = np.sum(arr == 'tp')
    cnt_fp = np.sum(arr == 'fp')
    cnt_fn = np.sum(arr == 'fn')
    cnt_tn = np.sum(arr == 'tn')
    print("tp:{}".format(cnt_tp))
    print("fp:{}".format(cnt_fp))
    print("fn:{}".format(cnt_fn))
    print("tn:{}".format(cnt_tn))
    precision = cnt_tp/(cnt_tp+cnt_fp) if (cnt_tp+cnt_fp)>0 else 0.0
    recall = cnt_tp/(cnt_tp+cnt_fn) if (cnt_tp+cnt_fn)>0 else 0.0
    accuracy = (cnt_tp+cnt_tn)/(cnt_tp+cnt_tn+cnt_fp+cnt_fn) if (cnt_tp+cnt_tn+cnt_fp+cnt_fn)>0 else 0.0
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("accuracy:{}".format(accuracy))

    # Save FP/FN logs with found_mc, another_mc, real_mc
    try:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        header = 'found_mc_x,found_mc_y,another_mc_x,another_mc_y,real_mc_x,real_mc_y'
        if len(fp_logs) > 0:
            np.savetxt(os.path.join(args.output_dir, 'homograph_fp_logs.csv'), np.array(fp_logs, dtype=float), delimiter=',', header=header, comments='')
        if len(fn_logs) > 0:
            np.savetxt(os.path.join(args.output_dir, 'homograph_fn_logs.csv'), np.array(fn_logs, dtype=float), delimiter=',', header=header, comments='')
        print("Saved FP/FN logs to {}".format(args.output_dir))
    except Exception as e:
        print("Failed to save FP/FN logs: {}".format(e))

if __name__ == "__main__":
    # Run parallel validation test
    validation_test()
