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

    return real_mc, found_mc, another_mc

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

def validation_test():
    N=100
    res=np.zeros((N,),dtype='uint8')

    for i in tqdm(range(N),total=N):
        # generate a camera pose
        radius=2
        theta = np.linspace(0, np.pi * 2, 200)
        # generate a 3D point of a circle
        P = np.vstack((radius * np.cos(theta), radius * np.sin(theta), np.zeros((1, 200))))

        # a co-planar non-co-centric circle
        radius_alt=3
        P_alt = np.vstack((radius_alt * np.cos(theta), radius_alt * np.sin(theta), np.zeros((1, 200))))
        P_alt = P_alt+np.array([[4],[4],[0]])
        # generate a rotation matrix
        R = np.eye(3)
        dR = cv.Rodrigues(np.random.randn(3, 1) * 0.6)[0]
        R = R @ dR
        # generate a translation vector
        t = np.array([[1], [0], [8]]) + np.random.randn(3, 1) * 1
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
        uv = uvraw + np.vstack((np.random.randn(*uvraw[:2, :].shape), np.zeros((1, uvraw.shape[1]))))
        uv_alt = uvraw_alt + np.vstack((np.random.randn(*uvraw_alt[:2, :].shape), np.zeros((1, uvraw_alt.shape[1]))))

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

        nominal_ratio = radius / radius_alt

        # make a numpy of ell
        real_mc, found_mc, another_mc = select_real_mc_using_homograph(ellparams, poly, ep, ep_alt, K, radius, nominal_ratio)

        print("real_mc:{}".format(real_mc))
        print("found_mc:{}".format(found_mc))
        print("another_mc:{}".format(another_mc))

        # we do a true positive, false positive, false negative, true negative computation
        error = np.linalg.norm(uv_circular_center.squeeze()[:2]-real_mc)

        found_real=False
        if error<3:
            found_real=True
            res[i]=1

    # print the successful rate
    print("successful rate:{}".format(np.nonzero(res==1)[0].shape[0]/N))


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


if __name__ == "__main__":
    # validation_test()

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