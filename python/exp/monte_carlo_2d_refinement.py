import os
import argparse
import numpy as np
from tqdm import tqdm, trange
import cv2 as cv
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from experiment_2d_refinement import generate_camera_pose, generate_points, fit_ellipse, recover_pose, compute_reprojection_error, compute_pose_error
from grid_seach_refinement import select_real_mc_using_homograph

def _validation_single_trial(seed: int, num: int = 50):
    """Single trial for validation_test - extracted for parallelization"""
    rng = np.random.default_rng(seed)

    # some parameters
    radius=2
    radius_alt=3
    # generate a K matrix
    fx = 600
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')
    # generate a camera pose
    R,t=generate_camera_pose()

    # point buffers
    p3d=np.zeros((num,3),dtype='float32')
    p2d={}
    p2d['real_center']=np.zeros((num,2),dtype='float32')
    p2d['ellipse_center']=np.zeros((num,2),dtype='float32')
    p2d['mass_center']=np.zeros((num,2),dtype='float32')
    p2d['true_minima']=np.zeros((num,2),dtype='float32')
    p2d['false_minima']=np.zeros((num,2),dtype='float32')

    for i in trange(num, desc='point generation'):
        while True:
            # generate a 3D point of a circle
            P,c3d=generate_points(radius)
            Pc=R@P+t
            Pcn=Pc/Pc[-1,:]
            uvraw=K@Pcn
            uv=uvraw+np.vstack((np.random.randn(*(uvraw[:2,:].shape)),np.zeros((1,uvraw.shape[1]))))
            uv=uv[:2,:].astype(dtype='float32')
            ex,ey,ea,eb,etheta,ep,poly=fit_ellipse(uv)
            ellparams = np.array([ex,ey,ea,eb,etheta])
            ellipse_center = np.array([ex,ey])

            # generate another non-co-centric circle
            P_alt,c3d_alt=generate_points(radius_alt)
            Pc_alt=R@P_alt+t
            Pcn_alt=Pc_alt/Pc_alt[-1,:]
            uvraw_alt=K@Pcn_alt
            uv_alt=uvraw_alt+np.vstack((np.random.randn(*(uvraw_alt[:2,:].shape)),np.zeros((1,uvraw_alt.shape[1]))))
            uv_alt=uv_alt[:2,:].astype(dtype='float32')
            _,_,_,_,_,ep_alt,_=fit_ellipse(uv_alt)

            if np.linalg.norm(c3d - c3d_alt) > 0.1:
                break

        # compute the center of mass
        height = int(K[1,2]*2)
        width = int(K[0,2]*2)
        img=np.zeros((height,width,1),dtype='uint8')
        mask=cv.ellipse(img,(int(ex),int(ey)),(int(ea),int(eb)),etheta*180/np.pi,0,360,(255,255,255),-1)
        mm=cv.moments(mask)
        mmcx = mm['m10'] / mm['m00']
        mmcy = mm['m01'] / mm['m00']
        mass_center = np.array([mmcx, mmcy])

        circular_center = R@c3d+t
        circular_center=circular_center/circular_center[-1,:]
        uv_circular_center = K@circular_center
        real_center = uv_circular_center.squeeze()[:2]  # Ground truth center

        nominal_ratio = radius / radius_alt

        # make a numpy of ell
        true_minima, false_minima = select_real_mc_using_homograph(ellparams, poly, ep, ep_alt, K, radius, nominal_ratio)

        print("real_center:{}".format(real_center))
        print("true_minima:{}".format(true_minima))
        print("false_minima:{}".format(false_minima))

        p3d[i,:]=c3d.T
        p2d['ellipse_center'][i,:]=ellipse_center
        p2d['mass_center'][i,:]=mass_center
        p2d['real_center'][i, :] = real_center
        p2d['true_minima'][i, :] = true_minima
        p2d['false_minima'][i, :] = false_minima

    # result
    Rres={}
    tres={}
    inliers={}
    # use real_center for pnp
    Rres['real_center'], tres['real_center'], inliers['real_center']=recover_pose(p3d,p2d['real_center'],K)
    # use ellipse center for pnp
    Rres['ellipse_center'], tres['ellipse_center'], inliers['ellipse_center'] = recover_pose(p3d, p2d['ellipse_center'], K)
    # use mass center for pnp
    Rres['mass_center'], tres['mass_center'], inliers['mass_center'] = recover_pose(p3d, p2d['mass_center'], K)
    # use true minima for pnp
    Rres['true_minima'], tres['true_minima'], inliers['true_minima'] = recover_pose(p3d, p2d['true_minima'], K)
    # use false minima for pnp
    Rres['false_minima'], tres['false_minima'], inliers['false_minima'] = recover_pose(p3d, p2d['false_minima'], K)

    errs = {}
    avg_err={}
    for key in Rres.keys():
        valid3d = p3d[inliers[key].squeeze(), :]
        valid2d = p2d[key][inliers[key].squeeze(), :]
        errs[key], avg_err[key] = compute_reprojection_error(valid3d.T,
                                                             Rres[key], tres[key], K, valid2d.T)

    r_error, t_error = compute_pose_error(Rres, tres, R, t)

    # the return values are average reprojection error (key: scalar), rotation error (key: scalar), translation error (key: scalar), Rres (dict), tres (dict), R (array), t (array)
    return avg_err, r_error, t_error, Rres, tres, R, t



def validation_test(args):
    # Limit OpenCV internal threads to avoid oversubscription
    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31-1)) for _ in range(args.trials)]

    avg_errs = []
    r_errors = []
    t_errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_validation_single_trial, s, args.num) for s in seeds]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Monte Carlo (PnP refinement validation)'):
            # try:
            result = f.result()
            avg_err, r_error, t_error, _, _, _, _ = result
            avg_errs.append(avg_err)
            r_errors.append(r_error)
            t_errors.append(t_error)

            # except Exception:
                # continue

    # get errors of each center
    real_center_errors = []
    ellipse_center_errors = []
    mass_center_errors = []
    true_minima_errors = []
    false_minima_errors = []
    for i in range(len(avg_errs)):
        real_center_errors.append([avg_errs[i]['real_center'], r_errors[i]['real_center'], t_errors[i]['real_center']])
        ellipse_center_errors.append([avg_errs[i]['ellipse_center'], r_errors[i]['ellipse_center'], t_errors[i]['ellipse_center']])
        mass_center_errors.append([avg_errs[i]['mass_center'], r_errors[i]['mass_center'], t_errors[i]['mass_center']])
        true_minima_errors.append([avg_errs[i]['true_minima'], r_errors[i]['true_minima'], t_errors[i]['true_minima']])
        false_minima_errors.append([avg_errs[i]['false_minima'], r_errors[i]['false_minima'], t_errors[i]['false_minima']])

    real_center_errors = np.array(real_center_errors)
    ellipse_center_errors = np.array(ellipse_center_errors)
    mass_center_errors = np.array(mass_center_errors)
    true_minima_errors = np.array(true_minima_errors)
    false_minima_errors = np.array(false_minima_errors)

    # Save error data
    if real_center_errors.shape[0] > 0:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        # Save error vectors
        error_data = np.array(real_center_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_real_center_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        error_data = np.array(ellipse_center_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_ellipse_center_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        error_data = np.array(mass_center_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_mass_center_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        error_data = np.array(true_minima_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_true_minima_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        error_data = np.array(false_minima_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_false_minima_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        print("Saved error data and plots to {}".format(args.output_dir))

        print('Mean reprojection error:')
        print('real_center:{}'.format(np.mean(real_center_errors[:,0])))
        print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,0])))
        print('mass_center:{}'.format(np.mean(mass_center_errors[:,0])))
        print('true_minima:{}'.format(np.mean(true_minima_errors[:,0])))
        print('false_minima:{}'.format(np.mean(false_minima_errors[:,0])))

        print('Mean rotation error:')
        print('real_center:{}'.format(np.mean(real_center_errors[:,1])))
        print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,1])))
        print('mass_center:{}'.format(np.mean(mass_center_errors[:,1])))
        print('true_minima:{}'.format(np.mean(true_minima_errors[:,1])))
        print('false_minima:{}'.format(np.mean(false_minima_errors[:,1])))

        print('Mean translation error:')
        print('real_center:{}'.format(np.mean(real_center_errors[:,2])))
        print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,2])))
        print('mass_center:{}'.format(np.mean(mass_center_errors[:,2])))
        print('true_minima:{}'.format(np.mean(true_minima_errors[:,2])))
        print('false_minima:{}'.format(np.mean(false_minima_errors[:,2])))

def load_error_data(args):
    real_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_real_center_errors.csv'), delimiter=',', skiprows=1)
    ellipse_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_ellipse_center_errors.csv'), delimiter=',', skiprows=1)
    mass_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_mass_center_errors.csv'), delimiter=',', skiprows=1)
    true_minima_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_true_minima_errors.csv'), delimiter=',', skiprows=1)
    false_minima_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_false_minima_errors.csv'), delimiter=',', skiprows=1)

    print("Loaded error data from {}".format(args.output_dir))

    print('Mean reprojection error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,0])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,0])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,0])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,0])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,0])))


    print('Mean rotation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,1])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,1])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,1])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,1])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,1])))


    print('Mean translation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,2])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,2])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,2])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,2])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,2])))

    # Create 4x3 subplots for error visualization
    plot_error_comparison(ellipse_center_errors, mass_center_errors,
                         true_minima_errors, args.output_dir)
    plot_error_bars(ellipse_center_errors, mass_center_errors,
                    true_minima_errors, args.output_dir)

def plot_error_comparison(ellipse_center_errors, mass_center_errors,
                         true_minima_errors, output_dir):
    """Plot error comparison with 3 subplots: one for each error type"""

    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 30

    # Create 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Define error data and labels
    error_data = [ellipse_center_errors, mass_center_errors, true_minima_errors]
    method_labels = ['Ellipse Center', 'Mass Center', 'Refined Center']
    error_types = ['Reprojection Error', 'Rotation Error', 'Translation Error']

    # Define light colors for each method
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    edge_colors = ['blue', 'green', 'red']

    # Plot each error type (column) in the first three subplots
    for j, error_type in enumerate(error_types):
        ax = axes[j]

        # Prepare data for this error type
        data_to_plot = [data[:, j] for data in error_data]
        positions = range(1, len(method_labels) + 1)

        # Create box plots for all methods
        box_plot = ax.boxplot(data_to_plot, positions=positions, patch_artist=True,
                             boxprops=dict(alpha=0.7),
                             medianprops=dict(color='black', linewidth=2),
                             whiskerprops=dict(color='black'),
                             capprops=dict(color='black'),
                             flierprops=dict(marker='o', markerfacecolor='red',
                                             markeredgecolor='black', markersize=4))

        # Color each box differently
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(edge_colors[colors.index(color)])
            patch.set_linewidth(1.5)

        # Set labels and title
        # ax.set_title(error_type)
        # ax.set_ylabel('Error Magnitude')
        ax.set_ylabel(error_type)

        # Set x-axis labels
        # ax.set_xticks(positions)
        # ax.set_xticklabels([], rotation=45, ha='right')
        ax.set_xticks([])

        # Use a logarithmic scale for the y-axis
        # ax.set_yscale('log')

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add single legend for all methods
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor=edge_color,
                                   alpha=0.7, label=label)
                      for color, edge_color, label in zip(colors, edge_colors, method_labels)]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, fontsize=30)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.85)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'boxplot_error_comparison_combined.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved error comparison plot to {output_dir}/boxplot_error_comparison_combined.png")

def plot_error_bars(ellipse_center_errors, mass_center_errors,
                    true_minima_errors, output_dir):
    """Plot a bar chart comparing error types across different methods."""

    # Calculate the mean errors for each method and type
    mean_errors = [
        np.mean(ellipse_center_errors, axis=0),
        np.mean(mass_center_errors, axis=0),
        np.mean(true_minima_errors, axis=0)
    ]

    # Define method labels and error types
    method_labels = ['Ellipse Center', 'Mass Center', 'Refined Center']
    error_types = ['Reprojection Error', 'Rotation Error', 'Translation Error']

    # Define colors for each method
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    edge_colors = ['blue', 'green', 'red']

    # Number of methods and error types
    num_methods = len(method_labels)
    num_error_types = len(error_types)

    # Create a figure for the bar plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set positions for each group of bars
    bar_width = 0.2
    indices = np.arange(num_error_types)  # positions for error types

    # Plot bars for each method
    for i, (method, color, edge_color) in enumerate(zip(method_labels, colors, edge_colors)):
        bars = ax.bar(indices + i * bar_width, mean_errors[i], bar_width,
                      label=method, color=color, edgecolor=edge_color, alpha=0.7)

        # Annotate the bars with their exact values
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    # Set x-ticks and labels
    ax.set_xticks(indices + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(error_types, rotation=10, ha='right', fontsize=20)

    # Add labels and title
    ax.set_ylabel('Mean Error')
    # ax.set_title('Error Comparison Across Methods and Types')

    # Use a logarithmic scale if needed to emphasize smaller error values
    # ax.set_yscale('log')

    # Add legend
    ax.legend(fontsize=20)

    # Add grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'error_bar_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved error bar comparison plot to {output_dir}/error_bar_comparison.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('--num', type=int, default=20, help='Number of circles per trial')
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--radius_alt', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Directory to save results')
    parser.add_argument('--load_only', action='store_true', help='Load error data from csv files')
    args = parser.parse_args()

    if args.load_only:
        load_error_data(args)
    else:
        validation_test(args)

if __name__ == "__main__":
    main()
