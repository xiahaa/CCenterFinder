from signal import raise_signal
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from utility import recover_pose, compute_reprojection_error, compute_pose_error
from ransac_2c_center_refinement import ransac_validation
import matplotlib.pyplot as plt


def run_ransac(args):
    # load the p3s and p2sets
    import os
    p3s = np.load(os.path.join(args.output_dir, 'p3ds.npz'), allow_pickle=True)['p3ds']
    p2sets = np.load(os.path.join(args.output_dir, 'p2ds.npz'), allow_pickle=True)['p2ds']
    Rs = np.load(os.path.join(args.output_dir, 'Rs.npz'), allow_pickle=True)['Rs']
    ts = np.load(os.path.join(args.output_dir, 'ts.npz'), allow_pickle=True)['ts']
    fx = 600
    K = np.array([[fx, 0, 640], [0, fx, 480], [0, 0, 1]], dtype='float32')

    avg_errs = []
    r_errors = []
    t_errors = []

    for i in range(len(p3s)):
        p3d = p3s[i]
        p2d = p2sets[i]
        R = Rs[i]
        t = ts[i]

        print(t)

        # result
        Rres={}
        tres={}
        inliers={}
        # use real_center for pnp
        Rres['real_center'], tres['real_center'], inliers['real_center']=recover_pose(p3d,p2d['real_center'],K,ransac_threshold=5)
        # use ellipse center for pnp
        Rres['ellipse_center'], tres['ellipse_center'], inliers['ellipse_center'] = recover_pose(p3d, p2d['ellipse_center'], K, ransac_threshold=5)
        # use mass center for pnp
        Rres['mass_center'], tres['mass_center'], inliers['mass_center'] = recover_pose(p3d, p2d['mass_center'], K, ransac_threshold=5)
        # use true minima for pnp
        Rres['true_minima'], tres['true_minima'], inliers['true_minima'] = recover_pose(p3d, p2d['true_minima'], K, ransac_threshold=5)
        # use false minima for pnp
        Rres['false_minima'], tres['false_minima'], inliers['false_minima'] = recover_pose(p3d, p2d['false_minima'], K, ransac_threshold=8)

        errs = {}
        avg_err={}
        for key in Rres.keys():
            # print(key)
            valid3d = p3d[inliers[key].squeeze(), :]
            valid2d = p2d[key][inliers[key].squeeze(), :]
            errs[key], avg_err[key] = compute_reprojection_error(valid3d.T,
                                                                Rres[key], tres[key], K, valid2d.T)



        # implement the ransac validation
        Rres['ransac'], tres['ransac'], avg_err['ransac'], _ = ransac_validation(p3d, p2d['true_minima'], p2d['false_minima'], K)

        r_error, t_error = compute_pose_error(Rres, tres, R, t)


        avg_errs.append(avg_err)
        r_errors.append(r_error)
        t_errors.append(t_error)


    # get errors of each center
    real_center_errors = []
    ellipse_center_errors = []
    mass_center_errors = []
    true_minima_errors = []
    false_minima_errors = []
    ransac_errors = []
    for i in range(len(avg_errs)):
        real_center_errors.append([avg_errs[i]['real_center'], r_errors[i]['real_center'], t_errors[i]['real_center']])
        ellipse_center_errors.append([avg_errs[i]['ellipse_center'], r_errors[i]['ellipse_center'], t_errors[i]['ellipse_center']])
        mass_center_errors.append([avg_errs[i]['mass_center'], r_errors[i]['mass_center'], t_errors[i]['mass_center']])
        true_minima_errors.append([avg_errs[i]['true_minima'], r_errors[i]['true_minima'], t_errors[i]['true_minima']])
        false_minima_errors.append([avg_errs[i]['false_minima'], r_errors[i]['false_minima'], t_errors[i]['false_minima']])
        ransac_errors.append([avg_errs[i]['ransac'], r_errors[i]['ransac'], t_errors[i]['ransac']])

    real_center_errors = np.array(real_center_errors)
    ellipse_center_errors = np.array(ellipse_center_errors)
    mass_center_errors = np.array(mass_center_errors)
    true_minima_errors = np.array(true_minima_errors)
    false_minima_errors = np.array(false_minima_errors)
    ransac_errors = np.array(ransac_errors)

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

        error_data = np.array(ransac_errors)
        np.savetxt(os.path.join(args.output_dir, 'pnp_ransac_errors.csv'),
                  error_data, delimiter=',',
                  header='avg_err,r_error,t_error', comments='')

        print("Saved error data and plots to {}".format(args.output_dir))

    print('Mean reprojection error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,0])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,0])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,0])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,0])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,0])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,0])))

    print('Mean rotation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,1])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,1])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,1])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,1])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,1])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,1])))

    print('Mean translation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,2])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,2])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,2])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,2])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,2])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,2])))


def load_error_data(args):
    real_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_real_center_errors.csv'), delimiter=',', skiprows=1)
    ellipse_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_ellipse_center_errors.csv'), delimiter=',', skiprows=1)
    mass_center_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_mass_center_errors.csv'), delimiter=',', skiprows=1)
    true_minima_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_true_minima_errors.csv'), delimiter=',', skiprows=1)
    false_minima_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_false_minima_errors.csv'), delimiter=',', skiprows=1)
    ransac_errors = np.loadtxt(os.path.join(args.output_dir, 'pnp_ransac_errors.csv'), delimiter=',', skiprows=1)

    print("Loaded error data from {}".format(args.output_dir))

    print('Mean reprojection error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,0])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,0])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,0])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,0])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,0])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,0])))

    print('Mean rotation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,1])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,1])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,1])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,1])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,1])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,1])))

    print('Mean translation error:')
    print('real_center:{}'.format(np.mean(real_center_errors[:,2])))
    print('ellipse_center:{}'.format(np.mean(ellipse_center_errors[:,2])))
    print('mass_center:{}'.format(np.mean(mass_center_errors[:,2])))
    print('true_minima:{}'.format(np.mean(true_minima_errors[:,2])))
    print('false_minima:{}'.format(np.mean(false_minima_errors[:,2])))
    print('ransac:{}'.format(np.mean(ransac_errors[:,2])))

    # Create 4x3 subplots for error visualization
    plot_error_comparison(ellipse_center_errors, mass_center_errors,
                         true_minima_errors, ransac_errors, args.output_dir)
    plot_error_bars(ellipse_center_errors, mass_center_errors,
                    true_minima_errors, ransac_errors, args.output_dir)

def plot_error_comparison(ellipse_center_errors, mass_center_errors,
                         true_minima_errors, ransac_errors, output_dir):
    """Plot error comparison with 3 subplots: one for each error type"""

    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 30

    # Create 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Define error data and labels
    error_data = [ellipse_center_errors, mass_center_errors, true_minima_errors, ransac_errors]
    method_labels = ['Ellipse Center', 'Mass Center', 'Refined Center', 'RANSAC Center']
    error_types = ['Reprojection Error', 'Rotation Error', 'Translation Error']

    # Define light colors for each method
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum']
    edge_colors = ['blue', 'green', 'red', 'purple']

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
                    true_minima_errors, ransac_errors, output_dir):
    """Plot a bar chart comparing error types across different methods."""

    # Calculate the mean errors for each method and type
    mean_errors = [
        np.mean(ellipse_center_errors, axis=0),
        np.mean(mass_center_errors, axis=0),
        np.mean(true_minima_errors, axis=0),
        np.mean(ransac_errors, axis=0)
    ]

    # Define method labels and error types
    method_labels = ['Ellipse Center', 'Mass Center', 'Refined Center', 'RANSAC Center']
    error_types = ['Reprojection Error', 'Rotation Error', 'Translation Error']

    # Define colors for each method
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum']
    edge_colors = ['blue', 'green', 'red', 'purple']

    # Number of methods and error types
    num_methods = len(method_labels)
    num_error_types = len(error_types)

    # Create a figure for the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

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
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

    # Set x-ticks and labels
    ax.set_xticks(indices + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(error_types, rotation=10, ha='center', fontsize=15)

    # Add labels and title
    ax.set_ylabel('Mean Error', fontsize=15)
    # ax.set_title('Error Comparison Across Methods and Types')

    # Use a logarithmic scale if needed to emphasize smaller error values
    # ax.set_yscale('log')

    # Add legend
    ax.legend(fontsize=15)

    # Add grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout for better appearance
    plt.tight_layout()

    # set y ticks to fontsize 20
    ax.tick_params(axis='y', labelsize=15)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'error_bar_comparison.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved error bar comparison plot to {output_dir}/error_bar_comparison.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    parser.add_argument('--load_only', action='store_true', help='Load error data from csv files')
    args = parser.parse_args()
    if args.load_only:
        load_error_data(args)
    else:
        run_ransac(args)