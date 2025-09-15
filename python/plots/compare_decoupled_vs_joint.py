import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple

# from python.utils.Circle3DFittingPCL import center

# Global font settings
# Prefer Arial but fall back to widely available sans-serif fonts
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['font.size'] = 15


def fit_circle_pcl(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a circle to a set of 3D points using PCL.
    """
    def compute_model_coeff(pcd):
        # Ensure the input point cloud is in the correct shape (3, N)
        if pcd.shape[0] < pcd.shape[1]:
            pcd = pcd.T

        # Select the first three points to define the plane and the circle
        p0 = pcd[:, 0]
        p1 = pcd[:, 1]
        p2 = pcd[:, 2]

        # Compute helper vectors between the points
        helper_v01 = p0 - p1
        helper_v02 = p0 - p2
        helper_v10 = p1 - p0
        helper_v12 = p1 - p2
        helper_v20 = p2 - p0
        helper_v21 = p2 - p1

        # Compute the normal vector to the plane defined by the three points
        common_helper_vec = np.cross(helper_v01, helper_v12)

        # Compute the denominator used in the barycentric coordinates
        common_dividend = 2 * np.linalg.norm(common_helper_vec) ** 2

        # Compute barycentric coordinates (alpha, beta, gamma) for the circle center
        alpha = (np.linalg.norm(helper_v12) ** 2 * helper_v01.dot(helper_v02)) / common_dividend
        beta = (np.linalg.norm(helper_v02) ** 2 * helper_v10.dot(helper_v12)) / common_dividend
        gamma = (np.linalg.norm(helper_v01) ** 2 * helper_v20.dot(helper_v21)) / common_dividend

        # Calculate the circle center as a weighted sum of the three points
        circle_center = alpha * p0 + beta * p1 + gamma * p2

        # Compute the radius as the distance from the center to one of the points
        circle_radius_vec = circle_center - p0
        circle_radius = np.linalg.norm(circle_radius_vec)

        # Compute the normalized normal vector of the circle's plane
        circlr_norm = common_helper_vec / np.linalg.norm(common_helper_vec)

        # Prepare the model coefficients array: [center_x, center_y, center_z, radius, normal_x, normal_y, normal_z]
        model_coeff = np.zeros((7,))

        model_coeff[:3] = circle_center.squeeze()
        model_coeff[3] = circle_radius
        model_coeff[4:] = circlr_norm.squeeze()

        return model_coeff

    def get_distance_to_model(pcd, model_coeff):
        """
        Compute the distance from each point in the point cloud to the fitted 3D circle.

        Parameters:
        - pcd: (N, 3) array of points (each row is a point)
        - model_coeff: array of 7 elements [center_x, center_y, center_z, radius, normal_x, normal_y, normal_z]

        Returns:
        - distances: (N,) array of distances from each point to the circle
        """
        # Extract circle center, normal, and radius from model coefficients
        C = model_coeff[:3].copy()      # Circle center (x, y, z)
        N = model_coeff[4:].copy()      # Normal vector of the circle's plane
        r = model_coeff[3]              # Circle radius

        if pcd.shape[0] < pcd.shape[1]:
            pcd=pcd.T

        distances=[]
        for i, p in enumerate(pcd):
            # Vector from center to point
            helper_vec = p - C

            # Project point onto the plane of the circle
            clambda = helper_vec.dot(N) / np.linalg.norm(N) ** 2
            p_proj = p + clambda * N

            # Vector from center to projected point
            helper_vec_2 = p_proj - C

            # Find the closest point K on the circle to the projected point
            K = C + r * helper_vec_2 / np.linalg.norm(helper_vec_2)

            # Distance from original point to the closest point on the circle
            distance_vec = p - K
            distances.append(np.linalg.norm(distance_vec))

        # Convert list to numpy array and remove any extra dimensions
        distances = np.array(distances).squeeze()

        return distances

    def cost_func(x, pcd):
        """
        Cost function for least squares optimization of 3D circle fitting.

        Parameters:
        - x: array-like, shape (7,)
            Model coefficients: [center_x, center_y, center_z, radius, normal_x, normal_y, normal_z]
        - pcd: array-like, shape (3, N)
            Point cloud data, each column is a 3D point.

        Returns:
        - fvec: array, shape (N,)
            Residuals: distances from each point to the closest point on the circle.
        """
        # Extract circle center from x
        C = np.array([x[0], x[1], x[2]]).squeeze()
        # Extract normal vector from x
        N = np.array([x[4], x[5], x[6]]).squeeze()
        # Extract radius from x
        r = x[3]
        # Initialize residual vector
        fvec = np.zeros((pcd.shape[1],))
        # Iterate over all points in the point cloud
        for i, pt in enumerate(pcd.T):
            # Vector from center to point
            P = pt - C
            # Project point onto the plane of the circle
            helper_vec = P - C
            plambda = -helper_vec.dot(N) / np.linalg.norm(N) ** 2
            p_proj = P + plambda * N
            # Vector from center to projected point
            helper_vec_2 = p_proj - C
            # Closest point K on the circle to the projected point
            K = C + r * helper_vec_2 / np.linalg.norm(helper_vec_2)
            # Distance from original point to the closest point on the circle
            distance_vec = P - K
            fvec[i] = (np.linalg.norm(distance_vec))

        return fvec

    def optimize_model_coeff(pcd, inliers, model_coeff):
        """
        Optimize the model coefficients for 3D circle fitting using least squares.

        Parameters:
        - pcd: array-like, shape (3, N)
            The full point cloud data, each column is a 3D point.
        - inliers: array-like
            Indices of inlier points to use for optimization.
        - model_coeff: array-like, shape (7,)
            Initial guess for the model coefficients: [center_x, center_y, center_z, radius, normal_x, normal_y, normal_z]

        Returns:
        - res.x if res.success else model_coeff: array-like, shape (7,)
            The optimized model coefficients if optimization succeeds, otherwise the initial coefficients.
        """
        # Make a copy of the initial model coefficients to avoid modifying the input
        optimized_coeff = model_coeff.copy()
        # Run least squares optimization using the Levenberg-Marquardt algorithm ('lm')
        # Only use the inlier points for fitting
        from scipy.optimize import least_squares
        res = least_squares(
            cost_func,                # The cost function to minimize
            optimized_coeff,          # Initial guess for the parameters
            method='lm',              # Optimization method: Levenberg-Marquardt
            args=(pcd[:, inliers],)   # Additional arguments passed to cost_func (the inlier points)
        )

        return res.x if res.success else model_coeff

    randomid = np.arange(0,points.shape[1])
    np.random.shuffle(randomid)

    # print("pcn.shape: ", points.shape)
    # print("randomid[:3]: ", randomid[:3])
    # print("pcn[:,randomid[:3]].shape: ", points[:,randomid[:3]].shape)

    model_coeff=compute_model_coeff(points[:,randomid[:3]])
    distances = get_distance_to_model(points,model_coeff)
    inlier_id=np.nonzero(distances<1)[0]

    optimize_model_coeff = optimize_model_coeff(points, inlier_id, model_coeff)

    # print("model_coeff: ", model_coeff)
    # print("optimize_model_coeff", optimize_model_coeff)

    center = optimize_model_coeff[:3]
    radius = optimize_model_coeff[3]
    normal = optimize_model_coeff[4:]

    return center, radius, normal


def fit_circle_2d(x, y, w=None):
    """
    Fit a circle to a set of 2D points using least squares.

    Parameters
    ----------
    x : array-like, shape (n_points,)
        x-coordinates of the points.
    y : array-like, shape (n_points,)
        y-coordinates of the points.
    w : array-like, shape (n_points,), optional
        Weights for each point. If None or not provided, unweighted fit is performed.

    Returns
    -------
    xc : float
        x-coordinate of the fitted circle center.
    yc : float
        y-coordinate of the fitted circle center.
    r : float
        Radius of the fitted circle.

    Notes
    -----
    The circle is defined as (x - xc)^2 + (y - yc)^2 = r^2.
    The method solves the linearized least squares problem for the circle parameters.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    n = x.size

    A = np.column_stack((x, y, np.ones(n)))
    b = x ** 2 + y ** 2

    # Modify A, b for weighted least squares if weights are provided
    if w is not None:
        w = np.asarray(w)
        if w.shape != x.shape:
            raise ValueError("Weights w must have the same shape as x and y")
        W = np.diag(w)
        A = W @ A
        b = W @ b

    # Solve by method of least squares
    c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r

def rodrigues_rot(P, n0, n1):
    """
    Rotate given points based on a starting and ending vector
    Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
    """
    # Ensure input is numpy array
    P = np.asarray(P)
    n0 = np.asarray(n0)
    n1 = np.asarray(n1)

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-12:
        # n0 and n1 are parallel or anti-parallel
        if np.allclose(n0, n1):
            # No rotation needed
            return P.copy()
        else:
            # 180 degree rotation: pick any orthogonal axis
            # Find a vector orthogonal to n0
            if not np.allclose(n0, [0, 0, 1]):
                ortho = np.cross(n0, [0, 0, 1])
            else:
                ortho = np.cross(n0, [0, 1, 0])
            ortho = ortho / np.linalg.norm(ortho)
            theta = np.pi
            k = ortho
    else:
        k = k / k_norm
        theta = np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        Pi = P[i]
        P_rot[i] = (
            Pi * np.cos(theta)
            + np.cross(k, Pi) * np.sin(theta)
            + k * np.dot(k, Pi) * (1 - np.cos(theta))
        )

    return P_rot

def circle_fitting_decoupled(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a circle to a set of 3D points using a decoupled approach:
    1. Fit a plane to the points using SVD.
    2. Project the points onto the best-fit plane.
    3. Fit a circle to the projected 2D points.
    4. Transform the circle center back to 3D coordinates.

    Parameters
    ----------
    points : np.ndarray
        An (N, 3) array of 3D points.

    Returns
    -------
    center_3d : np.ndarray
        The (3,) array representing the center of the fitted circle in 3D.
    radius : float
        The radius of the fitted circle.
    normal : np.ndarray
        The (3,) array representing the normal vector of the fitted plane.

    Notes
    -----
    - The function assumes that the input points are approximately coplanar.
    - The function will transpose the input if it is (3, N) instead of (N, 3).
    """
    # Ensure input is (N, 3)
    points = np.asarray(points)
    if points.ndim != 2 or 3 not in points.shape:
        raise ValueError("Input points must be a 2D array with shape (N, 3) or (3, N).")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError("Input points must have 3 columns representing x, y, z coordinates.")

    # (1) Fit plane by SVD for the mean-centered data
    P = points
    P_mean = P.mean(axis=0)
    P_centered = P - P_mean
    U, s, Vt = np.linalg.svd(P_centered)
    # Normal vector of fitting plane is given by last row of Vt (V transposed)
    normal = Vt[-1, :]
    d = -np.dot(P_mean, normal)  # d = -<p, n>

    # (2) Project points to coords X-Y in 2D plane
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

    # (3) Fit circle in new 2D coords
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

    # (4) Transform circle center back to 3D coords
    center_2d = np.array([[xc, yc, 0]])
    center_3d = rodrigues_rot(center_2d, [0, 0, 1], normal) + P_mean
    center_3d = center_3d.flatten()

    return center_3d, r, normal, d


def angle_between(u, v, n=None):
    """
    Compute the angle between two vectors u and v.

    If n is provided, the sign of the angle is determined by the orientation of the normal vector n.
    Otherwise, the unsigned angle is returned.

    Parameters
    ----------
    u : array_like
        First vector.
    v : array_like
        Second vector.
    n : array_like or None, optional
        Normal vector to define the sign of the angle. If None, returns the unsigned angle.

    Returns
    -------
    angle : float
        The angle between u and v in radians. If n is provided, the sign is determined by n.

    Notes
    -----
    The angle is computed using arctan2 for numerical stability.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        n = np.asarray(n)
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))


def set_axes_equal_3d(ax):
    """
    Set 3D plot axes to have equal scale.

    This is a workaround for Matplotlib's set_aspect('equal') and axis('equal'),
    which do not work for 3D axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes to set equal aspect ratio.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 0] - limits[:, 1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def generate_circle_by_vectors(t, C, r, n, u):
    """
    Generate points on a circle in 3D space using two orthonormal vectors.

    The circle is parameterized as:
        P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C

    Parameters
    ----------
    t : array_like
        Array of parameter values (angles in radians).
    C : array_like
        Center of the circle (3,).
    r : float
        Radius of the circle.
    n : array_like
        Normal vector of the circle's plane (3,).
    u : array_like
        A unit vector in the plane of the circle, orthogonal to n (3,).

    Returns
    -------
    P_circle : ndarray
        Array of shape (len(t), 3) containing the points on the circle.
    """
    t = np.asarray(t)
    C = np.asarray(C)
    n = np.asarray(n)
    u = np.asarray(u)
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    P_circle = (r * np.cos(t)[:, np.newaxis] * u +
                r * np.sin(t)[:, np.newaxis] * v +
                C)
    return P_circle


def generate_circle_by_angles(t, C, r, theta, phi):
    """
    Generate points on a circle in 3D space using spherical angles.

    The circle is parameterized as:
        P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C

    where n is the normal vector defined by (theta, phi) in spherical coordinates,
    and u is a unit vector orthogonal to n.

    Parameters
    ----------
    t : array_like
        Array of parameter values (angles in radians).
    C : array_like
        Center of the circle (3,).
    r : float
        Radius of the circle.
    theta : float
        Polar angle (from z-axis) in radians.
    phi : float
        Azimuthal angle (from x-axis) in radians.

    Returns
    -------
    P_circle : ndarray
        Array of shape (len(t), 3) containing the points on the circle.
    """
    t = np.asarray(t)
    C = np.asarray(C)
    # Normal vector in spherical coordinates
    n = np.array([np.cos(phi) * np.sin(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(theta)])
    # Find a vector u orthogonal to n
    # If n is close to z-axis, use x-axis as reference, else use z-axis
    if np.allclose(n, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(n, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    P_circle = (r * np.cos(t)[:, np.newaxis] * u +
                r * np.sin(t)[:, np.newaxis] * v +
                C)
    return P_circle


def demo_circle_fitting_decoupled_and_joint():
    #-------------------------------------------------------------------------------
    # Generating circle
    #-------------------------------------------------------------------------------
    r = 2.5               # Radius
    C = np.array([3,3,4])    # Center
    theta = 45/180*np.pi     # Azimuth
    phi   = -30/180*np.pi    # Zenith

    t = np.linspace(0, 2*np.pi, 100)
    P_gen = generate_circle_by_angles(t, C, r, theta, phi)

    #-------------------------------------------------------------------------------
    # Cluster of points
    #-------------------------------------------------------------------------------
    t = np.linspace(-np.pi, -0.25*np.pi, 100)
    n = len(t)
    P = generate_circle_by_angles(t, C, r, theta, phi)

    # Add some random noise to the points
    P += np.random.normal(size=P.shape) * 0.1

    #------------------ Decoupled Fitting ------------------
    center_fit, r_fit, normal_fit, d_fit = circle_fitting_decoupled(P)
    print('--- Decoupled Fitting ---')
    print('Fitting plane: n = %s' % np.array_str(normal_fit, precision=4))
    print('real circle: center = %s, r = %.4g' % (np.array_str(C, precision=4), r))
    print('fitting circle: center = %s, r = %.4g' % (np.array_str(center_fit, precision=4), r_fit))

    t = np.linspace(0, 2*np.pi, 100)
    u = P[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal_fit, u)

    #--- Generate points for fitting arc
    v = P[-1] - C
    theta_arc = angle_between(u, v, normal_fit)
    t_arc = np.linspace(0, theta_arc, 100)
    P_fitarc = generate_circle_by_vectors(t_arc, C, r, normal_fit, u)

    print('Fitting arc: u = %s, θ = %.4g' % (np.array_str(u, precision=4), theta_arc*180/np.pi))

    #------------------ Joint Fitting ------------------
    try:
        center_joint, r_joint, normal_joint = circle_fitting_joint(P)
        print('\n--- Joint Fitting ---')
        print('real circle: center = %s, r = %.4g' % (np.array_str(C, precision=4), r))
        print('fitting circle: center = %s, r = %.4g' % (np.array_str(center_joint, precision=4), r_joint))
        t_joint = np.linspace(0, 2*np.pi, 100)
        # For joint, use the same u as decoupled for visualization
        P_fitcircle_joint = generate_circle_by_vectors(t_joint, center_joint, r_joint, normal_joint, u)
    except Exception as e:
        print("Joint fitting failed:", e)
        center_joint, r_joint, normal_joint = None, None, None
        P_fitcircle_joint = None

    #------------------ Plotting ------------------
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(*P_gen.T, color='y', lw=3, label='Generating circle')
    ax.plot(*P.T, ls='', marker='o', alpha=0.5, label='Cluster points P')

    #--- Plot fitting plane (decoupled)
    xx, yy = np.meshgrid(np.linspace(0,6,11), np.linspace(0,6,11))
    zz = (-normal_fit[0]*xx - normal_fit[1]*yy - d_fit) / normal_fit[2]
    ax.plot_surface(xx, yy, zz, rstride=2, cstride=2, color='y' ,alpha=0.2, shade=False)

    #--- Plot decoupled fitting circle and arc
    ax.plot(*P_fitcircle.T, color='k', ls='--', lw=2, label='Fitting circle (decoupled)')
    ax.plot(*P_fitarc.T, color='k', ls='-', lw=3, label='Fitting arc (decoupled)')

    #--- Plot joint fitting circle if available
    if P_fitcircle_joint is not None:
        ax.plot(*P_fitcircle_joint.T, color='r', ls='-.', lw=2, label='Fitting circle (joint)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.set_aspect('equal', 'datalim')
    set_axes_equal_3d(ax)
    plt.show()


def skew(x: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a 3D vector.

    Parameters
    ----------
    x : np.ndarray
        A 1D array of shape (3,) representing a 3D vector.

    Returns
    -------
    np.ndarray
        A (3, 3) skew-symmetric matrix corresponding to vector x.
    """
    xx = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
    return xx

def outer_product(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the outer product for circle fitting in 3D conformal geometry.

    Parameters
    ----------
    y : np.ndarray
        A (5,) or (5,1) array.
    x : np.ndarray
        A (5,) or (5,1) array.

    Returns
    -------
    np.ndarray
        A (10,) array representing the outer product result.
    """
    if y.shape != (5, 1):
        y = y.reshape((5, 1))
    if x.shape != (5, 1):
        x = x.reshape((5, 1))

    yx = skew(y[:3, 0])
    yoi = y[3, 0] * np.eye(3)
    yinfi = y[4, 0] * np.eye(3)
    A = np.zeros((10, 5))
    A[:3, :3] = yx
    A[3:6, 0:3] = yoi
    A[3:6, 3] = -y[0:3, 0]
    A[6:9, 0:3] = -yinfi
    A[6:9, 4] = y[0:3, 0]
    A[9, 3] = -y[4, 0]
    A[9, 4] = y[3, 0]
    val = A @ x

    return val

def construct_p_from_batch(ps: np.ndarray) -> np.ndarray:
    """
    Construct the matrix P from a batch of 3D points for joint circle fitting.

    Parameters
    ----------
    ps : np.ndarray
        An (N, 3) array of 3D points.

    Returns
    -------
    np.ndarray
        A (5, 5) matrix used in the joint circle fitting algorithm.
    """
    N = ps.shape[0]
    D = np.zeros((5, N))
    for i, p in enumerate(ps):
        D[:3, i] = ps[i, :].T
        D[3, i] = 1
        D[4, i] = 0.5 * np.linalg.norm(ps[i, :]) ** 2
    M = np.block([
        [np.eye(3), np.zeros((3, 2))],
        [np.zeros((1, 4)), -1],
        [np.zeros((1, 3)), -1, 0]
    ])
    P2 = D @ D.T @ M
    P2 = P2 / N
    return P2

def construct_a_single_p(p: np.ndarray) -> np.ndarray:
    """
    Construct the matrix P for a single 3D point for joint circle fitting.

    Parameters
    ----------
    p : np.ndarray
        A (3,) or (3,1) array representing a 3D point.

    Returns
    -------
    np.ndarray
        A (5, 5) matrix for the single point.
    """
    if p.shape != (3, 1):
        p = p.reshape((3, 1))
    P = np.zeros((5, 5))
    P[:3, :3] = p @ p.T
    pnorm = np.linalg.norm(p)
    P[:3, 3] = -0.5 * pnorm ** 2 * p.squeeze()
    P[:3, 4] = -p.squeeze()
    P[3, :3] = p.T
    P[3, 3] = -0.5 * pnorm ** 2
    P[3, 4] = -1
    P[4, :3] = 0.5 * pnorm ** 2 * p.T
    P[4, 3] = -0.25 * pnorm ** 4
    P[4, 4] = -0.5 * pnorm ** 2

    return P

def recover_circle_parameter(e: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Recover the circle parameters (center, radius, normal) from the solution vector.

    Parameters
    ----------
    e : np.ndarray
        A (10,) array representing the solution vector from the joint circle fitting.
    verbose : bool, optional
        If True, print intermediate results.

    Returns
    -------
    center : np.ndarray
        The (3,) array representing the center of the fitted circle in 3D.
    radius : float
        The radius of the fitted circle.
    normal : np.ndarray
        The (3,) array representing the normal vector of the fitted plane.
    """
    if e.shape != (10,):
        e = e.squeeze()
    ei = e[:3]
    eoi = e[3:6]
    einfi = e[6:9]
    eoinf = -e[9]

    alpha = np.linalg.norm(eoi)
    n1 = -eoi / alpha
    n = -eoi

    B0 = eoinf
    B1 = ei[0]
    B2 = ei[1]
    B3 = ei[2]

    c = np.array([
        [B0, -B3, B2],
        [B3, B0, -B1],
        [-B2, B1, B0]
    ]) @ n.reshape((3, 1)) / (np.linalg.norm(n) ** 2)
    c = c.squeeze()

    radius_square = np.linalg.norm(c) ** 2 - 2 * n1.dot(einfi) / alpha - 2 * (c.dot(n1)) ** 2
    radius = np.sqrt(radius_square)
    return (c, radius, n1)


def circle_fitting_joint(points: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a circle to a set of 3D points using a joint approach.

    Parameters
    ----------
    points : np.ndarray
        An (N, 3) array of 3D points.

    Returns
    -------
    center_3d : np.ndarray
        The (3,) array representing the center of the fitted circle in 3D.
    radius : float
        The radius of the fitted circle.
    normal : np.ndarray
        The (3,) array representing the normal vector of the fitted plane.

    Notes
    -----
    - The function assumes that the input points are approximately coplanar.
    - The function will transpose the input if it is (3, N) instead of (N, 3).
    """
    # Ensure input is (N, 3)
    points = np.asarray(points)
    if points.ndim != 2 or 3 not in points.shape:
        raise ValueError("Input points must be a 2D array with shape (N, 3) or (3, N).")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError("Input points must have 3 columns representing x, y, z coordinates.")

    M = construct_p_from_batch(points)
    evals, evecs = np.linalg.eig(M)
    indx = np.argsort(evals)
    indx1 = (evals[indx] > 0).nonzero()[0]

    sol1 = evecs[:, indx[indx1[0]]]
    sol2 = evecs[:, indx[indx1[1]]]

    sol_final = outer_product(sol2, sol1)
    if verbose:
        print(sol_final)

    center, radius, normal = recover_circle_parameter(sol_final, verbose)
    return center, radius, normal

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _pick_u_from_normal(n: np.ndarray) -> np.ndarray:
    n = _normalize(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u = _normalize(u)
    if np.linalg.norm(u) < 1e-12:
        u = np.array([1.0, 0.0, 0.0])
    return u

def _mean_radial_residual(points: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float) -> float:
    n = _normalize(normal)
    vecs = points - center.reshape(1, 3)
    d = vecs @ n
    proj = points - np.outer(d, n)
    rr = np.linalg.norm(proj - center.reshape(1, 3), axis=1)
    return float(np.mean(np.abs(rr - radius)))

def _angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    n1 = _normalize(n1)
    n2 = _normalize(n2)
    c = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _simulate_case_offplane_noise(C: np.ndarray, r: float, n: np.ndarray, num: int = 100,
                                  noise_std: float = 0.1,
                                  seed: int = 7) -> np.ndarray:
    """Generate circle points and add isotropic 3D Gaussian noise to all points."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2*np.pi, size=num)
    nn = _normalize(n)
    u = _pick_u_from_normal(nn)
    P = generate_circle_by_vectors(t, C, r, nn, u)
    return P + rng.normal(0.0, noise_std, size=P.shape)

def _simulate_case_limited_arc_nonuniform(C: np.ndarray, r: float, n: np.ndarray, num: int = 100,
                                          arc_deg: float = 70.0, noise_std: float = 0.1,
                                          bias_power: float = 2.0, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arc = np.radians(arc_deg)
    u_rand = rng.random(num)
    # bias density to one end of arc
    t = (u_rand ** bias_power) * arc - 0.2*arc
    u = _pick_u_from_normal(n)
    P = generate_circle_by_vectors(t, C, r, n, u)
    return P + rng.normal(0, noise_std, size=P.shape)

def _simulate_case_sparse_nonuniform(C: np.ndarray, r: float, n: np.ndarray,
                                     num: int = 12, noise_std: float = 0.1,
                                     seed: int = 19) -> np.ndarray:
    """Generate few (6-20) points on non-uniform angles, then add small 3D noise.

    Uses 2-3 angular clusters to mimic calibration dots distributed unevenly.
    """
    rng = np.random.default_rng(seed)
    num = int(num)
    if num < 6:
        num = 6
    if num > 20:
        num = 20

    # choose 2 or 3 clusters
    k = rng.integers(2, 8)
    centers = rng.uniform(0, 2*np.pi, size=k)
    widths = rng.uniform(np.radians(6), np.radians(20), size=k)

    # allocate points per cluster (non-uniform)
    weights = rng.random(k)
    weights = weights / weights.sum()
    counts = np.maximum(1, np.round(weights * num)).astype(int)
    # adjust to match exactly num
    while counts.sum() > num:
        i = rng.integers(0, k)
        if counts[i] > 1:
            counts[i] -= 1
    while counts.sum() < num:
        i = rng.integers(0, k)
        counts[i] += 1

    t_list = []
    for ci, wi, cnt in zip(centers, widths, counts):
        t_list.append(rng.normal(ci, wi, size=cnt))
    t = np.concatenate(t_list)

    nn = _normalize(n)
    u = _pick_u_from_normal(nn)
    P = generate_circle_by_vectors(t, C, r, nn, u)
    return P + rng.normal(0.0, noise_std, size=P.shape)

def _simulate_case_two_side_different_intervals(
    C: np.ndarray, r: float, n: np.ndarray,
    num_side1: int = 5, arc1_deg: float = 180.0,
    noise_std: float = 0.1, seed: int = 21
) -> np.ndarray:
    """
    Simulate two symmetric arcs on a circle, with non-uniform but symmetric angular intervals.
    For every angle t in [0, 2pi), if t is sampled, so is (2pi - t).
    The angular intervals are not uniform, but vary (e.g., pi/6, pi/4, pi*2/3, ...).
    The two arcs can have different angular extents and different numbers of points.
    """
    rng = np.random.default_rng(seed)
    nn = _normalize(n)
    u = _pick_u_from_normal(nn)

    arc1 = np.radians(arc1_deg)

    # Helper to generate non-uniform, symmetric angles for an arc centered at center_angle
    def symmetric_nonuniform_arc(center_angle, arc_extent, num_points, jitter_scale=0.15):
        # Generate non-uniform intervals (randomized, but positive, sum to arc_extent/2)
        if num_points < 2:
            return np.array([center_angle])
        half = num_points // 2
        # Generate random positive intervals for half the arc
        intervals = rng.uniform(0.8, 1.2, size=half)
        intervals = intervals / intervals.sum() * (arc_extent / 2)
        # Cumulative sum to get angles from center to one end
        t_half = np.cumsum(intervals)
        t_half = np.insert(t_half, 0, 0.0)  # include center
        # Mirror to get symmetric angles
        t_full = np.concatenate([center_angle - t_half[::-1], center_angle + t_half[1:]])
        # If odd, add center point only once
        if num_points % 2 == 1:
            t_full = np.concatenate([center_angle - t_half[::-1][1:], [center_angle], center_angle + t_half[1:]])
        # Add small jitter to each angle (but keep symmetry)
        jitter = rng.normal(0, jitter_scale * (arc_extent / num_points), size=t_full.shape[0] // 2)
        t_full[:t_full.shape[0] // 2] += jitter
        t_full[-(t_full.shape[0] // 2):] -= jitter[::-1]
        return t_full

    # Arc 1: centered at 0
    t = symmetric_nonuniform_arc(0.0, arc1, num_side1, jitter_scale=0.12)

    # Ensure all angles are in [0, 2pi)
    t = np.mod(t, 2 * np.pi)
    P = generate_circle_by_vectors(t, C, r, nn, u)
    return P + rng.normal(0.0, noise_std, size=P.shape)

_LIGHT = {
    'points': '#000000',    # black
    'dec': '#85c1e9',       # teal
    'joint': '#f9e79f',     # light orange
    'true': '#bb8fce',      # lavender
}

def _plot_compare(ax, P: np.ndarray, C: np.ndarray, r: float, n: np.ndarray, title: str):
    # Import improved joint fitting
    try:
        from cga_joint_fitting import cga_robust_circle_fitting_joint
        C_joint, r_joint, n_joint = cga_robust_circle_fitting_joint(P)
    except ImportError:
        C_joint, r_joint, n_joint = circle_fitting_joint(P)

    # fits
    # C_dec, r_dec, n_dec, _ = circle_fitting_decoupled(P)
    print(P.shape)
    C_dec, r_dec, n_dec = fit_circle_pcl(P.T)

    print("Classical fit: center =", C_dec, "radius =", r_dec, "normal =", n_dec)
    print("Proposed fit: center =", C_joint, "radius =", r_joint, "normal =", n_joint)

    # metrics
    m_center_dec = np.linalg.norm(C_dec - C)
    m_center_joint = np.linalg.norm(C_joint - C)
    m_r_dec = abs(r_dec - r)
    m_r_joint = abs(r_joint - r)

    # draw points
    ax.scatter(P[:,0], P[:,1], P[:,2], s=8, c=_LIGHT['points'], alpha=0.6, depthshade=False, label='points')

    # draw true circle
    u_true = _pick_u_from_normal(n)
    tt = np.linspace(0, 2*np.pi, 200)
    P_true = generate_circle_by_vectors(tt, C, r, n, u_true)
    ax.plot(*P_true.T, color=_LIGHT['true'], alpha=0.7, lw=1.5, label='true')

    # draw decoupled circle
    u_dec = _pick_u_from_normal(n_dec)
    P_c_dec = generate_circle_by_vectors(tt, C_dec, r_dec, n_dec, u_dec)
    ax.plot(*P_c_dec.T, color=_LIGHT['dec'], alpha=0.95, lw=2.0, label='Classical')

    # draw joint circle
    u_joint = _pick_u_from_normal(n_joint)
    P_c_joint = generate_circle_by_vectors(tt, C_joint, r_joint, n_joint, u_joint)
    ax.plot(*P_c_joint.T, color=_LIGHT['joint'], alpha=0.95, lw=2.0, label='Proposed')

    ax.set_title(title, color='#2c3e50')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_facecolor('white')
    ax.grid(False)
    set_axes_equal_3d(ax)
    # Consistent, clear 3D view angle
    ax.view_init(elev=12, azim=-30)
    # No per-axes legend; a single global legend will be added in run_two_cases

    text = (
        f"Classical  |Δc|={m_center_dec:.3f}, Δr={m_r_dec:.3f}\n"
        f"Proposed       |Δc|={m_center_joint:.3f}, Δr={m_r_joint:.3f}"
    )
    ax.text2D(0.03, 0.97, text, transform=ax.transAxes, fontsize=9, va='top', ha='left', color='#34495e')

def run_two_cases(save_path: str | None = None):
    # ground-truth circle
    rng = np.random.default_rng(42)
    C = rng.uniform(-2, 2, size=3)
    r = rng.uniform(1.0, 5.0)
    n = _normalize(rng.normal(size=3))

    # simulate
    P1 = _simulate_case_offplane_noise(C, r, n)
    P2 = _simulate_case_limited_arc_nonuniform(C, r, n)
    P3 = _simulate_case_sparse_nonuniform(C, r, n, num=14)
    P4 = _simulate_case_two_side_different_intervals(C, r, n,
                                                     num_side1=20,
                                                     arc1_deg=200.0)

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_compare(ax1, P1, C, r, n, 'Isotropic Noise')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_compare(ax2, P2, C, r, n, 'Limited Arc')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_compare(ax3, P3, C, r, n, 'Sparse Points')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_compare(ax4, P4, C, r, n, 'Symmetric Distribution')

    # subtle, readable ticks
    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(colors='#7f8c8d')

    # Single, global legend: gather unique handles/labels from the first axis
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4, frameon=True)
    if leg is not None:
        leg.get_frame().set_alpha(0.2)
        leg.get_frame().set_edgecolor('#bdc3c7')

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path is None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'compare_decoupled_vs_joint.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved figure to: {save_path}')

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
	run_two_cases()

    # generate points
    # rng = np.random.default_rng(42)
    # C = rng.uniform(-2, 2, size=3)
    # r = rng.uniform(1.0, 5.0)
    # n = _normalize(rng.normal(size=3))
    # P = _simulate_case_offplane_noise(C, r, n)
    # center, radius, normal = fit_circle_pcl(P.T)
    # print("center: ", center)
    # print("radius: ", radius)
    # print("normal: ", normal)
    # print("np.linalg.norm(center - C): ", np.linalg.norm(center - C))
    # print("abs(radius - r): ", abs(radius - r))
    # print("np.linalg.norm(normal - n): ", np.linalg.norm(normal - n))
