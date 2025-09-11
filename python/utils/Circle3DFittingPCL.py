import numpy as np
from scipy.optimize import least_squares

"""
This code is written to fit a circle to a point cloud using the same implementation like the C++ code in PCL library.
The basic idea of 3d circle fitting in PCL is:
1. pick 3 points from the point cloud to form a plane;
2. project the points to the plane;
3. fit a circle to the projected points;
4. find inliers of the circle;
5. optimize the circle parameters using the inliers;
6. transform the circle center back to the 3D space;
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
    res = least_squares(
        cost_func,                # The cost function to minimize
        optimized_coeff,          # Initial guess for the parameters
        method='lm',              # Optimization method: Levenberg-Marquardt
        args=(pcd[:, inliers],)   # Additional arguments passed to cost_func (the inlier points)
    )

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

    print("pcn.shape: ", pcn.shape)
    print("randomid[:3]: ", randomid[:3])
    print("pcn[:,randomid[:3]].shape: ", pcn[:,randomid[:3]].shape)

    model_coeff=compute_model_coeff(pcn[:,randomid[:3]])
    distances = get_distance_to_model(pcn,model_coeff)
    inlier_id=np.nonzero(distances<1)[0]

    optimize_model_coeff = optimize_model_coeff(pcn, inlier_id, model_coeff)

    print("model_coeff: ", model_coeff)
    print("optimize_model_coeff", optimize_model_coeff)

    import Circle3DFitting as lsqfit

    res_lsq = lsqfit.lsq_fit_3d_circle(pcn)
    center = res_lsq['center']

    print("res_lsq: ", res_lsq)

    # some drawing function
