import numpy as np
from typing import Tuple

def cga_circle_fitting_joint(points: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Improved joint circle fitting using conformal geometric algebra.

    This implementation fixes several issues in the original CGA approach:
    1. Proper eigenvalue selection (smallest positive eigenvalues)
    2. Better numerical stability
    3. Robust parameter recovery
    4. Fallback mechanisms
    """
    points = np.asarray(points)
    if points.ndim != 2 or 3 not in points.shape:
        raise ValueError("Input points must be a 2D array with shape (N, 3) or (3, N).")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError("Input points must have 3 columns representing x, y, z coordinates.")

    N = points.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points to fit a circle")

    # Center the points for better numerical stability
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Construct the conformal representation matrix
    M = construct_cga_p_from_batch(centered_points)

    # Find eigenvalues and eigenvectors
    try:
        evals, evecs = np.linalg.eig(M)
        evals = np.real(evals)  # Ensure real eigenvalues
        evecs = np.real(evecs)  # Ensure real eigenvectors
    except np.linalg.LinAlgError:
        # Fallback to SVD if eig fails
        U, s, Vt = np.linalg.svd(M)
        evals = s
        evecs = U

    # Select the two smallest positive eigenvalues (or least negative)
    positive_mask = evals > 0
    if np.sum(positive_mask) >= 2:
        # Use smallest positive eigenvalues
        sorted_indices = np.argsort(evals)
        positive_indices = sorted_indices[evals[sorted_indices] > 0][:2]
    else:
        # Fallback: use smallest eigenvalues (even if negative)
        sorted_indices = np.argsort(evals)
        positive_indices = sorted_indices[:2]

    if len(positive_indices) < 2:
        raise ValueError("Insufficient positive eigenvalues for circle fitting")

    sol1 = evecs[:, positive_indices[0]]
    sol2 = evecs[:, positive_indices[1]]

    # Compute outer product
    sol_final = cga_outer_product(sol2, sol1)

    if verbose:
        print(f"Eigenvalues: {evals[positive_indices]}")
        print(f"Solution vector: {sol_final}")

    # Recover circle parameters
    center, radius, normal = cga_recover_circle_parameter(sol_final, verbose)

    # Transform center back to original coordinate system
    center = center + centroid

    return center, radius, normal

def construct_cga_p_from_batch(ps: np.ndarray) -> np.ndarray:
    """
    Improved construction of the conformal representation matrix.
    """
    N = ps.shape[0]

    # Create the conformal representation matrix D
    D = np.zeros((5, N))
    for i, p in enumerate(ps):
        D[:3, i] = p.T
        D[3, i] = 1.0
        D[4, i] = 0.5 * np.dot(p, p)  # More stable than np.linalg.norm(p)**2

    # Metric matrix for conformal space
    M = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, -1],
        [0, 0, 0, -1, 0]
    ])

    # Compute P = D * D^T * M
    P = D @ D.T @ M
    P = P / N  # Normalize by number of points

    return P

def cga_outer_product(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Improved outer product computation with better numerical stability.
    """
    if y.shape != (5,):
        y = y.reshape(5)
    if x.shape != (5,):
        x = x.reshape(5)

    # Skew-symmetric matrix for cross product
    def skew_matrix(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    # Compute outer product components
    yx = skew_matrix(y[:3])
    yoi = y[3] * np.eye(3)
    yinfi = y[4] * np.eye(3)

    # Build the 10x5 matrix A
    A = np.zeros((10, 5))
    A[:3, :3] = yx
    A[3:6, :3] = yoi
    A[3:6, 3] = -y[:3]
    A[6:9, :3] = -yinfi
    A[6:9, 4] = y[:3]
    A[9, 3] = -y[4]
    A[9, 4] = y[3]

    result = A @ x
    return result

def cga_recover_circle_parameter(e: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Improved parameter recovery with better numerical stability and validation.
    """
    if e.shape != (10,):
        e = e.squeeze()

    # Extract components
    ei = e[:3]
    eoi = e[3:6]
    einfi = e[6:9]
    eoinf = -e[9]

    # Compute normal vector
    alpha = np.linalg.norm(eoi)
    if alpha < 1e-12:
        raise ValueError("Degenerate normal vector in CGA solution")

    n1 = -eoi / alpha
    n = -eoi

    # Compute center using improved formula
    B0 = eoinf
    B1 = ei[0]
    B2 = ei[1]
    B3 = ei[2]

    # More stable center computation
    n_norm_sq = np.dot(n, n)
    if n_norm_sq < 1e-12:
        raise ValueError("Zero normal vector magnitude")

    # Cross product matrix for n
    n_cross = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])

    # Center computation: c = (B0*I + B1*i + B2*j + B3*k) * n / |n|^2
    # where I is identity, i,j,k are cross product matrices
    B_matrix = B0 * np.eye(3) + B1 * np.array([[0,0,0],[0,0,-1],[0,1,0]]) + \
               B2 * np.array([[0,0,1],[0,0,0],[-1,0,0]]) + \
               B3 * np.array([[0,-1,0],[1,0,0],[0,0,0]])

    c = (B_matrix @ n) / n_norm_sq

    # Compute radius with improved stability
    c_dot_n = np.dot(c, n1)
    radius_sq = np.dot(c, c) - 2 * np.dot(n1, einfi) / alpha - 2 * c_dot_n**2

    if radius_sq < 0:
        if verbose:
            print(f"Warning: Negative radius squared: {radius_sq}")
        radius_sq = max(radius_sq, 1e-12)  # Prevent negative radius

    radius = np.sqrt(radius_sq)

    if verbose:
        print(f"Center: {c}")
        print(f"Radius: {radius}")
        print(f"Normal: {n1}")

    return c, radius, n1

def cga_robust_circle_fitting_joint(points: np.ndarray, max_iterations: int = 3) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Robust joint fitting with multiple attempts and fallback strategies.
    """
    best_result = None
    best_error = float('inf')

    for attempt in range(max_iterations):
        try:
            # Add small random perturbation to break symmetries
            if attempt > 0:
                noise_scale = 1e-6 * attempt
                perturbed_points = points + np.random.normal(0, noise_scale, points.shape)
            else:
                perturbed_points = points

            center, radius, normal = cga_circle_fitting_joint(perturbed_points)

            # Validate result
            if radius > 0 and np.all(np.isfinite(center)) and np.all(np.isfinite(normal)):
                # Compute fitting error
                error = compute_circle_fitting_error(points, center, radius, normal)

                if error < best_error:
                    best_error = error
                    best_result = (center, radius, normal)

        except Exception as e:
            if attempt == max_iterations - 1:
                print(f"All attempts failed. Last error: {e}")
            continue

    if best_result is None:
        raise ValueError("Failed to fit circle with joint method")

    return best_result

def compute_circle_fitting_error(points: np.ndarray, center: np.ndarray, radius: float, normal: np.ndarray) -> float:
    """
    Compute the fitting error for validation.
    """
    n = normal / np.linalg.norm(normal)
    vecs = points - center.reshape(1, 3)

    # Distance to plane
    dist_plane = vecs @ n

    # Project onto plane
    proj = points - np.outer(dist_plane, n)

    # Distance to circle center in plane
    dist_circle = np.linalg.norm(proj - center.reshape(1, 3), axis=1)

    # Total error: combination of plane distance and circle distance
    plane_error = np.mean(np.abs(dist_plane))
    circle_error = np.mean(np.abs(dist_circle - radius))

    return plane_error + circle_error

# Test function
def test_improved_joint_fitting():
    """Test the improved joint fitting on synthetic data."""
    np.random.seed(42)

    # Generate test circle
    center = np.array([1.0, 2.0, 3.0])
    radius = 2.5
    normal = np.array([0.3, 0.7, 0.6])
    normal = normal / np.linalg.norm(normal)

    # Generate points on circle
    t = np.linspace(0, 2*np.pi, 50)
    u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    points = center + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)

    # Add noise
    points += np.random.normal(0, 0.1, points.shape)

    try:
        fitted_center, fitted_radius, fitted_normal = cga_robust_circle_fitting_joint(points)

        print("Test Results:")
        print(f"True center: {center}")
        print(f"Fitted center: {fitted_center}")
        print(f"Center error: {np.linalg.norm(fitted_center - center)}")

        print(f"True radius: {radius}")
        print(f"Fitted radius: {fitted_radius}")
        print(f"Radius error: {abs(fitted_radius - radius)}")

        print(f"True normal: {normal}")
        print(f"Fitted normal: {fitted_normal}")
        cos_angle = np.dot(normal, fitted_normal) / (np.linalg.norm(normal) * np.linalg.norm(fitted_normal))
        angle_error = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        print(f"Normal angle error: {angle_error} degrees")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_improved_joint_fitting()
