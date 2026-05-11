import numpy as np
from scipy.spatial import KDTree
import scipy.sparse as sp


def _build_laplacian(points, k=12):
    """Builds a normalized graph Laplacian from a KNN graph."""
    tree = KDTree(points)
    N = len(points)
    distances, indices = tree.query(points, k=k)
    sigma = np.mean(distances[:, 1]).clip(1e-8)
    rows = np.repeat(np.arange(N), k - 1)
    cols = indices[:, 1:].flatten()
    weights = np.exp(-distances[:, 1:].flatten() ** 2 / (2 * sigma ** 2))
    W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = (W + W.T) / 2
    degree = np.asarray(W.sum(axis=1)).flatten().clip(1e-8)
    D_inv = sp.diags(1.0 / degree)
    L_norm = D_inv @ (sp.diags(degree) - W)
    return L_norm


def laplacian_smooth(points, k=12, iterations=1, lam=0.5):
    """Standard Laplacian smoothing. Tends to shrink the shape over iterations."""
    L = _build_laplacian(points, k=k)
    p = points.copy()
    for _ in range(iterations):
        p = p - lam * (L @ p)
    return p


def _elliptic_gabriel_neighbors(points, k=40, alpha=0.75):
    """
    Approximate elliptic Gabriel neighborhood from the paper.

    For each point p and each kNN candidate q, keep q only if no closer
    neighbor lies inside the ellipsoid centered at the midpoint of pq with
    semiaxes (d, alpha*d, alpha*d), where d = ||p-q|| / 2.
    """
    tree = KDTree(points)
    _, indices = tree.query(points, k=k)
    neighbors = indices[:, 1:].copy()
    n_points, n_neighbors = neighbors.shape

    for i in range(n_points):
        p = points[i]
        row = neighbors[i]

        for j in range(n_neighbors - 1, -1, -1):
            q_idx = row[j]
            if q_idx < 0:
                continue

            q = points[q_idx]
            diff = q - p
            dist = np.linalg.norm(diff)
            if dist <= 1e-12:
                row[j] = -1
                continue

            midpoint = 0.5 * (p + q)
            x_axis = diff / dist
            d = 0.5 * dist
            keep = True

            for h in range(j):
                cand_idx = row[h]
                if cand_idx < 0:
                    continue

                cand = points[cand_idx] - midpoint
                x = np.dot(cand, x_axis)
                radial_sq = np.dot(cand, cand) - x * x
                if x * x + radial_sq / (alpha * alpha) < d * d:
                    keep = False
                    break

            if not keep:
                row[j] = -1

    return neighbors


def _egt_step(points, neighbors, factor):
    """One Gaussian-weighted EGT update with a Taubin factor."""
    out = points.copy()

    for i in range(len(points)):
        nbr_idx = neighbors[i]
        valid = nbr_idx[nbr_idx >= 0]
        if len(valid) == 0:
            continue

        p = points[i]
        nbrs = points[valid]
        disp = nbrs - p
        dist_sq = np.sum(disp * disp, axis=1)
        weights = np.exp(-dist_sq)
        weight_sum = weights.sum()
        if weight_sum <= 1e-12:
            continue

        delta = (weights[:, None] * disp).sum(axis=0) / weight_sum
        out[i] = p + factor * delta

    return out


def taubin_smooth(points, k=40, iterations=20, lam=0.63, mu=-0.64, alpha=0.75):
    """
    Elliptic Gabriel Taubin smoothing from Agathos et al. (2022).

    This replaces the plain kNN Laplacian Taubin update with:
      1. a fixed approximate elliptic Gabriel neighborhood built from kNN
      2. Gaussian weights w_ij = exp(-||p_i - p_j||^2)
      3. alternating lambda / mu Taubin steps to reduce shrinkage

    Parameters mirror the paper's notation:
      k     : number of nearest-neighbor candidates used to build EGN
      alpha : ellipsoid aspect-ratio parameter for the Gabriel test
      lam   : smoothing factor
      mu    : inflation factor, typically slightly more negative than -lam
    """
    p = points.copy()
    neighbors = _elliptic_gabriel_neighbors(p, k=k, alpha=alpha)

    for _ in range(iterations):
        p = _egt_step(p, neighbors, lam)
        p = _egt_step(p, neighbors, mu)
    return p


def gaussian_smooth(points, k=12, sigma_factor=1.0, iterations=1):
    """
    Gaussian-weighted neighbor averaging.
    Higher sigma_factor = wider kernel = more smoothing.
    After smoothing, match the original centroid and overall scale to
    counteract shrinkage.
    """
    p_orig = points.copy()
    p = points.copy()
    tree = KDTree(p)
    distances, indices = tree.query(p, k=k)
    sigma = np.mean(distances[:, 1]).clip(1e-8) * sigma_factor
    weights = np.exp(-distances[:, 1:] ** 2 / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        neighbor_pts = p[indices[:, 1:]]
        p = (weights[:, :, None] * neighbor_pts).sum(axis=1)

    centroid_orig = p_orig.mean(axis=0)
    centroid_smooth = p.mean(axis=0)
    scale_orig = np.std(p_orig)
    scale_smooth = np.std(p).clip(1e-8)
    scale = scale_orig / scale_smooth
    p = (p - centroid_smooth) * scale + centroid_orig

    return p


def _estimate_normals_pca(points, indices):
    """Estimate one normal per point from its KNN neighborhood via PCA."""
    center = points.mean(axis=0)
    nbrs = points[indices[:, 1:]]
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(centered.shape[1], 1)
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    # Keep normal orientation roughly outward for stable inflation.
    radial = points - center
    flip_mask = np.sum(normals * radial, axis=1) < 0
    normals[flip_mask] *= -1.0

    norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return normals / norms


def normal_aware_smooth(
    points,
    k=24,
    iterations=12,
    tangential_weight=0.45,
    normal_weight=0.12,
    volume_restore=0.65,
    sigma_factor=1.5,
):
    """
    Smooth using KNN averaging while resisting shrinkage along local normals.

    Each iteration:
      1. Estimate local normals from the current point cloud.
      2. Compute a Gaussian-weighted neighbor average.
      3. Split the average displacement into tangential and normal parts.
      4. Smooth more strongly tangentially than normally.
      5. Restore some displacement back toward the original point along
         the current normal direction to keep volume from collapsing.
    """
    p0 = points.copy()
    p = points.copy()

    for _ in range(iterations):
        tree = KDTree(p)
        distances, indices = tree.query(p, k=k)
        sigma = np.mean(distances[:, 1]).clip(1e-8) * sigma_factor

        weights = np.exp(-distances[:, 1:] ** 2 / (2 * sigma ** 2))
        weights /= weights.sum(axis=1, keepdims=True)

        neighbor_pts = p[indices[:, 1:]]
        local_avg = (weights[:, :, None] * neighbor_pts).sum(axis=1)
        normals = _estimate_normals_pca(p, indices)

        disp = local_avg - p
        normal_mag = np.sum(disp * normals, axis=1, keepdims=True)
        disp_normal = normal_mag * normals
        disp_tangent = disp - disp_normal

        # Prefer tangential redistribution over inward normal contraction.
        p_trial = p + tangential_weight * disp_tangent + normal_weight * disp_normal

        # Recover some local thickness by pulling each point back toward its
        # original signed offset along the current normal.
        restore_mag = np.sum((p0 - p_trial) * normals, axis=1, keepdims=True)
        p = p_trial + volume_restore * restore_mag * normals

    return p


if __name__ == '__main__':
    points = np.loadtxt('../data/bathtub_uniform.csv', delimiter=',', skiprows=1)
    print(f"Loaded {len(points)} points")

    lap = laplacian_smooth(points, k=12, iterations=1, lam=0.5)
    np.savetxt('../data/bathtub_laplacian.csv', lap, delimiter=',', header='x,y,z', comments='')
    print("Laplacian done.")

    tau = taubin_smooth(points, k=12, iterations=20, lam=0.5, mu=-0.53)
    np.savetxt('../data/bathtub_taubin.csv', tau, delimiter=',', header='x,y,z', comments='')
    print("Taubin done.")

    gau = gaussian_smooth(points, k=12, sigma_factor=1.0, iterations=5)
    np.savetxt('../data/bathtub_gaussian.csv', gau, delimiter=',', header='x,y,z', comments='')
    print("Gaussian done.")
