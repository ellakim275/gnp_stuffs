from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


def normalize_points(points: np.ndarray, mode: str = "unit_sphere") -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an N x 3 point cloud, got shape {points.shape}")
    xyz = points[:, :3].copy()
    if mode == "none":
        return xyz
    xyz -= xyz.mean(axis=0, keepdims=True)
    if mode == "unit_sphere":
        scale = np.linalg.norm(xyz, axis=1).max()
    elif mode == "rms":
        scale = np.sqrt(np.mean(np.sum(xyz * xyz, axis=1)))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return xyz / max(float(scale), EPS)


def build_knn_graph(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(points) <= k:
        raise ValueError(f"Need more points than k; got {len(points)} points and k={k}")
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(points)
    distances, indices = nn.kneighbors(points, return_distance=True)
    return indices[:, 1:], distances[:, 1:]


def gaussian_weight_matrix(
    points: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    sigma_mode: str = "local",
) -> sparse.csr_matrix:
    n_points, k = knn_indices.shape
    if sigma_mode == "local":
        local_sigma = np.median(knn_distances, axis=1)
        local_sigma = np.maximum(local_sigma, EPS)
        rows = np.repeat(np.arange(n_points), k)
        cols = knn_indices.reshape(-1)
        sigma = local_sigma[rows] * local_sigma[cols]
    elif sigma_mode == "global":
        global_sigma = max(float(np.median(knn_distances)), EPS)
        rows = np.repeat(np.arange(n_points), k)
        cols = knn_indices.reshape(-1)
        sigma = global_sigma * global_sigma
    else:
        raise ValueError(f"Unknown sigma mode: {sigma_mode}")

    dist2 = (knn_distances.reshape(-1) ** 2)
    weights = np.exp(-dist2 / np.maximum(sigma, EPS))
    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n_points, n_points)).tocsr()
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def density_correct_weights(W: sparse.csr_matrix, alpha: float = 1.0) -> sparse.csr_matrix:
    if alpha == 0.0:
        return W.copy()
    q = np.asarray(W.sum(axis=1)).ravel()
    q = np.maximum(q, EPS)
    inv_q_alpha = sparse.diags(1.0 / np.power(q, alpha), format="csr")
    W_tilde = inv_q_alpha @ W @ inv_q_alpha
    W_tilde.eliminate_zeros()
    return W_tilde.tocsr()


def build_laplacian(W: sparse.csr_matrix, kind: str = "symmetric") -> sparse.csr_matrix:
    degree = np.asarray(W.sum(axis=1)).ravel()
    degree = np.maximum(degree, EPS)
    if kind == "unnormalized":
        return sparse.diags(degree, format="csr") - W
    if kind == "symmetric":
        inv_sqrt_degree = 1.0 / np.sqrt(degree)
        D_inv_sqrt = sparse.diags(inv_sqrt_degree, format="csr")
        identity = sparse.identity(W.shape[0], format="csr")
        L = identity - D_inv_sqrt @ W @ D_inv_sqrt
        return L.tocsr()
    if kind == "random_walk":
        inv_degree = 1.0 / degree
        D_inv = sparse.diags(inv_degree, format="csr")
        identity = sparse.identity(W.shape[0], format="csr")
        return (identity - D_inv @ W).tocsr()
    raise ValueError(f"Unknown Laplacian kind: {kind}")


def compute_eigendecomposition(
    L: sparse.csr_matrix,
    n_eigenvectors: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = L.shape[0]
    if n_eigenvectors >= n:
        raise ValueError(f"n_eigenvectors must be smaller than N; got {n_eigenvectors} for N={n}")
    rng = np.random.default_rng(random_state)
    v0 = rng.normal(size=n)
    values, vectors = eigsh(L, k=n_eigenvectors, which="SM", v0=v0, tol=1e-5, maxiter=max(5000, n * 20))
    order = np.argsort(values)
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    return values.astype(np.float64), vectors.astype(np.float64)

