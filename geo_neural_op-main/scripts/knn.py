import numpy as np
from scipy.spatial import KDTree
import scipy.sparse as sp


def laplacian_smooth(points, k=12, iterations=1, lam=0.5):
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
    p = points.copy()
    for _ in range(iterations):
        p = p - lam * (L_norm @ p)
    return p


if __name__ == '__main__':
    points = np.loadtxt('../data/bathtub_uniform.csv', delimiter=',', skiprows=1)
    smoothed = laplacian_smooth(points, k=12, iterations=100, lam=0.5)
    np.savetxt('../data/bathtub_uniform_blobby.csv', smoothed, delimiter=',', header='x,y,z', comments='')
    print(f"Points: {len(points)}, done.")
