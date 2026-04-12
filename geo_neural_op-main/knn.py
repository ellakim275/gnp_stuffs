import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import scipy.sparse as sp

points = np.loadtxt('bathtub_uniform.csv', delimiter=',', skiprows=1)

k = 12
tree = KDTree(points)
distances, indices = tree.query(points, k=k) 

N = len(points)

sigma = np.mean(distances[:, 1])
rows = np.repeat(np.arange(N), k - 1)
cols = indices[:, 1:].flatten()
weights = np.exp(-distances[:, 1:].flatten()**2 / (2 * sigma**2)) #weighted by gaussian kernel 

W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N)) #adjacency matrix 
W = (W + W.T) / 2 #symmetric adjacency matrix

degree = np.asarray(W.sum(axis=1)).flatten()
D = sp.diags(degree) #diagonal degree matrix
L = D - W #unnormalized laplacian matrix 

D_inv = sp.diags(1.0 / degree) #MOOREPENROSE?
L_norm = D_inv @ L  #normalized random walk laplacian matrix

def knn_laplacian_smooth(points, L_norm, iterations=100, lam=0.5):
    p = points.copy()
    for i in range(iterations):
        p = p - lam * (L_norm @ p)
    return p

smoothed = knn_laplacian_smooth(points, L_norm, iterations=100, lam=0.5)

np.savetxt('bathtub_uniform_blobby.csv',   smoothed, delimiter=',', header='x,y,z', comments='')

print(f"Points: {N}, k={k}, sigma={sigma:.4f}")