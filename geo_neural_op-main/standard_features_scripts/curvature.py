import sys
import torch
import numpy as np
from pathlib import Path
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gnp.estimator import GeometryEstimator


def canonicalize_points(points):
    """
    Center, PCA-align, fix axis signs deterministically, and normalize to [-1, 1].

    This makes axis-aligned downstream features more comparable across samples
    of the same class.
    """
    xyz = np.asarray(points, dtype=np.float64)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)

    cov = np.cov(xyz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    aligned = xyz @ basis

    # Resolve PCA sign ambiguity deterministically: flip each axis so the
    # positive extent is at least as large as the negative extent.
    for axis in range(3):
        if abs(aligned[:, axis].min()) > abs(aligned[:, axis].max()):
            aligned[:, axis] *= -1.0

    aligned = aligned / max(np.max(np.abs(aligned)), 1e-8)
    return aligned.astype(np.float32)


def estimate_curvatures(points, device):
    xyz = canonicalize_points(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    normals = np.asarray(pcd.normals)
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    n_t   = torch.tensor(normals, dtype=torch.float32, device=device)
    estimator = GeometryEstimator(xyz_t, orientation=n_t, model='clean_30k', device=device)
    output = estimator.estimate_quantities(['mean_curvature', 'gaussian_curvature'])
    return xyz_t, output


if __name__ == '__main__':
    import pandas as pd
    import pyvista as pv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pv.set_jupyter_backend("static")

    df = pd.read_csv('../data/cat_blobby.csv')
    points = df[['x', 'y', 'z']].values

    xyz_t, output = estimate_curvatures(points, device)

    pvdata = pv.PolyData(xyz_t.cpu().numpy())
    pvdata['mean_curvature']     = output['mean_curvature'].cpu().numpy()
    pvdata['gaussian_curvature'] = output['gaussian_curvature'].cpu().numpy()

    scalars = ['mean_curvature', 'gaussian_curvature']
    lims    = [5, 20]

    plotter = pv.Plotter(shape=(1, 2))
    for j, (scalar, lim) in enumerate(zip(scalars, lims)):
        plotter.subplot(0, j)
        plotter.add_text('mean curvature' if j == 0 else 'gaussian curvature', font_size=18)
        plotter.add_points(pvdata,
                           style='points',
                           point_size=5,
                           render_points_as_spheres=True,
                           cmap='coolwarm',
                           scalars=scalar,
                           clim=[-lim, lim],
                           copy_mesh=True)
        plotter.reset_camera()
        plotter.remove_scalar_bar()
    plotter.show()
