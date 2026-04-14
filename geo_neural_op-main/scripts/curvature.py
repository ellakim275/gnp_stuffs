import sys
import torch
import numpy as np
from pathlib import Path
import pyvista as pv
import pandas as pd
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gnp.estimator import GeometryEstimator

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

pv.set_jupyter_backend("static")

# Load CSV 
df = pd.read_csv('../data/cat_blobby.csv')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(100)

normals = np.asarray(pcd.normals)
xyz = np.asarray(pcd.points)
xyz = xyz - xyz.mean(axis=0)
xyz = xyz / np.max(np.abs(xyz))

data = {
    'xyz': torch.tensor(xyz, dtype=torch.float32).to(device),
    'normals': torch.tensor(normals, dtype=torch.float32).to(device)
}

estimator_clean = GeometryEstimator(data['xyz'], 
                              orientation=data['normals'], 
                              model="clean_30k",
                              device=device)
output = estimator_clean.estimate_quantities(['mean_curvature', 'gaussian_curvature'])
mean_clean = output['mean_curvature']
gauss_clean = output['gaussian_curvature']

pvdata = pv.PolyData(data['xyz'].cpu().numpy())
pvdata['mean_curvature'] = output['mean_curvature'].cpu().numpy()
pvdata['gaussian_curvature'] = output['gaussian_curvature'].cpu().numpy()

scalars = ['mean_curvature', 'gaussian_curvature']
lims = [5, 20]

plotter = pv.Plotter(shape=(1, 2))
for j, (scalar, lim) in enumerate(zip(scalars, lims)):
    plotter.subplot(0, j)
    if j == 0:
        plotter.add_text('mean curvature', font_size=18)
    if j == 1:
        plotter.add_text('gaussian curvature', font_size=18)
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