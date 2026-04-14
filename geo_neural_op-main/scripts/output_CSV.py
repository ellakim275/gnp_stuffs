import open3d as o3d
import numpy as np


def sample_mesh_to_points(mesh_path, n_points):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    return np.asarray(pcd.points)


if __name__ == '__main__':
    points = sample_mesh_to_points(
        '/Users/ellakim/Downloads/GNP_tests/ModelNet10/bathtub/test/bathtub_0107.off',
        n_points=10000,
    )
    np.savetxt('../data/bathtub_uniform.csv', points, delimiter=',', header='x,y,z', comments='')
