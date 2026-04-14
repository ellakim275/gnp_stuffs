import open3d as o3d
import numpy as np


mesh = o3d.io.read_triangle_mesh('/Users/ellakim/Downloads/GNP_tests/ModelNet10/bathtub/test/bathtub_0107.off')
#mesh = o3d.io.read_triangle_mesh('/Users/ellakim/Downloads/GNP_tests/oiiaioooooiai_cat.glb')

'''
mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh('desk_smoothed_laplace_mesh.ply', mesh)  
'''

pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
#pcd = mesh.sample_points_uniformly(number_of_points=10000)


points = np.asarray(pcd.points)
np.savetxt('bathtub_uniform.csv', points, delimiter=',', header='x,y,z', comments='')
