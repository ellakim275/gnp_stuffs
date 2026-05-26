"""
close_meshes_poisson.py
-----------------------
Create watertight toy versions of the smooth-shape source meshes using
Open3D Poisson surface reconstruction.

These closed meshes are intended for proof-of-concept Gauss-Bonnet experiments:
they should be treated as cleaned/reconstructed inputs, not as the original
dataset geometry.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for

ROOTS = roots_for(__file__)
DATA_ROOT = ROOTS["DATA_ROOT"]
DEFAULT_OUTPUT_DIR = DATA_ROOT / "closed_poisson_meshes"


SOURCE_MESHES = [
    ("bob", DATA_ROOT / "bob_tri.obj"),
    ("blub", DATA_ROOT / "blub_triangulated.obj"),
    ("spot", DATA_ROOT / "spot_triangulated.obj"),
    ("homer", DATA_ROOT / "homer.obj"),
    ("rabbit", DATA_ROOT / "stanford-bunny.obj"),
]


def mesh_topology(mesh: o3d.geometry.TriangleMesh) -> dict[str, int | float | bool]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in [(a, b), (b, c), (c, a)]:
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    n_edges = len(edge_counts)
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    return {
        "vertices": int(len(vertices)),
        "edges": int(n_edges),
        "triangles": int(len(triangles)),
        "euler_characteristic_mesh": int(len(vertices) - n_edges + len(triangles)),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "watertight": bool(boundary_edges == 0 and nonmanifold_edges == 0),
        "surface_area": float(mesh.get_surface_area()),
    }


def boundary_loops(mesh: o3d.geometry.TriangleMesh) -> list[list[int]]:
    triangles = np.asarray(mesh.triangles)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in [(a, b), (b, c), (c, a)]:
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    adjacency: dict[int, list[int]] = {}
    for (u, v), count in edge_counts.items():
        if count == 1:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

    loops = []
    visited_edges: set[tuple[int, int]] = set()
    for start in sorted(adjacency):
        for neighbor in adjacency[start]:
            edge = tuple(sorted((start, neighbor)))
            if edge in visited_edges:
                continue
            loop = [start]
            prev = start
            current = neighbor
            visited_edges.add(edge)
            while True:
                loop.append(current)
                next_candidates = [
                    candidate
                    for candidate in adjacency.get(current, [])
                    if candidate != prev and tuple(sorted((current, candidate))) not in visited_edges
                ]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                visited_edges.add(tuple(sorted((current, nxt))))
                prev, current = current, nxt
                if current == start:
                    break
            if len(loop) >= 3:
                if loop[-1] == loop[0]:
                    loop = loop[:-1]
                loops.append(loop)
    return loops


def cap_boundary_loops(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Fill boundary loops with a simple centroid fan.

    This is intentionally a toy cleanup for closed-surface experiments. It is
    not a high-quality geometric hole-filling algorithm.
    """
    loops = boundary_loops(mesh)
    if not loops:
        return mesh

    vertices = np.asarray(mesh.vertices).tolist()
    triangles = np.asarray(mesh.triangles).tolist()
    for loop in loops:
        loop_points = np.asarray([vertices[index] for index in loop], dtype=float)
        center_index = len(vertices)
        vertices.append(loop_points.mean(axis=0).tolist())
        for i, current in enumerate(loop):
            nxt = loop[(i + 1) % len(loop)]
            triangles.append([current, nxt, center_index])

    capped = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    capped.remove_duplicated_vertices()
    capped.remove_duplicated_triangles()
    capped.remove_degenerate_triangles()
    capped.remove_unreferenced_vertices()
    capped.compute_vertex_normals()
    return capped


def load_source_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_oriented_points(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    normal_knn: int,
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.orient_normals_consistent_tangent_plane(normal_knn * 4)
    return pcd


def poisson_reconstruct(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    min_depth: int,
    density_quantile: float,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    last_error = None
    for current_depth in range(depth, min_depth - 1, -1):
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=current_depth,
            )
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"  Poisson failed at depth={current_depth}; retrying lower depth", flush=True)
    else:
        raise RuntimeError(f"Poisson reconstruction failed for depths {depth}..{min_depth}") from last_error

    densities_np = np.asarray(densities)
    if density_quantile > 0.0:
        cutoff = np.quantile(densities_np, density_quantile)
        mesh.remove_vertices_by_mask(densities_np < cutoff)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, current_depth


def close_one_mesh(
    class_name: str,
    mesh_path: Path,
    output_dir: Path,
    n_points: int,
    normal_knn: int,
    depth: int,
    min_depth: int,
    density_quantile: float,
    cap_boundaries: bool,
    preserve_watertight_source: bool,
) -> dict[str, str | int | float | bool]:
    source_mesh = load_source_mesh(mesh_path)
    source_stats = mesh_topology(source_mesh)

    if preserve_watertight_source and source_stats["watertight"]:
        closed_mesh = source_mesh
        actual_depth = 0
        reconstruction_method = "preserved_source_watertight"
    else:
        pcd = sample_oriented_points(source_mesh, n_points=n_points, normal_knn=normal_knn)
        closed_mesh, actual_depth = poisson_reconstruct(
            pcd,
            depth=depth,
            min_depth=min_depth,
            density_quantile=density_quantile,
        )
        if cap_boundaries:
            closed_mesh = cap_boundary_loops(closed_mesh)
        reconstruction_method = "poisson"
    closed_stats = mesh_topology(closed_mesh)

    output_path = output_dir / f"{class_name}_closed_poisson.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), closed_mesh)

    row = {
        "class_name": class_name,
        "source_mesh_path": str(mesh_path),
        "closed_mesh_path": str(output_path),
        "poisson_points": n_points,
        "poisson_depth_requested": depth,
        "poisson_depth": actual_depth,
        "density_quantile": density_quantile,
        "capped_boundaries": int(cap_boundaries),
        "reconstruction_method": reconstruction_method,
    }
    for key, value in source_stats.items():
        row[f"source_{key}"] = value
    for key, value in closed_stats.items():
        row[f"closed_{key}"] = value
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close source meshes with Open3D Poisson reconstruction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of classes to reconstruct, e.g. --classes rabbit spot.",
    )
    parser.add_argument("--n-points", type=int, default=30000, help="Poisson samples used for reconstruction.")
    parser.add_argument("--normal-knn", type=int, default=30)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.0,
        help="Trim vertices below this density quantile; use 0 to keep the raw closed Poisson mesh.",
    )
    parser.add_argument(
        "--no-cap-boundaries",
        action="store_true",
        help="Do not fill boundary loops left by the extracted Poisson mesh.",
    )
    parser.add_argument(
        "--reconstruct-watertight-sources",
        action="store_true",
        help="Run Poisson even when the original source mesh is already watertight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    o3d.utility.random.seed(args.seed)
    source_meshes = SOURCE_MESHES
    if args.classes:
        requested = set(args.classes)
        source_meshes = [(class_name, path) for class_name, path in SOURCE_MESHES if class_name in requested]
        missing = requested - {class_name for class_name, _ in source_meshes}
        if missing:
            raise ValueError(f"Unknown classes: {sorted(missing)}")

    rows = []
    for class_name, mesh_path in source_meshes:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")
        print(f"[close] {class_name:8s} <- {mesh_path.name}", flush=True)
        row = close_one_mesh(
            class_name=class_name,
            mesh_path=mesh_path,
            output_dir=args.output_dir,
            n_points=args.n_points,
            normal_knn=args.normal_knn,
            depth=args.depth,
            min_depth=args.min_depth,
            density_quantile=args.density_quantile,
            cap_boundaries=not args.no_cap_boundaries,
            preserve_watertight_source=not args.reconstruct_watertight_sources,
        )
        rows.append(row)
        print(
            f"  wrote {Path(row['closed_mesh_path']).name} "
            f"watertight={row['closed_watertight']} "
            f"chi={row['closed_euler_characteristic_mesh']} "
            f"boundary_edges={row['closed_boundary_edges']}",
            flush=True,
        )

    summary_path = args.output_dir / "closed_mesh_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
