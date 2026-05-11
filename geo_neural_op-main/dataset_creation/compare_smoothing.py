"""
compare_smoothing.py
--------------------
Sample a fixed set of meshes to point clouds, apply two smoothing methods,
and export the results plus a side-by-side PNG comparison for each mesh.

Outputs are written to:
    output/smoothing_comparison/<mesh_name>/

Per mesh, the script writes:
    laplacian.csv
    gaussian.csv
    comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from point_sampling.knn import gaussian_smooth, laplacian_smooth
from point_sampling.sample_pointcloud import sample_mesh_to_points


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
MODELNET_ROOT = WORKSPACE_ROOT / "ModelNet10"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "output" / "smoothing_comparison"


def build_mesh_list() -> list[tuple[str, Path]]:
    return [
        ("desk_0023", MODELNET_ROOT / "desk" / "train" / "desk_0023.off"),
        ("toilet_0004", MODELNET_ROOT / "toilet" / "train" / "toilet_0004.off"),
        ("bathtub_0007", MODELNET_ROOT / "bathtub" / "train" / "bathtub_0007.off"),
        ("spot_triangulated", DATA_ROOT / "spot_triangulated.obj"),
        ("bob_tri", DATA_ROOT / "bob_tri.obj"),
        ("blub_triangulated", DATA_ROOT / "blub_triangulated.obj"),
    ]


def save_point_cloud_csv(points: np.ndarray, out_path: Path) -> None:
    np.savetxt(out_path, points, delimiter=",", header="x,y,z", comments="")


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_comparison(
    laplacian_pts: np.ndarray,
    gaussian_pts: np.ndarray,
    out_path: Path,
    title: str,
    point_size: float,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    method_data = [
        ("Laplacian", laplacian_pts),
        ("Gaussian", gaussian_pts),
    ]

    combined = np.vstack([laplacian_pts, gaussian_pts])
    zmin = combined[:, 2].min()
    zmax = combined[:, 2].max()

    for idx, (name, pts) in enumerate(method_data, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=pts[:, 2],
            cmap="viridis",
            s=point_size,
            linewidths=0,
            vmin=zmin,
            vmax=zmax,
        )
        set_equal_axes(ax, combined)
        ax.view_init(elev=20, azim=35)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def process_mesh(
    mesh_name: str,
    mesh_path: Path,
    n_points: int,
    knn_k: int,
    laplacian_iterations: int,
    laplacian_lambda: float,
    gaussian_iterations: int,
    gaussian_sigma_factor: float,
    point_size: float,
) -> None:
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh: {mesh_path}")

    print(f"[mesh] {mesh_name} <- {mesh_path}")
    sampled = sample_mesh_to_points(mesh_path, n_points)

    lap = laplacian_smooth(
        sampled,
        k=knn_k,
        iterations=laplacian_iterations,
        lam=laplacian_lambda,
    )
    gau = gaussian_smooth(
        sampled,
        k=knn_k,
        sigma_factor=gaussian_sigma_factor,
        iterations=gaussian_iterations,
    )

    mesh_out_dir = OUTPUT_ROOT / mesh_name
    mesh_out_dir.mkdir(parents=True, exist_ok=True)

    save_point_cloud_csv(lap, mesh_out_dir / "laplacian.csv")
    save_point_cloud_csv(gau, mesh_out_dir / "gaussian.csv")
    # save_point_cloud_csv(tau, mesh_out_dir / "taubin.csv")
    render_comparison(
        laplacian_pts=lap,
        gaussian_pts=gau,
        out_path=mesh_out_dir / "comparison.png",
        title=f"{mesh_name} smoothing comparison",
        point_size=point_size,
    )

    print(f"  wrote {mesh_out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Laplacian and Gaussian smoothing on a fixed mesh set."
    )
    parser.add_argument("--n-points", type=int, default=30000, help="Points sampled per mesh.")
    parser.add_argument("--knn-k", type=int, default=24, help="Neighbor count for smoothing.")

    parser.add_argument("--laplacian-iterations", type=int, default=8)
    parser.add_argument("--laplacian-lambda", type=float, default=0.75)

    parser.add_argument("--gaussian-iterations", type=int, default=6)
    parser.add_argument("--gaussian-sigma-factor", type=float, default=4)

    # parser.add_argument("--taubin-iterations", type=int, default=100)
    # parser.add_argument("--taubin-lambda", type=float, default=0.65)
    # parser.add_argument("--taubin-mu", type=float, default=-0.64)
    # parser.add_argument("--taubin-alpha", type=float, default=0.75)

    parser.add_argument("--point-size", type=float, default=0.2, help="Scatter point size for PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mesh_list = build_mesh_list()
    print(f"Writing outputs to {OUTPUT_ROOT}")

    for mesh_name, mesh_path in mesh_list:
        process_mesh(
            mesh_name=mesh_name,
            mesh_path=mesh_path,
            n_points=args.n_points,
            knn_k=args.knn_k,
            laplacian_iterations=args.laplacian_iterations,
            laplacian_lambda=args.laplacian_lambda,
            gaussian_iterations=args.gaussian_iterations,
            gaussian_sigma_factor=args.gaussian_sigma_factor,
            point_size=args.point_size,
        )

    print("Done.")


if __name__ == "__main__":
    main()
