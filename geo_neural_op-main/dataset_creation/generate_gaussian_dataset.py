"""
generate_gaussian_dataset.py
----------------------------
Create a small class-style dataset from six source meshes by:
  1. sampling each mesh to a point cloud
  2. Gaussian smoothing the sampled cloud
  3. generating deformed variants per mesh using a smooth random
     vector field plus mild random linear transforms
  4. exporting one CSV and one PNG per variant

By default this yields 180 point clouds and 180 PNGs (360 files total)
for 6 classes x 30 variants, plus a 24/6 train/test manifest per class.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from point_sampling.knn import gaussian_smooth
from point_sampling.sample_pointcloud import sample_mesh_to_points


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
MODELNET_ROOT = WORKSPACE_ROOT / "ModelNet10"
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "output" / "gaussian_augmented_dataset"
SPLIT_MANIFEST = OUTPUT_ROOT / "train_test_split.csv"


def build_mesh_list() -> list[tuple[str, Path]]:
    return [
        ("desk", MODELNET_ROOT / "desk" / "train" / "desk_0023.off"),
        ("toilet", MODELNET_ROOT / "toilet" / "train" / "toilet_0004.off"),
        ("bathtub", MODELNET_ROOT / "bathtub" / "train" / "bathtub_0007.off"),
        ("spot", DATA_ROOT / "spot_triangulated.obj"),
        ("bob", DATA_ROOT / "bob_tri.obj"),
        ("blub", DATA_ROOT / "blub_triangulated.obj"),
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


def render_point_cloud(points: np.ndarray, out_path: Path, title: str, point_size: float) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, 2],
        cmap="viridis",
        s=point_size,
        linewidths=0,
    )
    set_equal_axes(ax, points)
    ax.view_init(elev=20, azim=35)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = max(np.max(np.linalg.norm(centered, axis=1)), 1e-8)
    return centered / scale, centroid, scale


def apply_axis_aligned_transform(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Deform without random rotation so class members stay in a shared frame.
    """
    scales = rng.uniform(0.88, 1.14, size=3)
    return points * scales[None, :]


def random_fourier_vector_field(
    points: np.ndarray,
    rng: np.random.Generator,
    n_modes: int = 10,
) -> np.ndarray:
    field = np.zeros_like(points)
    for _ in range(n_modes):
        wave = rng.integers(-3, 4, size=3).astype(float)
        if np.allclose(wave, 0.0):
            wave[0] = 1.0
        phase = rng.uniform(0.0, 2.0 * np.pi)
        magnitude = np.linalg.norm(wave)
        coeff = np.exp(-magnitude) * rng.normal(size=3)
        signal = np.sin(points @ wave + phase)[:, None]
        field += signal * coeff[None, :]
    return field


def deform_by_flow(
    points: np.ndarray,
    rng: np.random.Generator,
    n_steps: int = 8,
    step_size: float = 0.24,
) -> np.ndarray:
    deformed = points.copy()
    for _ in range(n_steps):
        field = random_fourier_vector_field(deformed, rng)
        deformed = deformed + step_size * field
    return deformed


def make_variant(points: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normed, centroid, scale = normalize_points(points)
    transformed = apply_axis_aligned_transform(normed, rng)
    flowed = deform_by_flow(transformed, rng)
    flowed = apply_axis_aligned_transform(flowed, rng)
    return flowed * scale + centroid


def process_mesh(
    mesh_name: str,
    mesh_path: Path,
    n_points: int,
    knn_k: int,
    gaussian_iterations: int,
    gaussian_sigma_factor: float,
    point_size: float,
    seed_base: int,
    variants_per_class: int,
) -> None:
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh: {mesh_path}")

    print(f"[mesh] {mesh_name} <- {mesh_path}", flush=True)
    sampled = sample_mesh_to_points(mesh_path, n_points)
    smoothed = gaussian_smooth(
        sampled,
        k=knn_k,
        sigma_factor=gaussian_sigma_factor,
        iterations=gaussian_iterations,
    )

    for variant_idx in range(1, variants_per_class + 1):
        variant_name = f"{mesh_name}_variant{variant_idx:02d}"
        variant_points = make_variant(smoothed, seed=seed_base + variant_idx)
        save_point_cloud_csv(variant_points, OUTPUT_ROOT / f"{variant_name}.csv")
        render_point_cloud(
            variant_points,
            OUTPUT_ROOT / f"{variant_name}.png",
            title=variant_name,
            point_size=point_size,
        )
        print(f"  wrote {variant_name}", flush=True)


def write_split_manifest(
    mesh_list: list[tuple[str, Path]],
    variants_per_class: int,
    train_per_class: int,
    test_per_class: int,
    out_path: Path,
) -> None:
    needed = train_per_class + test_per_class
    if needed > variants_per_class:
        raise ValueError(
            f"Split requests {needed} variants per class, but only "
            f"{variants_per_class} are generated."
        )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "variant_idx", "csv_path", "png_path"],
        )
        writer.writeheader()
        for class_name, _ in mesh_list:
            for variant_idx in range(1, needed + 1):
                variant_name = f"{class_name}_variant{variant_idx:02d}"
                writer.writerow(
                    {
                        "split": "train" if variant_idx <= train_per_class else "test",
                        "class_name": class_name,
                        "variant_idx": variant_idx,
                        "csv_path": str(OUTPUT_ROOT / f"{variant_name}.csv"),
                        "png_path": str(OUTPUT_ROOT / f"{variant_name}.png"),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gaussian-smoothed point-cloud variants for six meshes."
    )
    parser.add_argument("--n-points", type=int, default=10000, help="Points sampled per mesh.")
    parser.add_argument("--knn-k", type=int, default=24, help="Neighbor count for Gaussian smoothing.")
    parser.add_argument("--gaussian-iterations", type=int, default=6)
    parser.add_argument("--gaussian-sigma-factor", type=float, default=4.0)
    parser.add_argument("--point-size", type=float, default=0.2)
    parser.add_argument("--variants-per-class", type=int, default=30)
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--test-per-class", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mesh_list = build_mesh_list()
    print(f"Writing outputs to {OUTPUT_ROOT}", flush=True)

    for mesh_idx, (mesh_name, mesh_path) in enumerate(mesh_list):
        process_mesh(
            mesh_name=mesh_name,
            mesh_path=mesh_path,
            n_points=args.n_points,
            knn_k=args.knn_k,
            gaussian_iterations=args.gaussian_iterations,
            gaussian_sigma_factor=args.gaussian_sigma_factor,
            point_size=args.point_size,
            seed_base=1000 * (mesh_idx + 1),
            variants_per_class=args.variants_per_class,
        )

    write_split_manifest(
        mesh_list=mesh_list,
        variants_per_class=args.variants_per_class,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        out_path=SPLIT_MANIFEST,
    )
    print(f"Saved split manifest: {SPLIT_MANIFEST}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
