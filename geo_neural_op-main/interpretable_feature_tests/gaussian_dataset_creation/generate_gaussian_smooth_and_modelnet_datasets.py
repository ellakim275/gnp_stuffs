"""
generate_gaussian_svm_datasets.py
---------------------------------
Build two Gaussian-smoothed deformation datasets for SVM experiments:

1. modelnet10_smooth_shapes
   Multiple source meshes from bathtub, desk, sofa, and toilet.
   Train and test splits use different source meshes.

2. smooth_mesh_distortions
   Distorted variants of bob, blub, spot, homer, and rabbit.

Each generated shape is sampled from a mesh, Gaussian-smoothed, gently
distorted, and saved as both CSV point cloud and PNG preview.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for

ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
TEST_ROOT = ROOTS["TEST_ROOT"]
WORKSPACE_ROOT = ROOTS["WORKSPACE_ROOT"]
MODELNET_ROOT = ROOTS["MODELNET_ROOT"]
DATA_ROOT = ROOTS["DATA_ROOT"]
MODELNET_COPY_ROOT = DATA_ROOT / "ModelNet10 copy"
DEFAULT_OUTPUT_ROOT = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets"

sys.path.insert(0, str(SCRIPT_DIR))

from knn import gaussian_smooth
from sample_pointcloud import sample_mesh_to_points


@dataclass(frozen=True)
class SourceMesh:
    dataset: str
    split: str
    class_name: str
    source_id: str
    mesh_path: Path
    variants: int
    distort: bool = True
    distortion_strength: float = 0.18


def save_point_cloud_csv(points: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, points, delimiter=",", header="x,y,z", comments="")


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_point_cloud(points: np.ndarray, out_path: Path, title: str, point_size: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5))
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
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = max(np.max(np.linalg.norm(centered, axis=1)), 1e-8)
    return centered / scale, centroid, scale


def apply_axis_aligned_transform(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scales = rng.uniform(0.9, 1.12, size=3)
    shear = np.eye(3)
    shear[0, 1] = rng.uniform(-0.04, 0.04)
    shear[1, 2] = rng.uniform(-0.04, 0.04)
    return (points * scales[None, :]) @ shear


def random_fourier_vector_field(
    points: np.ndarray,
    rng: np.random.Generator,
    n_modes: int = 8,
) -> np.ndarray:
    field = np.zeros_like(points)
    for _ in range(n_modes):
        wave = rng.integers(-3, 4, size=3).astype(float)
        if np.allclose(wave, 0.0):
            wave[0] = 1.0
        phase = rng.uniform(0.0, 2.0 * np.pi)
        magnitude = np.linalg.norm(wave)
        coeff = np.exp(-magnitude) * rng.normal(size=3)
        field += np.sin(points @ wave + phase)[:, None] * coeff[None, :]
    return field


def make_variant(points: np.ndarray, seed: int, distortion_strength: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normed, centroid, scale = normalize_points(points)
    deformed = apply_axis_aligned_transform(normed, rng)
    for _ in range(6):
        deformed = deformed + distortion_strength * random_fourier_vector_field(deformed, rng)
    deformed = apply_axis_aligned_transform(deformed, rng)
    return deformed * scale + centroid


def select_modelnet_meshes(
    class_name: str,
    split: str,
    count: int,
    root: Path,
) -> list[Path]:
    files = sorted((root / class_name / split).glob("*.off"))
    if len(files) < count:
        raise ValueError(f"Not enough {split} meshes for {class_name} in {root}: {len(files)} < {count}")
    return files[:count]


def build_modelnet_sources(
    train_meshes_per_class: int,
    test_meshes_per_class: int,
    variants_per_mesh: int,
    split_filter: str,
) -> list[SourceMesh]:
    modelnet_root = MODELNET_COPY_ROOT if MODELNET_COPY_ROOT.exists() else MODELNET_ROOT
    sources: list[SourceMesh] = []
    for class_name in ["bathtub", "desk", "sofa", "toilet"]:
        for split, count, variants, distort in [
            ("train", train_meshes_per_class, variants_per_mesh, True),
            ("test", test_meshes_per_class, 1, False),
        ]:
            if split_filter != "all" and split != split_filter:
                continue
            for mesh_path in select_modelnet_meshes(class_name, split, count, root=modelnet_root):
                sources.append(
                    SourceMesh(
                        dataset="modelnet10_smooth_shapes",
                        split=split,
                        class_name=class_name,
                        source_id=mesh_path.stem,
                        mesh_path=mesh_path,
                        variants=variants,
                        distort=distort,
                    )
                )
    return sources


def build_smooth_mesh_sources(
    variants_per_shape: int,
    train_variants_per_shape: int,
    distortion_strength: float,
) -> list[SourceMesh]:
    meshes = [
        ("bob", DATA_ROOT / "bob_tri.obj"),
        ("blub", DATA_ROOT / "blub_triangulated.obj"),
        ("spot", DATA_ROOT / "spot_triangulated.obj"),
        ("homer", DATA_ROOT / "homer.obj"),
        ("rabbit", DATA_ROOT / "stanford-bunny.obj"),
    ]
    sources: list[SourceMesh] = []
    for class_name, mesh_path in meshes:
        for split, variants in [
            ("train", train_variants_per_shape),
            ("test", variants_per_shape - train_variants_per_shape),
        ]:
            sources.append(
                SourceMesh(
                    dataset="only_smoothmeshes_gaussian_dataset",
                    split=split,
                    class_name=class_name,
                    source_id=mesh_path.stem,
                    mesh_path=mesh_path,
                    variants=variants,
                    distortion_strength=distortion_strength,
                )
            )
    return sources


def process_sources(
    sources: list[SourceMesh],
    output_root: Path,
    n_points: int,
    knn_k: int,
    gaussian_iterations: int,
    gaussian_sigma_factor: float,
    point_size: float,
    preserve_manifest_splits: set[str] | None = None,
) -> None:
    manifest_rows: dict[str, list[dict[str, str | int]]] = {}

    for source_idx, source in enumerate(sources):
        if not source.mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh: {source.mesh_path}")

        dataset_dir = output_root / source.dataset
        csv_dir = dataset_dir / "csv" / source.split / source.class_name
        png_dir = dataset_dir / "previews" / source.split / source.class_name
        manifest_rows.setdefault(source.dataset, [])

        print(f"[{source.dataset}] {source.split:5s} {source.class_name:8s} {source.mesh_path.name}", flush=True)
        sampled = sample_mesh_to_points(source.mesh_path, n_points)
        smoothed = gaussian_smooth(
            sampled,
            k=knn_k,
            sigma_factor=gaussian_sigma_factor,
            iterations=gaussian_iterations,
        )

        for local_variant_idx in range(1, source.variants + 1):
            global_seed = 100_000 * (source_idx + 1) + local_variant_idx
            variant = make_variant(
                smoothed,
                seed=global_seed,
                distortion_strength=source.distortion_strength,
            ) if source.distort else smoothed
            item_id = f"{source.class_name}_{source.split}_{source.source_id}_variant{local_variant_idx:02d}"
            csv_path = csv_dir / f"{item_id}.csv"
            png_path = png_dir / f"{item_id}.png"
            save_point_cloud_csv(variant, csv_path)
            render_point_cloud(variant, png_path, item_id, point_size)
            manifest_rows[source.dataset].append(
                {
                    "dataset": source.dataset,
                    "split": source.split,
                    "class_name": source.class_name,
                    "source_id": source.source_id,
                    "mesh_path": str(source.mesh_path),
                    "variant_idx": local_variant_idx,
                    "csv_path": str(csv_path),
                    "png_path": str(png_path),
                    "n_points": n_points,
                    "distorted": int(source.distort),
                    "distortion_strength": source.distortion_strength if source.distort else 0.0,
                }
            )

    for dataset, rows in manifest_rows.items():
        manifest_path = output_root / dataset / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if preserve_manifest_splits and manifest_path.exists():
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                preserved_rows = [
                    row for row in csv.DictReader(handle) if row.get("split") in preserve_manifest_splits
                ]
            rows = preserved_rows + rows
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved manifest: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gaussian-smoothed SVM datasets.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--knn-k", type=int, default=24)
    parser.add_argument("--gaussian-iterations", type=int, default=6)
    parser.add_argument("--gaussian-sigma-factor", type=float, default=4.0)
    parser.add_argument("--point-size", type=float, default=0.25)
    parser.add_argument("--modelnet-train-meshes-per-class", type=int, default=3)
    parser.add_argument("--modelnet-test-meshes-per-class", type=int, default=3)
    parser.add_argument("--modelnet-variants-per-mesh", type=int, default=3)
    parser.add_argument(
        "--modelnet-split",
        choices=["all", "train", "test"],
        default="all",
        help="Limit ModelNet10 generation to one split. Existing rows from the other split are preserved.",
    )
    parser.add_argument("--smooth-variants-per-shape", type=int, default=10)
    parser.add_argument("--smooth-train-variants-per-shape", type=int, default=8)
    parser.add_argument("--smooth-distortion-strength", type=float, default=0.45)
    parser.add_argument(
        "--dataset",
        choices=["all", "modelnet10_smooth_shapes", "smooth_mesh_distortions", "only_smoothmeshes_gaussian_dataset"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = []
    if args.dataset in {"all", "modelnet10_smooth_shapes"}:
        sources.extend(
            build_modelnet_sources(
                train_meshes_per_class=args.modelnet_train_meshes_per_class,
                test_meshes_per_class=args.modelnet_test_meshes_per_class,
                variants_per_mesh=args.modelnet_variants_per_mesh,
                split_filter=args.modelnet_split,
            )
        )
    if args.dataset in {"all", "smooth_mesh_distortions", "only_smoothmeshes_gaussian_dataset"}:
        sources.extend(
            build_smooth_mesh_sources(
                variants_per_shape=args.smooth_variants_per_shape,
                train_variants_per_shape=args.smooth_train_variants_per_shape,
                distortion_strength=args.smooth_distortion_strength,
            )
        )
    process_sources(
        sources=sources,
        output_root=args.output_root,
        n_points=args.n_points,
        knn_k=args.knn_k,
        gaussian_iterations=args.gaussian_iterations,
        gaussian_sigma_factor=args.gaussian_sigma_factor,
        point_size=args.point_size,
        preserve_manifest_splits=({"test"} if args.dataset == "modelnet10_smooth_shapes" and args.modelnet_split == "train" else None),
    )


if __name__ == "__main__":
    main()
