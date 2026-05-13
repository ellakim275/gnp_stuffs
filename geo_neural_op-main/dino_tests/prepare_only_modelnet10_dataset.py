#!/usr/bin/env python3
"""
Render the Gaussian ModelNet10 point clouds into an ImageFolder layout for DINO.

The source dataset uses train/test. The DINO eval convention uses train/val, so
this script maps test -> val. By default it renders fresh snapshots from CSV
point clouds on a flat background with no 3D axes or pane box, because those
visual artifacts can become shortcuts for attention maps.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

MPL_CACHE = Path(__file__).resolve().parent / "outputs" / ".matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
XDG_CACHE = Path(__file__).resolve().parent / "outputs" / ".cache"
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT
    / "interpretable_feature_tests"
    / "output"
    / "gaussian"
    / "datasets"
    / "only_modelnet10_gaussian_dataset"
)
DEFAULT_SOURCE = DEFAULT_DATASET / "previews"
DEFAULT_MANIFEST = DEFAULT_DATASET / "manifest.csv"
DEFAULT_DEST = Path(__file__).resolve().parent / "data" / "only_modelnet10_gaussian_dataset"


def copy_split(source_root: Path, dest_root: Path, source_split: str, dest_split: str) -> int:
    source_split_dir = source_root / source_split
    dest_split_dir = dest_root / dest_split
    if not source_split_dir.exists():
        raise FileNotFoundError(f"Missing source split: {source_split_dir}")

    if dest_split_dir.exists():
        shutil.rmtree(dest_split_dir)
    shutil.copytree(source_split_dir, dest_split_dir)
    return len(list(dest_split_dir.rglob("*.png")))


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1e-6)
    radius *= 1.08

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_flat_point_cloud(
    csv_path: Path,
    out_path: Path,
    image_size: float,
    dpi: int,
    point_size: float,
    elev: float,
    azim: float,
    background: str,
) -> None:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(image_size, image_size), dpi=dpi, frameon=False)
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor(background)
    ax.set_proj_type("ortho")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, 2],
        cmap="viridis",
        s=point_size,
        linewidths=0,
        depthshade=False,
    )
    set_equal_axes(ax, points)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, facecolor=background, edgecolor=background, pad_inches=0)
    plt.close(fig)


def render_from_manifest(
    manifest_path: Path,
    dest_root: Path,
    image_size: float,
    dpi: int,
    point_size: float,
    elev: float,
    azim: float,
    background: str,
) -> tuple[int, int]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    for split in ["train", "val"]:
        split_dir = dest_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    counts = {"train": 0, "val": 0}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_split = row["split"]
            dest_split = "val" if source_split == "test" else source_split
            if dest_split not in counts:
                continue

            csv_path = resolve_manifest_path(manifest_path.parent, Path(row["csv_path"]))
            out_path = (
                dest_root
                / dest_split
                / row["class_name"]
                / f"{csv_path.stem}.png"
            )
            render_flat_point_cloud(
                csv_path=csv_path,
                out_path=out_path,
                image_size=image_size,
                dpi=dpi,
                point_size=point_size,
                elev=elev,
                azim=azim,
                background=background,
            )
            counts[dest_split] += 1

    return counts["train"], counts["val"]


def resolve_manifest_path(dataset_root: Path, path: Path) -> Path:
    if path.exists():
        return path

    parts = path.parts
    if "csv" in parts:
        csv_idx = parts.index("csv")
        candidate = dataset_root.joinpath(*parts[csv_idx:])
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Manifest path does not exist and could not be resolved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--copy-existing-previews",
        action="store_true",
        help="Copy existing preview PNGs instead of rendering flat snapshots from CSV.",
    )
    parser.add_argument("--image-size", type=float, default=4.0, help="Matplotlib figure size in inches.")
    parser.add_argument("--dpi", type=int, default=96, help="Output DPI; 4 inches x 96 DPI gives 384 px.")
    parser.add_argument("--point-size", type=float, default=0.18)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--background", default="white")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    if args.copy_existing_previews:
        train_count = copy_split(args.source, args.dest, "train", "train")
        val_count = copy_split(args.source, args.dest, "test", "val")
        action = "Copied"
    else:
        train_count, val_count = render_from_manifest(
            manifest_path=args.manifest,
            dest_root=args.dest,
            image_size=args.image_size,
            dpi=args.dpi,
            point_size=args.point_size,
            elev=args.elev,
            azim=args.azim,
            background=args.background,
        )
        action = "Rendered"

    print(f"{action} {train_count} train images and {val_count} val images.")
    print(f"DINO dataset ready at: {args.dest}")


if __name__ == "__main__":
    main()
