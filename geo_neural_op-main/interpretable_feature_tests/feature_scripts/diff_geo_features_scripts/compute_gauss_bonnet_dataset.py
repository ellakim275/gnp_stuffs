"""
compute_gauss_bonnet_dataset.py
--------------------------------
Compute Gauss-Bonnet quantities for every shape in the Gaussian dataset.

Expected input files:
    output/gaussian/datasets/first_mixed_gaussian_dataset/gaussian_augmented_dataset/<class_name>_variantXX.csv

Outputs:
    output/gaussian/datasets/first_mixed_gaussian_dataset/gaussian_augmented_dataset/gauss_bonnet_results.csv
    output/gaussian/datasets/first_mixed_gaussian_dataset/gaussian_augmented_dataset/gauss_bonnet_class_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from path_config import roots_for

ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
TEST_ROOT = ROOTS["TEST_ROOT"]
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "first_mixed_gaussian_dataset" / "gaussian_augmented_dataset"
DEFAULT_RESULTS_PATH = DEFAULT_DATASET_DIR / "gauss_bonnet_results.csv"
DEFAULT_SUMMARY_PATH = DEFAULT_DATASET_DIR / "gauss_bonnet_class_summary.csv"
VARIANT_RE = re.compile(r"^(?P<class_name>[a-zA-Z0-9_]+)_variant(?P<idx>\d+)\.csv$")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOTS["GAUSSIAN_SCRIPTS_ROOT"]))
sys.path.insert(0, str(SCRIPT_DIR))

from curvature import estimate_curvatures
from gauss_bonnet_features import (
    estimate_point_area_weights,
    euler_characteristic_from_gauss_bonnet,
    total_gaussian_curvature,
)


def load_points(csv_path: Path) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{csv_path} must contain at least x,y,z columns.")
    return points[:, :3]


def dataset_files(dataset_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        match = VARIANT_RE.match(csv_path.name)
        if match:
            grouped[match.group("class_name")].append(csv_path)
    if not grouped:
        raise FileNotFoundError(f"No <class>_variantXX.csv files found in {dataset_dir}")
    return dict(sorted(grouped.items()))


def variant_index(csv_path: Path) -> int:
    match = VARIANT_RE.match(csv_path.name)
    if match is None:
        return -1
    return int(match.group("idx"))


def process_shape(csv_path: Path, class_name: str, device: torch.device, area_k: int) -> dict:
    points = load_points(csv_path)
    xyz_t, curvatures = estimate_curvatures(points, device)
    gaussian_curvature = curvatures["gaussian_curvature"]

    area_weights = estimate_point_area_weights(xyz_t, k=area_k)
    total_curvature = total_gaussian_curvature(
        gaussian_curvature=gaussian_curvature,
        area_weights=area_weights,
    )
    euler_characteristic = euler_characteristic_from_gauss_bonnet(
        gaussian_curvature=gaussian_curvature,
        area_weights=area_weights,
    )

    chi = float(euler_characteristic.detach().cpu().item())
    total = float(total_curvature.detach().cpu().item())
    return {
        "class_name": class_name,
        "variant": variant_index(csv_path),
        "file_name": csv_path.name,
        "n_points": int(points.shape[0]),
        "total_gaussian_curvature": total,
        "euler_characteristic": chi,
        "euler_characteristic_rounded": int(np.rint(chi)),
    }


def summarize_by_class(rows: list[dict]) -> list[dict]:
    summaries = []
    class_names = sorted({row["class_name"] for row in rows})
    for class_name in class_names:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        totals = np.array([row["total_gaussian_curvature"] for row in class_rows])
        chis = np.array([row["euler_characteristic"] for row in class_rows])
        summaries.append(
            {
                "class_name": class_name,
                "n_shapes": len(class_rows),
                "total_curvature_mean": float(totals.mean()),
                "total_curvature_std": float(totals.std(ddof=0)),
                "euler_characteristic_mean": float(chis.mean()),
                "euler_characteristic_std": float(chis.std(ddof=0)),
                "euler_characteristic_median": float(np.median(chis)),
                "euler_characteristic_rounded_median": int(np.rint(np.median(chis))),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute total Gaussian curvature and Euler characteristic for the Gaussian dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--area-k", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used by the GNP curvature estimator.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap for quick test runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    grouped = dataset_files(args.dataset_dir)

    print(f"Dataset : {args.dataset_dir}")
    print(f"Device  : {device}")
    print(f"Classes : {list(grouped.keys())}")

    rows = []
    for class_name, files in grouped.items():
        files = sorted(files, key=variant_index)
        if args.limit_per_class is not None:
            files = files[: args.limit_per_class]
        for csv_path in files:
            print(f"  {class_name:16s} {csv_path.name}", flush=True)
            rows.append(
                process_shape(
                    csv_path=csv_path,
                    class_name=class_name,
                    device=device,
                    area_k=args.area_k,
                )
            )

    summary_rows = summarize_by_class(rows)
    write_csv(args.results_csv, rows)
    write_csv(args.summary_csv, summary_rows)

    print(f"\nSaved per-shape results: {args.results_csv}")
    print(f"Saved class summary    : {args.summary_csv}")


if __name__ == "__main__":
    main()
