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
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "only_smoothmeshes_gaussian_dataset"
VARIANT_RE = re.compile(r"^(?P<class_name>[a-zA-Z0-9_]+)_variant(?P<idx>\d+)\.csv$")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOTS["GAUSSIAN_SCRIPTS_ROOT"]))
sys.path.insert(0, str(SCRIPT_DIR))

from curvature import estimate_curvatures
from gauss_bonnet_features import (
    estimate_point_area_weights,
    euler_characteristic_from_gauss_bonnet,
    gaussian_curvature_integral,
)


def load_points(csv_path: Path) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"{csv_path} must contain at least x,y,z columns.")
    return points[:, :3]


def optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        return None
    return result


def resolve_manifest_csv_path(dataset_dir: Path, raw_path: str) -> Path:
    csv_path = Path(raw_path)
    if csv_path.exists():
        return csv_path
    fallback = dataset_dir / csv_path.name
    if fallback.exists():
        return fallback
    matches = sorted(dataset_dir.rglob(csv_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve CSV path {raw_path!r} relative to {dataset_dir}")


def dataset_entries(dataset_dir: Path) -> list[dict]:
    manifest = dataset_dir / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = []
            for row in csv.DictReader(handle):
                csv_path = resolve_manifest_csv_path(dataset_dir, row["csv_path"])
                rows.append(
                    {
                        "class_name": row["class_name"],
                        "split": row.get("split", ""),
                        "source_id": row.get("source_id", ""),
                        "variant": int(row.get("variant_idx") or variant_index(csv_path)),
                        "csv_path": csv_path,
                        "mesh_path": row.get("mesh_path", ""),
                        "source_surface_area": optional_float(
                            row.get("source_surface_area") or row.get("surface_area")
                        ),
                    }
                )
        if not rows:
            raise ValueError(f"Manifest is empty: {manifest}")
        return sorted(rows, key=lambda row: (row["class_name"], row["split"], row["variant"], row["csv_path"].name))

    grouped: dict[str, list[Path]] = defaultdict(list)
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        match = VARIANT_RE.match(csv_path.name)
        if match:
            grouped[match.group("class_name")].append(csv_path)
    if not grouped:
        dataset_root = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets"
        candidates = sorted(
            {
                candidate.parent
                for candidate in dataset_root.rglob("*")
                if candidate.name in {"manifest.csv", "train_test_split.csv"}
            }
        )
        candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"No manifest.csv or <class>_variantXX.csv files found in {dataset_dir}\n"
            f"Available manifest dataset dirs:\n{candidate_text}"
        )
    return [
        {
            "class_name": class_name,
            "split": "",
            "source_id": "",
            "variant": variant_index(csv_path),
            "csv_path": csv_path,
            "mesh_path": "",
            "source_surface_area": None,
        }
        for class_name, files in sorted(grouped.items())
        for csv_path in sorted(files, key=variant_index)
    ]


def variant_index(csv_path: Path) -> int:
    match = VARIANT_RE.match(csv_path.name)
    if match is None:
        return -1
    return int(match.group("idx"))


def process_shape(
    entry: dict,
    device: torch.device,
    area_k: int,
) -> dict:
    csv_path = entry["csv_path"]
    points = load_points(csv_path)
    xyz_t, curvatures = estimate_curvatures(points, device)
    gaussian_curvature = curvatures["gaussian_curvature"]
    area_weights = estimate_point_area_weights(xyz_t, k=area_k)

    curvature_integral = gaussian_curvature_integral(
        gaussian_curvature=gaussian_curvature,
        area_weights=area_weights,
    )
    euler_characteristic = euler_characteristic_from_gauss_bonnet(
        gaussian_curvature=gaussian_curvature,
        area_weights=area_weights,
    )

    chi = float(euler_characteristic.detach().cpu().item())
    integral = float(curvature_integral.detach().cpu().item())
    return {
        "class_name": entry["class_name"],
        "split": entry["split"],
        "source_id": entry["source_id"],
        "variant": entry["variant"],
        "file_name": csv_path.name,
        "csv_path": str(csv_path),
        "n_points": int(points.shape[0]),
        "area_weight_source": "knn_local_disk",
        "gaussian_curvature_integral": integral,
        "euler_characteristic": chi,
        "euler_characteristic_rounded": int(np.rint(chi)),
    }


def summarize_by_class(rows: list[dict]) -> list[dict]:
    summaries = []
    class_names = sorted({row["class_name"] for row in rows})
    for class_name in class_names:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        integrals = np.array([row["gaussian_curvature_integral"] for row in class_rows])
        chis = np.array([row["euler_characteristic"] for row in class_rows])
        summaries.append(
            {
                "class_name": class_name,
                "n_shapes": len(class_rows),
                "gaussian_curvature_integral_mean": float(integrals.mean()),
                "gaussian_curvature_integral_std": float(integrals.std(ddof=0)),
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
        description="Compute the Gaussian curvature integral and Euler characteristic for the Gaussian dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
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
    dataset_dir = args.dataset_dir.resolve()
    results_csv = args.results_csv or dataset_dir / "gauss_bonnet_results.csv"
    summary_csv = args.summary_csv or dataset_dir / "gauss_bonnet_class_summary.csv"
    entries = dataset_entries(dataset_dir)

    print(f"Dataset : {dataset_dir}")
    print(f"Device  : {device}")
    print(f"Area mode: knn")
    print(f"Classes : {sorted({entry['class_name'] for entry in entries})}")

    rows = []
    counts_by_class: dict[str, int] = defaultdict(int)
    for entry in entries:
        class_name = entry["class_name"]
        if args.limit_per_class is not None and counts_by_class[class_name] >= args.limit_per_class:
            continue
        counts_by_class[class_name] += 1
        print(f"  {class_name:16s} {entry['split']:5s} {entry['csv_path'].name}", flush=True)
        rows.append(
            process_shape(
                entry=entry,
                device=device,
                area_k=args.area_k,
            )
        )

    summary_rows = summarize_by_class(rows)
    write_csv(results_csv, rows)
    write_csv(summary_csv, summary_rows)

    print(f"\nSaved per-shape results: {results_csv}")
    print(f"Saved class summary    : {summary_csv}")


if __name__ == "__main__":
    main()
