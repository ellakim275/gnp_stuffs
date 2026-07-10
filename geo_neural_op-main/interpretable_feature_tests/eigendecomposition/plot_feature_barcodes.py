"""
plot_feature_barcodes.py
------------------------
Render one barcode-style bar graph per eigendecomposition feature vector.

The script reads the cached features.npz produced by
train_svm_eigendecomposition.py and uses the dataset manifest to recover sample
names. Every graph uses the same feature-index x-axis and the same y-axis
limits so the barcodes are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for


ROOTS = roots_for(__file__)
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "only_smoothmeshes_gaussian_dataset"


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if row.get("split") == "train"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    return train_rows, test_rows


def sample_label(row: dict[str, str]) -> str:
    csv_path = Path(row["csv_path"])
    return csv_path.stem


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def scaled_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if mode == "raw":
        return X_train, X_test, "Raw feature value"
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, "Train z-scored feature value"


def robust_ylim(values: np.ndarray, percentile: float) -> tuple[float, float]:
    if percentile >= 100.0:
        low = float(np.min(values))
        high = float(np.max(values))
    else:
        tail = (100.0 - percentile) / 2.0
        low, high = np.percentile(values, [tail, 100.0 - tail])
        low = float(low)
        high = float(high)
    span = max(high - low, 1e-6)
    pad = 0.05 * span
    return low - pad, high + pad


def draw_barcode(
    values: np.ndarray,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float],
    ylabel: str,
    color: str,
    xtick_step: int,
) -> None:
    n_features = len(values)
    x = np.arange(n_features)
    fig, ax = plt.subplots(figsize=(14, 3.4), dpi=180)
    ax.bar(x, values, width=1.0, color=color, linewidth=0)
    ax.axhline(0.0, color="#222222", linewidth=0.7)
    ax.set_xlim(-1, n_features)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Feature index")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(0, n_features, xtick_step))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_index_csv(
    output_path: Path,
    rows: list[dict[str, str]],
    split: str,
    class_names: np.ndarray,
    image_paths: list[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "sample", "csv_path", "image_path"],
        )
        writer.writeheader()
        for row, class_name, image_path in zip(rows, class_names, image_paths):
            writer.writerow(
                {
                    "split": split,
                    "class_name": class_name,
                    "sample": sample_label(row),
                    "csv_path": row["csv_path"],
                    "image_path": str(image_path),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot barcode bar graphs for eigendecomposition feature vectors.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scale", choices=["zscore", "raw"], default="zscore")
    parser.add_argument("--ylim-percentile", type=float, default=99.0)
    parser.add_argument("--xtick-step", type=int, default=25)
    parser.add_argument("--max-per-split", type=int, default=None, help="Optional quick-look limit per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    features_path = args.features_path or dataset_dir / "svm_eigendecomposition_outputs" / "features.npz"
    output_dir = args.output_dir or dataset_dir / "svm_eigendecomposition_outputs" / f"feature_barcodes_{args.scale}"
    rows = read_manifest(dataset_dir)
    train_rows, test_rows = split_rows(rows)

    cached = np.load(features_path, allow_pickle=True)
    X_train = cached["X_train"]
    X_test = cached["X_test"]
    y_train = cached["y_train"]
    y_test = cached["y_test"]
    classes = np.asarray(cached["classes"]).astype(str)

    if len(train_rows) != len(X_train) or len(test_rows) != len(X_test):
        raise ValueError(
            "Manifest split counts do not match cached feature matrices: "
            f"train rows/features {len(train_rows)}/{len(X_train)}, "
            f"test rows/features {len(test_rows)}/{len(X_test)}"
        )

    X_train_plot, X_test_plot, ylabel = scaled_features(X_train, X_test, args.scale)
    all_plot_values = np.vstack([X_train_plot, X_test_plot])
    y_limits = robust_ylim(all_plot_values, percentile=args.ylim_percentile)
    color_by_split = {"train": "#2f6f9f", "test": "#b45f06"}

    if args.max_per_split is not None:
        train_rows = train_rows[: args.max_per_split]
        test_rows = test_rows[: args.max_per_split]
        X_train_plot = X_train_plot[: args.max_per_split]
        X_test_plot = X_test_plot[: args.max_per_split]
        y_train = y_train[: args.max_per_split]
        y_test = y_test[: args.max_per_split]

    print(f"Dataset: {dataset_dir}")
    print(f"Features: {features_path}")
    print(f"Output: {output_dir}")
    print(f"Scale: {args.scale}")
    print(f"Shared y-limits: {y_limits[0]:.4g}, {y_limits[1]:.4g}")

    for split, split_rows_in, X_split, y_split in [
        ("train", train_rows, X_train_plot, y_train),
        ("test", test_rows, X_test_plot, y_test),
    ]:
        image_paths: list[Path] = []
        class_names = classes[y_split]
        for values, row, class_name in zip(X_split, split_rows_in, class_names):
            label = sample_label(row)
            image_path = output_dir / split / str(class_name) / f"{safe_filename(label)}.png"
            title = f"{split} | {class_name} | {label}"
            draw_barcode(
                values,
                title=title,
                output_path=image_path,
                y_limits=y_limits,
                ylabel=ylabel,
                color=color_by_split[split],
                xtick_step=args.xtick_step,
            )
            image_paths.append(image_path)
        write_index_csv(output_dir / f"{split}_barcode_index.csv", split_rows_in, split, class_names, image_paths)
        print(f"Wrote {len(image_paths)} {split} barcodes")


if __name__ == "__main__":
    main()
