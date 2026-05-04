"""
plot_feature_signatures.py
--------------------------
Create 103-feature bar-chart signatures for point-cloud CSV files.

A 103-d signature corresponds to one scalar signal:
  12 half-space indicator features
  27 voxel features
  64 Fourier features

This script is focused on mean-curvature signatures by default.

By default this script processes all variant CSVs in:
    output/gaussian_augmented_dataset/

Outputs:
    output/gaussian_augmented_dataset/feature_signatures/<name>__<signal>.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_DIR = REPO_ROOT / "output" / "gaussian_augmented_dataset"
OUTPUT_DIR = DATASET_DIR / "feature_signatures"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from curvature import estimate_curvatures
from features import extract_features, fourier_features, indicator_features, indicator_features_3d


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANT_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_]+)\.csv$")
SIGNAL_CHOICES = {"mean_curvature", "gaussian_curvature", "density"}

INDICATOR_BINS = 4
VOXEL_BINS = 3
N_FOURIER = 64


def load_points(csv_path: Path) -> np.ndarray:
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


def compute_signature(points: np.ndarray, signal_name: str) -> np.ndarray:
    xyz_t, curvatures = estimate_curvatures(points, DEVICE)

    if signal_name == "density":
        # A constant density signal becomes all zeros if passed through the
        # full extract_features normalization path. For interpretability plots,
        # use raw occupancy-style features instead.
        signal = torch.ones(xyz_t.shape[0], device=xyz_t.device)
        hs = indicator_features(xyz_t, signal, n_bins=INDICATOR_BINS).cpu().numpy()
        vx = indicator_features_3d(xyz_t, signal, n_bins=VOXEL_BINS).cpu().numpy()
        ff = fourier_features(
            xyz_t,
            signal,
            n_features=N_FOURIER,
            max_freq=4.0,
            seed=42,
        ).cpu().numpy()
        return np.concatenate([hs, vx, ff])
    else:
        signal_dict = {signal_name: curvatures[signal_name]}

    feats = extract_features(
        xyz_t,
        signal_dict,
        indicator_bins=INDICATOR_BINS,
        voxel_bins=VOXEL_BINS,
        n_fourier=N_FOURIER,
    )

    prefix = signal_name
    hs = feats[f"{prefix}_indicator_halfspace"].cpu().numpy()
    vx = feats[f"{prefix}_indicator_voxel"].cpu().numpy()
    ff = feats[f"{prefix}_fourier"].cpu().numpy()
    return np.concatenate([hs, vx, ff])


def render_signature(signature: np.ndarray, out_path: Path, title: str) -> None:
    x = np.arange(signature.shape[0])
    colors = (
        ["#287271"] * 12 +
        ["#ee964b"] * 27 +
        ["#5b5f97"] * 64
    )

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x, signature, color=colors, width=0.9)
    max_abs = max(float(np.max(np.abs(signature))), 1e-6)
    ylim = 1.1 * max_abs
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(11.5, color="black", linewidth=1.0, alpha=0.5)
    ax.axvline(38.5, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlim(-1, len(signature))
    ax.set_xlabel("Feature Index (12 indicator | 27 voxel | 64 fourier)")
    ax.set_ylabel("Feature Value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    # Sparse ticks so the axis stays readable.
    tick_positions = list(range(0, 12, 2)) + [12, 18, 24, 30, 36] + [39, 47, 55, 63, 71, 79, 87, 95, 102]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions], rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def iter_input_csvs(target: str | None) -> list[Path]:
    if target:
        path = Path(target).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing input path: {path}")
        if path.is_file():
            return [path]
        return sorted(p for p in path.glob("*.csv") if VARIANT_RE.match(p.name))

    return sorted(p for p in DATASET_DIR.glob("*.csv") if VARIANT_RE.match(p.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 103-feature mean-curvature bar signatures for point-cloud CSV files."
    )
    parser.add_argument(
        "--signal",
        choices=sorted(SIGNAL_CHOICES),
        default="mean_curvature",
        help="Which 103-feature signal signature to visualize. Defaults to mean_curvature.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional CSV file or directory. Defaults to the gaussian dataset folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = iter_input_csvs(args.input)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found to visualize.")

    print(f"Device : {DEVICE}", flush=True)
    print(f"Signal : {args.signal}", flush=True)
    print(f"Inputs : {len(csv_paths)} CSV files", flush=True)

    for csv_path in csv_paths:
        name = csv_path.stem
        print(f"  plotting {name}", flush=True)
        points = load_points(csv_path)
        signature = compute_signature(points, args.signal)
        out_path = OUTPUT_DIR / f"{name}__{args.signal}.png"
        render_signature(signature, out_path, f"{name} — {args.signal} signature")

    print(f"Saved plots to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
