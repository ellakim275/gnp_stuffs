"""
plot_feature_signatures.py
--------------------------
Create 103-feature visual signatures for point-cloud CSV files.

A 103-d signature corresponds to one scalar signal:
  12 half-space indicator features
  27 voxel features
  64 Fourier features

This script is focused on mean-curvature signatures by default. The default
view is class-aware: features are globally z-scored, sorted by how well they
separate classes, lightly smoothed for readability, and summarized as class
prototype barcode heatmaps.

By default this script processes all variant CSVs in:
    output/gaussian/datasets/first_mixed_gaussian_dataset/gaussian_augmented_dataset/

Outputs:
    output/gaussian/datasets/first_mixed_gaussian_dataset/gaussian_augmented_dataset/feature_signatures/
        class_prototypes__<signal>.png
        class_examples__<signal>.png
        feature_order__<signal>.csv
        <name>__<signal>.png  (when --mode individual or --mode both)
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
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from path_config import roots_for

ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
TEST_ROOT = ROOTS["TEST_ROOT"]
DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "first_mixed_gaussian_dataset" / "gaussian_augmented_dataset"
OUTPUT_DIR = DATASET_DIR / "feature_signatures"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOTS["FEATURE_SCRIPTS_ROOT"]))
sys.path.insert(0, str(ROOTS["GAUSSIAN_SCRIPTS_ROOT"]))

from curvature import estimate_curvatures
from features import extract_features, fourier_features, indicator_features, indicator_features_3d


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANT_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_]+)\.csv$")
SIGNAL_CHOICES = {"mean_curvature", "gaussian_curvature", "density"}

INDICATOR_BINS = 4
VOXEL_BINS = 3
N_FOURIER = 64
FOURIER_START = 39
FOURIER_END = 103
FOURIER_MAX_FREQ = 4.0
FOURIER_SEED = 42
FOURIER_HARMONICS = 16
FEATURE_GROUPS = (
    ("indicator", 0, 12, "#287271"),
    ("voxel", 12, 39, "#ee964b"),
    ("fourier", 39, 103, "#5b5f97"),
)


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
            max_freq=FOURIER_MAX_FREQ,
            seed=FOURIER_SEED,
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


def infer_class_label(csv_path: Path) -> str:
    stem = csv_path.stem
    for token in ("_variant", "__", "_aug"):
        if token in stem:
            return stem.split(token)[0]
    match = re.match(r"^(?P<label>.+?)_\d+$", stem)
    return match.group("label") if match else stem


def feature_group(index: int) -> str:
    for group_name, start, end, _ in FEATURE_GROUPS:
        if start <= index < end:
            return group_name
    return "unknown"


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def zscore_signatures(signatures: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = signatures.mean(axis=0)
    std = signatures.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (signatures - mean) / std, mean, std


def class_stats(
    z_signatures: np.ndarray,
    labels: list[str],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    class_names = sorted(set(labels))
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    label_arr = np.array(labels)
    for class_name in class_names:
        rows = z_signatures[label_arr == class_name]
        means[class_name] = rows.mean(axis=0)
        stds[class_name] = rows.std(axis=0) if len(rows) > 1 else np.zeros(rows.shape[1])
    return class_names, means, stds


def discriminative_feature_order(
    z_signatures: np.ndarray,
    labels: list[str],
    top_features: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    class_names, means, stds = class_stats(z_signatures, labels)
    mean_matrix = np.vstack([means[name] for name in class_names])
    std_matrix = np.vstack([stds[name] for name in class_names])

    between_class = mean_matrix.var(axis=0)
    within_class = np.mean(std_matrix ** 2, axis=0)
    scores = between_class / (within_class + 1e-6)
    order = np.argsort(scores)[::-1]
    if top_features is not None:
        order = order[:top_features]
    return order, scores


def feature_opacity(scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    ordered = scores[order]
    if np.allclose(ordered.max(), ordered.min()):
        return np.ones_like(ordered)
    scaled = (ordered - ordered.min()) / (ordered.max() - ordered.min())
    return 0.35 + 0.65 * scaled


def ordered_smoothed(values: np.ndarray, order: np.ndarray, smooth_window: int) -> np.ndarray:
    return smooth_1d(values[order], smooth_window)


def render_signature(
    signature: np.ndarray,
    out_path: Path,
    title: str,
    order: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    smooth_window: int = 1,
    y_label: str = "Feature Value",
) -> None:
    if order is None:
        order = np.arange(signature.shape[0])

    plotted = ordered_smoothed(signature, order, smooth_window)
    x = np.arange(plotted.shape[0])
    opacity = feature_opacity(scores, order) if scores is not None else np.ones_like(plotted)
    group_colors = {name: color for name, _, _, color in FEATURE_GROUPS}
    colors = []
    for alpha, original_index in zip(opacity, order):
        base = group_colors.get(feature_group(int(original_index)), "#555555")
        colors.append((*matplotlib.colors.to_rgb(base), float(alpha)))

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x, plotted, color=colors, width=0.9)
    max_abs = max(float(np.max(np.abs(plotted))), 1e-6)
    ylim = 1.1 * max_abs
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlim(-1, len(plotted))
    ax.set_xlabel("Ranked Feature Position (most class-discriminative on the left)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    # Sparse ticks so the axis stays readable.
    tick_positions = np.linspace(0, len(plotted) - 1, min(12, len(plotted)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(order[t])) for t in tick_positions], rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_class_prototypes(
    class_names: list[str],
    class_means: dict[str, np.ndarray],
    class_stds: dict[str, np.ndarray],
    order: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    signal: str,
    smooth_window: int,
) -> None:
    matrix = np.vstack([
        ordered_smoothed(class_means[class_name], order, smooth_window)
        for class_name in class_names
    ])
    consistency = np.vstack([
        1.0 / (ordered_smoothed(class_stds[class_name], order, smooth_window) + 0.25)
        for class_name in class_names
    ])
    consistency = consistency / max(float(consistency.max()), 1e-6)
    consistency = 0.35 + 0.65 * consistency
    consistency *= feature_opacity(scores, order)[None, :]

    cmap = plt.get_cmap("vlag" if "vlag" in plt.colormaps() else "coolwarm")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
    rgba = cmap(norm(np.clip(matrix, -2.5, 2.5)))
    rgba[..., 3] = np.clip(consistency, 0.25, 1.0)

    height = max(3.2, 0.72 * len(class_names) + 1.6)
    fig, ax = plt.subplots(figsize=(18, height))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Ranked Feature Position (most class-discriminative on the left)")
    ax.set_title(f"Class Prototype Curvature Signatures — {signal}")
    tick_positions = np.linspace(0, len(order) - 1, min(12, len(order)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(order[t])) for t in tick_positions], rotation=0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Global z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_class_examples(
    z_signatures: np.ndarray,
    names: list[str],
    labels: list[str],
    class_names: list[str],
    class_means: dict[str, np.ndarray],
    order: np.ndarray,
    out_path: Path,
    signal: str,
    smooth_window: int,
    examples_per_class: int,
) -> None:
    rows = []
    y_labels = []
    label_arr = np.array(labels)

    for class_name in class_names:
        rows.append(ordered_smoothed(class_means[class_name], order, smooth_window))
        y_labels.append(f"{class_name} prototype")
        class_indices = np.where(label_arr == class_name)[0][:examples_per_class]
        for idx in class_indices:
            rows.append(ordered_smoothed(z_signatures[idx], order, smooth_window))
            y_labels.append(f"  {names[idx]}")

    matrix = np.vstack(rows)
    cmap = plt.get_cmap("vlag" if "vlag" in plt.colormaps() else "coolwarm")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
    height = max(6.0, 0.34 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(18, height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Ranked Feature Position (most class-discriminative on the left)")
    ax.set_title(f"Prototype + Example Curvature Signatures — {signal}")
    tick_positions = np.linspace(0, len(order) - 1, min(12, len(order)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(order[t])) for t in tick_positions], rotation=0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Global z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fourier_effective_frequencies() -> np.ndarray:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(FOURIER_SEED)
    omega = (torch.rand(N_FOURIER, FOURIER_HARMONICS, 3, generator=rng) * 2 - 1) * FOURIER_MAX_FREQ
    alpha = torch.randn(N_FOURIER, FOURIER_HARMONICS, generator=rng)
    alpha = alpha / alpha.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.sqrt((alpha.pow(2) * omega.norm(dim=2).pow(2)).sum(dim=1)).numpy()


def render_fourier_spectrum(
    signatures: np.ndarray,
    names: list[str],
    labels: list[str],
    class_names: list[str],
    out_path: Path,
    csv_path: Path,
    signal: str,
    examples_per_class: int,
) -> None:
    coeffs = signatures[:, FOURIER_START:FOURIER_END]
    energy = coeffs ** 2
    frequencies = fourier_effective_frequencies()
    order = np.argsort(frequencies)
    ordered_frequencies = frequencies[order]
    label_arr = np.array(labels)
    colors = ["#287271", "#ee964b", "#5b5f97", "#c44536", "#6a994e", "#725ac1"]

    fig, ax = plt.subplots(figsize=(13, 6))
    csv_rows = ["class,name,mode_index,effective_frequency,energy\n"]
    x = np.arange(N_FOURIER)

    for class_idx, class_name in enumerate(class_names):
        color = colors[class_idx % len(colors)]
        class_indices = np.where(label_arr == class_name)[0][:examples_per_class]
        for example_idx, row_idx in enumerate(class_indices, start=1):
            example_energy = energy[row_idx, order]
            line_label = f"{class_name} {example_idx}: {names[row_idx]}"
            ax.plot(x, example_energy, label=line_label, color=color, linewidth=1.5, alpha=0.72)
            for mode_position, original_mode in enumerate(order):
                csv_rows.append(
                    f"{class_name},{names[row_idx]},{int(original_mode)},"
                    f"{ordered_frequencies[mode_position]:.8g},{example_energy[mode_position]:.8g}\n"
                )

    ax.set_title(f"Individual Fourier Curvature Signatures - {signal}")
    ax.set_xlabel("Effective frequency")
    ax.set_ylabel("Energy |a_k|^2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncols=2)

    tick_positions = np.linspace(0, N_FOURIER - 1, 6, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{ordered_frequencies[idx]:.1f}" for idx in tick_positions])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.writelines(csv_rows)


def save_feature_order(out_path: Path, order: np.ndarray, scores: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("rank,feature_index,feature_group,score\n")
        for rank, feature_index in enumerate(order, start=1):
            handle.write(
                f"{rank},{int(feature_index)},{feature_group(int(feature_index))},"
                f"{float(scores[feature_index]):.8g}\n"
            )


def iter_input_csvs(target: str | None) -> list[Path]:
    if target:
        path = Path(target).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing input path: {path}")
        if path.is_file():
            return [path]
        return sorted(p for p in path.glob("*.csv") if VARIANT_RE.match(p.name))

    return sorted(p for p in DATASET_DIR.glob("*.csv") if VARIANT_RE.match(p.name))


def filter_csvs_by_class(
    csv_paths: list[Path],
    class_filter: str | None,
    max_per_class: int | None,
) -> list[Path]:
    allowed = None
    if class_filter:
        allowed = {item.strip() for item in class_filter.split(",") if item.strip()}

    counts: dict[str, int] = {}
    selected: list[Path] = []
    for csv_path in csv_paths:
        class_name = infer_class_label(csv_path)
        if allowed is not None and class_name not in allowed:
            continue
        if max_per_class is not None and counts.get(class_name, 0) >= max_per_class:
            continue
        selected.append(csv_path)
        counts[class_name] = counts.get(class_name, 0) + 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot class-aware 103-feature curvature signatures for point-cloud CSV files."
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to the gaussian dataset feature_signatures folder.",
    )
    parser.add_argument(
        "--mode",
        choices=("class", "individual", "both", "fourier-spectrum"),
        default="class",
        help="Output class prototype sheets, individual barcodes, Fourier spectrum, or both. Defaults to class.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class labels to include, for example bathtub,desk.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on CSVs processed per class after class filtering.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=103,
        help="Number of ranked features to visualize. Defaults to all 103.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Odd rolling window for visual smoothing after feature ranking. Use 1 to disable.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=4,
        help="Number of individual examples shown under each prototype in the class example sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = iter_input_csvs(args.input)
    csv_paths = filter_csvs_by_class(csv_paths, args.classes, args.max_per_class)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found to visualize.")

    print(f"Device : {DEVICE}", flush=True)
    print(f"Signal : {args.signal}", flush=True)
    print(f"Inputs : {len(csv_paths)} CSV files", flush=True)

    names: list[str] = []
    labels: list[str] = []
    signatures: list[np.ndarray] = []
    for csv_path in csv_paths:
        name = csv_path.stem
        print(f"  extracting {name}", flush=True)
        points = load_points(csv_path)
        signatures.append(compute_signature(points, args.signal))
        names.append(name)
        labels.append(infer_class_label(csv_path))

    signature_matrix = np.vstack(signatures)
    z_signatures, _, _ = zscore_signatures(signature_matrix)
    top_features = min(max(args.top_features, 1), signature_matrix.shape[1])
    order, scores = discriminative_feature_order(z_signatures, labels, top_features)
    class_names, class_means, class_stds = class_stats(z_signatures, labels)
    save_feature_order(output_dir / f"feature_order__{args.signal}.csv", order, scores)
    np.savez(
        output_dir / f"signatures__{args.signal}.npz",
        signatures=signature_matrix,
        z_signatures=z_signatures,
        names=np.array(names),
        labels=np.array(labels),
        classes=np.array(class_names),
        feature_order=order,
        feature_scores=scores,
    )

    if args.mode in ("class", "both"):
        print("  rendering class prototype sheets", flush=True)
        render_class_prototypes(
            class_names,
            class_means,
            class_stds,
            order,
            scores,
            output_dir / f"class_prototypes__{args.signal}.png",
            args.signal,
            args.smooth_window,
        )
        render_class_examples(
            z_signatures,
            names,
            labels,
            class_names,
            class_means,
            order,
            output_dir / f"class_examples__{args.signal}.png",
            args.signal,
            args.smooth_window,
            args.examples_per_class,
        )

    if args.mode in ("individual", "both"):
        print("  rendering individual ranked barcodes", flush=True)
        for name, signature in zip(names, z_signatures):
            out_path = output_dir / f"{name}__{args.signal}.png"
            render_signature(
                signature,
                out_path,
                f"{name} — ranked {args.signal} signature",
                order=order,
                scores=scores,
                smooth_window=args.smooth_window,
                y_label="Global z-score",
            )

    if args.mode == "fourier-spectrum":
        print("  rendering Fourier spectrum", flush=True)
        class_suffix = "__" + "_".join(class_names) if args.classes else ""
        render_fourier_spectrum(
            signature_matrix,
            names,
            labels,
            class_names,
            output_dir / f"fourier_spectrum__{args.signal}{class_suffix}.png",
            output_dir / f"fourier_individual_energy__{args.signal}{class_suffix}.csv",
            args.signal,
            args.examples_per_class,
        )

    print(f"Saved plots to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
