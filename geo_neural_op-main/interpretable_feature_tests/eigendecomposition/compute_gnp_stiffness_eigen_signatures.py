"""
compute_gnp_stiffness_eigen_signatures.py
-----------------------------------------
Build GNP stiffness matrices for the smooth Gaussian dataset, compute
eigenpairs of K = S.T @ S, and render eigenvector signature plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for


ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "only_smoothmeshes_gaussian_dataset"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOTS["FEATURE_SCRIPTS_ROOT"] / "standard_features_scripts"))

from gnp import GeometryEstimator
from graph_laplacian import normalize_points
from plot_feature_signatures import render_signature, zscore_signatures


FEATURE_VERSION = "gnp_gmls_stiffness_eigenvectors_v1"


def read_rows(dataset_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    manifest = dataset_dir / "manifest.csv"
    split_manifest = dataset_dir / "train_test_split.csv"
    if manifest.exists():
        path = manifest
    elif split_manifest.exists():
        path = split_manifest
    else:
        raise FileNotFoundError(f"No manifest.csv or train_test_split.csv in {dataset_dir}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return path, rows


def resolve_csv_path(dataset_dir: Path, row: dict[str, str]) -> Path:
    raw = row.get("csv_path")
    if not raw:
        raise ValueError(f"Manifest row has no csv_path: {row}")
    csv_path = Path(raw)
    if csv_path.exists():
        return csv_path
    fallback = dataset_dir / csv_path.name
    if fallback.exists():
        return fallback
    matches = sorted(dataset_dir.rglob(csv_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve CSV path {raw!r} relative to {dataset_dir}")


def load_points(csv_path: Path, max_points: int | None, random_state: int) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[:, :3]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[indices]
    return points


def estimate_normals(points: np.ndarray, normal_knn: int, orient_knn: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.orient_normals_consistent_tangent_plane(orient_knn)
    return np.asarray(pcd.normals, dtype=np.float32)


def stiffness_energy_matrix(
    points: np.ndarray,
    device: torch.device,
    model: str,
    normal_knn: int,
    orient_knn: int,
    drop_ratio: float,
    radius: float,
    p: int,
    remove_outliers: bool,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    normals = estimate_normals(points, normal_knn=normal_knn, orient_knn=orient_knn)
    xyz_t = torch.tensor(points, dtype=torch.float32, device=device)
    normals_t = torch.tensor(normals, dtype=torch.float32, device=device)
    estimator = GeometryEstimator(xyz_t, orientation=normals_t, model=model, device=device)
    stiffness = estimator.stiffness_matrix_gmls(
        drop_ratio=drop_ratio,
        radius=radius,
        p=p,
        remove_outliers=remove_outliers,
    ).tocsr()
    collocation_mask = estimator.data["stiffness_mask"].detach().cpu().numpy().astype(bool)
    energy = (stiffness.T @ stiffness).tocsr()
    energy = 0.5 * (energy + energy.T)
    energy.eliminate_zeros()
    return stiffness, energy, collocation_mask


def compute_stiffness_eigenpairs(
    energy: sp.csr_matrix,
    n_eigenvectors: int,
    random_state: int,
    eigen_tol: float,
    maxiter: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = energy.shape[0]
    if n_eigenvectors >= n:
        raise ValueError(f"n_eigenvectors must be smaller than N; got {n_eigenvectors} for N={n}")
    rng = np.random.default_rng(random_state)
    v0 = rng.normal(size=n)
    solver_maxiter = maxiter or max(20000, n * 80)
    try:
        values, vectors = eigsh(
            energy,
            k=n_eigenvectors,
            which="SM",
            v0=v0,
            tol=eigen_tol,
            maxiter=solver_maxiter,
        )
    except ArpackNoConvergence as exc:
        values = exc.eigenvalues
        vectors = exc.eigenvectors
        got = 0 if values is None else len(values)
        if got >= n_eigenvectors:
            order = np.argsort(values)
            return np.maximum(values[order], 0.0).astype(np.float64), vectors[:, order].astype(np.float64)

        print(
            f"      plain ARPACK converged {got}/{n_eigenvectors}; retrying with regularized shift-invert...",
            flush=True,
        )
        diag_mean = float(np.mean(np.abs(energy.diagonal()))) if energy.shape[0] else 1.0
        regularization = max(diag_mean, 1.0) * 1e-8
        regularized = energy + regularization * sp.identity(n, format="csr")
        try:
            values, vectors = eigsh(
                regularized,
                k=n_eigenvectors,
                sigma=0.0,
                which="LM",
                v0=v0,
                tol=eigen_tol,
                maxiter=solver_maxiter,
            )
            values = values - regularization
        except Exception as shift_exc:
            raise RuntimeError(
                "ARPACK did not converge enough GNP stiffness eigenpairs, and the "
                "regularized shift-invert fallback also failed. Try running with "
                "--eigenvectors 8 first, or increase --eigen-maxiter."
            ) from shift_exc
    order = np.argsort(values)
    return np.maximum(values[order], 0.0).astype(np.float64), vectors[:, order].astype(np.float64)


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def sample_label(row: dict[str, str]) -> str:
    return Path(row["csv_path"]).stem


def write_index_csv(output_path: Path, rows: list[dict[str, str]], image_paths: list[Path]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "sample", "csv_path", "image_path"],
        )
        writer.writeheader()
        for row, image_path in zip(rows, image_paths):
            writer.writerow(
                {
                    "split": row.get("split", ""),
                    "class_name": row.get("class_name", ""),
                    "sample": sample_label(row),
                    "csv_path": row.get("csv_path", ""),
                    "image_path": str(image_path),
                }
            )


def plot_eigenvector_signatures(
    output_dir: Path,
    rows: list[dict[str, str]],
    eigenvectors: list[np.ndarray],
    eigenvector_index: int,
    max_plots: int | None,
) -> None:
    if not eigenvectors:
        return
    if eigenvector_index < 0 or eigenvector_index >= eigenvectors[0].shape[1]:
        raise ValueError(f"eigenvector_index {eigenvector_index} is outside 0..{eigenvectors[0].shape[1] - 1}")

    vectors = np.vstack([vecs[:, eigenvector_index] for vecs in eigenvectors])
    z_vectors, _, _ = zscore_signatures(vectors)
    order = np.arange(z_vectors.shape[1])
    scores = np.ones(z_vectors.shape[1], dtype=np.float64)
    plot_rows = rows if max_plots is None else rows[:max_plots]
    plot_vectors = z_vectors if max_plots is None else z_vectors[:max_plots]
    image_paths: list[Path] = []

    for row, signature in zip(plot_rows, plot_vectors):
        split = row.get("split", "unknown")
        class_name = row.get("class_name", "unknown")
        label = sample_label(row)
        image_path = output_dir / "eigenvector_signatures" / split / class_name / f"{safe_filename(label)}__eig{eigenvector_index:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        render_signature(
            signature,
            image_path,
            f"{split} | {class_name} | {label} | GNP stiffness eigenvector {eigenvector_index}",
            order=order,
            scores=scores,
            smooth_window=1,
            y_label="Global z-score",
        )
        image_paths.append(image_path)

    write_index_csv(output_dir / f"eigenvector_{eigenvector_index:02d}_signature_index.csv", plot_rows, image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute GNP/GMLS stiffness matrices, eigenpairs, and eigenvector signature plots."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["clean_30k", "clean_50k", "noise_70k", "outlier_50k"], default="clean_30k")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--normalize", choices=["none", "unit_sphere", "rms"], default="unit_sphere")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--normal-knn", type=int, default=30)
    parser.add_argument("--orient-knn", type=int, default=100)
    parser.add_argument("--drop-ratio", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--gmls-p", type=int, default=4)
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--eigenvectors", type=int, default=32)
    parser.add_argument("--eigen-tol", type=float, default=1e-4)
    parser.add_argument("--eigen-maxiter", type=int, default=None)
    parser.add_argument("--signature-eigenvector", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick-run limit over manifest rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based manifest row to start from.")
    parser.add_argument("--end-index", type=int, default=None, help="Zero-based manifest row to stop before.")
    parser.add_argument("--max-plots", type=int, default=None, help="Optional cap on rendered signature plots.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-matrices", action="store_true", help="Save S and K sparse matrices for every sample.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute samples even when eigenpair files already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest_path, rows = read_rows(dataset_dir)
    rows = rows[args.start_index : args.end_index]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No manifest rows selected. Check --start-index, --end-index, and --max-samples.")

    output_dir = args.output_dir or dataset_dir / "gnp_stiffness_eigendecomposition_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "matrices"
    eigenpairs_dir = output_dir / "eigenpairs"
    eigenpairs_dir.mkdir(parents=True, exist_ok=True)
    if args.save_matrices:
        matrices_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = {
        "feature_version": FEATURE_VERSION,
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "model": args.model,
        "device": str(device),
        "normalize": args.normalize,
        "max_points": args.max_points,
        "normal_knn": args.normal_knn,
        "orient_knn": args.orient_knn,
        "drop_ratio": args.drop_ratio,
        "radius": args.radius,
        "gmls_p": args.gmls_p,
        "remove_outliers": args.remove_outliers,
        "eigenvectors": args.eigenvectors,
        "eigen_tol": args.eigen_tol,
        "eigen_maxiter": args.eigen_maxiter,
        "signature_eigenvector": args.signature_eigenvector,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_samples": args.max_samples,
        "random_state": args.random_state,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Output : {output_dir}")
    print(f"Device : {device}")
    print(f"GNP    : model={args.model}, drop_ratio={args.drop_ratio}, radius={args.radius}, p={args.gmls_p}")
    if args.max_points is None and device.type == "cpu":
        print("Note   : full 6000-point CPU runs can be slow; use --max-points 1000 for a faster exploratory run.", flush=True)

    eigenvalue_rows: list[np.ndarray] = []
    eigenvector_rows: list[np.ndarray] = []
    processed_rows: list[dict[str, str]] = []

    for row_idx, row in enumerate(rows):
        csv_path = resolve_csv_path(dataset_dir, row)
        label = sample_label(row)
        stem = safe_filename(label)
        eigenpair_path = eigenpairs_dir / f"{stem}__eigenpairs.npz"
        print(f"  [{row_idx + 1:03d}/{len(rows):03d}] {row.get('split', '?'):5s} {row.get('class_name', '?'):8s} {label}", flush=True)
        if eigenpair_path.exists() and not args.overwrite:
            cached = np.load(eigenpair_path, allow_pickle=True)
            print(f"      using cached eigenpairs: {eigenpair_path.name}", flush=True)
            eigenvalue_rows.append(cached["eigenvalues"])
            eigenvector_rows.append(cached["eigenvectors"])
            processed_rows.append(row)
            continue
        points = load_points(csv_path, max_points=args.max_points, random_state=args.random_state + row_idx)
        points = normalize_points(points, mode=args.normalize).astype(np.float32)
        print(f"      points={len(points)}; estimating normals and GNP stiffness...", flush=True)
        stiffness, energy, collocation_mask = stiffness_energy_matrix(
            points=points,
            device=device,
            model=args.model,
            normal_knn=args.normal_knn,
            orient_knn=args.orient_knn,
            drop_ratio=args.drop_ratio,
            radius=args.radius,
            p=args.gmls_p,
            remove_outliers=args.remove_outliers,
        )
        print(f"      stiffness={stiffness.shape}, nnz={stiffness.nnz}; solving {args.eigenvectors} eigenpairs...", flush=True)
        eigenvalues, eigenvectors = compute_stiffness_eigenpairs(
            energy,
            n_eigenvectors=args.eigenvectors,
            random_state=args.random_state + row_idx,
            eigen_tol=args.eigen_tol,
            maxiter=args.eigen_maxiter,
        )

        if args.save_matrices:
            sp.save_npz(matrices_dir / f"{stem}__stiffness_S.npz", stiffness)
            sp.save_npz(matrices_dir / f"{stem}__energy_K.npz", energy)
        np.savez(
            eigenpair_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            collocation_mask=collocation_mask,
            csv_path=str(csv_path),
            class_name=row.get("class_name", ""),
            split=row.get("split", ""),
            sample=label,
        )
        eigenvalue_rows.append(eigenvalues)
        eigenvector_rows.append(eigenvectors)
        processed_rows.append(row)
        print(f"      saved {eigenpair_path.name}", flush=True)

    np.savez(
        output_dir / "features.npz",
        eigenvalues=np.vstack(eigenvalue_rows),
        classes=np.array(sorted({row.get("class_name", "") for row in processed_rows})),
        samples=np.array([sample_label(row) for row in processed_rows]),
        labels=np.array([row.get("class_name", "") for row in processed_rows]),
        splits=np.array([row.get("split", "") for row in processed_rows]),
        feature_names=np.array([f"lambda_{idx:02d}" for idx in range(args.eigenvectors)]),
        feature_version=FEATURE_VERSION,
        **{
            key: value
            for key, value in config.items()
            if key != "feature_version" and (isinstance(value, (str, int, float, bool)) or value is None)
        },
    )

    plot_eigenvector_signatures(
        output_dir=output_dir,
        rows=processed_rows,
        eigenvectors=eigenvector_rows,
        eigenvector_index=args.signature_eigenvector,
        max_plots=args.max_plots,
    )
    print(f"Saved eigenpairs to {eigenpairs_dir}")
    print(f"Saved summary features to {output_dir / 'features.npz'}")
    print(f"Saved eigenvector signature plots to {output_dir / 'eigenvector_signatures'}")


if __name__ == "__main__":
    main()
