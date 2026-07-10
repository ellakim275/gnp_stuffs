"""
train_svm_eigendecomposition.py
-------------------------------
Train/test an SVM using density-corrected graph Laplacian eigenvalues.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for

ROOTS = roots_for(__file__)
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "only_smoothmeshes_gaussian_dataset"

sys.path.insert(0, str(SCRIPT_DIR))

from graph_laplacian import (
    build_knn_graph,
    build_laplacian,
    compute_eigendecomposition,
    density_correct_weights,
    gaussian_weight_matrix,
    normalize_points,
)


FEATURE_VERSION = "density_corrected_graph_laplacian_eigenvalues_v2"


def eigenvalue_feature_vector(eigenvalues: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    names = [f"lambda_{idx:02d}" for idx in range(len(values))]
    return values.astype(np.float32), names


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


def load_points(csv_path: Path, max_points: int | None) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[:, :3]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[indices]
    return points


def process_points(
    points: np.ndarray,
    knn_k: int,
    n_eigenvectors: int,
    normalize: str,
    sigma_mode: str,
    density_alpha: float,
    laplacian: str,
    random_state: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    points = normalize_points(points, mode=normalize)
    indices, distances = build_knn_graph(points, k=knn_k)
    W = gaussian_weight_matrix(points, indices, distances, sigma_mode=sigma_mode)
    W_tilde = density_correct_weights(W, alpha=density_alpha)
    L = build_laplacian(W_tilde, kind=laplacian)
    eigenvalues, _ = compute_eigendecomposition(L, n_eigenvectors=n_eigenvectors, random_state=random_state)
    feature, feature_names = eigenvalue_feature_vector(eigenvalues)
    return feature, feature_names, eigenvalues


def cache_is_fresh(
    cache_path: Path,
    manifest_path: Path,
    rows: list[dict[str, str]],
    dataset_dir: Path,
    config: dict[str, str | int | float],
) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached["feature_version"]) != FEATURE_VERSION:
            return False
        for key, expected in config.items():
            if key not in cached.files:
                return False
            actual = cached[key].item() if cached[key].shape == () else cached[key]
            if isinstance(expected, float):
                if not np.isclose(float(actual), expected):
                    return False
            elif actual != expected:
                return False
    except Exception:
        return False
    cache_mtime = cache_path.stat().st_mtime
    if manifest_path.stat().st_mtime > cache_mtime:
        return False
    for row in rows:
        if resolve_csv_path(dataset_dir, row).stat().st_mtime > cache_mtime:
            return False
    return True


def build_features(
    rows: list[dict[str, str]],
    dataset_dir: Path,
    classes: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[np.ndarray]]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    X_train, y_train, X_test, y_test = [], [], [], []
    feature_names: list[str] | None = None
    eigenvalue_rows: list[np.ndarray] = []

    for row_idx, row in enumerate(rows):
        split = row["split"]
        class_name = row["class_name"]
        csv_path = resolve_csv_path(dataset_dir, row)
        print(f"  [{split}] {class_name:8s} {csv_path.name}", flush=True)
        feature, names, eigenvalues = process_points(
            load_points(csv_path, max_points=args.max_points),
            knn_k=args.knn_k,
            n_eigenvectors=args.eigenvectors,
            normalize=args.normalize,
            sigma_mode=args.sigma_mode,
            density_alpha=args.density_alpha,
            laplacian=args.laplacian,
            random_state=args.random_state + row_idx,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed between samples")
        eigenvalue_rows.append(eigenvalues)

        if split == "train":
            X_train.append(feature)
            y_train.append(class_to_idx[class_name])
        elif split == "test":
            X_test.append(feature)
            y_test.append(class_to_idx[class_name])
        else:
            raise ValueError(f"Unknown split {split!r} in {csv_path}")

    if feature_names is None:
        raise ValueError("No features were built")
    return (
        np.vstack(X_train),
        np.asarray(y_train),
        np.vstack(X_test),
        np.asarray(y_test),
        feature_names,
        eigenvalue_rows,
    )


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> tuple[StandardScaler, SVC, dict, float, float, str, np.ndarray]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    cv = min(5, int(np.bincount(y_train).min()))
    param_grid = [
        {"kernel": ["linear"], "C": [0.1, 1.0, 10.0, 100.0]},
        {"kernel": ["rbf"], "C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", "auto", 1e-3, 1e-2]},
        {"kernel": ["poly"], "C": [0.1, 1.0, 10.0], "degree": [2, 3], "gamma": ["scale", "auto"]},
    ]
    grid = GridSearchCV(SVC(decision_function_shape="ovr", random_state=0), param_grid, cv=cv, scoring="accuracy")
    grid.fit(X_train_s, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    test_acc = float(accuracy_score(y_test, y_pred))
    return scaler, clf, grid.best_params_, float(grid.best_score_), test_acc, report, cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM using graph Laplacian eigenvalues.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--eigenvectors", type=int, default=32)
    parser.add_argument("--normalize", choices=["none", "unit_sphere", "rms"], default="unit_sphere")
    parser.add_argument("--sigma-mode", choices=["local", "global"], default="local")
    parser.add_argument("--density-alpha", type=float, default=1.0)
    parser.add_argument("--laplacian", choices=["symmetric", "unnormalized", "random_walk"], default="symmetric")
    parser.add_argument("--max-points", type=int, default=None, help="Optional deterministic point subsample per shape.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest_path, rows = read_rows(dataset_dir)
    classes = sorted({row["class_name"] for row in rows})
    output_dir = args.output_dir or dataset_dir / "svm_eigendecomposition_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "features.npz"
    config = {
        "knn_k": args.knn_k,
        "eigenvectors": args.eigenvectors,
        "normalize": args.normalize,
        "sigma_mode": args.sigma_mode,
        "density_alpha": args.density_alpha,
        "laplacian": args.laplacian,
        "max_points": -1 if args.max_points is None else args.max_points,
    }

    print(f"Dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Classes: {classes}")
    print(
        "Graph: "
        f"k={args.knn_k}, K={args.eigenvectors}, normalize={args.normalize}, "
        f"sigma={args.sigma_mode}, alpha={args.density_alpha}, laplacian={args.laplacian}"
    )

    if not args.recompute and cache_is_fresh(cache_path, manifest_path, rows, dataset_dir, config):
        cached = np.load(cache_path, allow_pickle=True)
        X_train = cached["X_train"]
        y_train = cached["y_train"]
        X_test = cached["X_test"]
        y_test = cached["y_test"]
        feature_names = [str(value) for value in cached["feature_names"]]
        print(f"Loaded cached features: {cache_path}")
    else:
        X_train, y_train, X_test, y_test, feature_names, eigenvalue_rows = build_features(rows, dataset_dir, classes, args)
        np.savez(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            classes=np.array(classes),
            feature_names=np.array(feature_names),
            eigenvalues=np.vstack(eigenvalue_rows),
            feature_version=FEATURE_VERSION,
            **config,
        )
        print(f"Saved features: {cache_path}")

    scaler, clf, best_params, best_cv, test_acc, report, cm = train_and_eval(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
    )

    results_path = output_dir / "results.txt"
    results_path.write_text(
        f"Classes: {classes}\n"
        f"Train shape: {X_train.shape}\n"
        f"Test shape: {X_test.shape}\n"
        f"Feature version: {FEATURE_VERSION}\n"
        f"Feature count: {len(feature_names)}\n"
        f"Graph k: {args.knn_k}\n"
        f"Eigenvectors: {args.eigenvectors}\n"
        f"Normalization: {args.normalize}\n"
        f"Sigma mode: {args.sigma_mode}\n"
        f"Density correction alpha: {args.density_alpha}\n"
        f"Laplacian: {args.laplacian}\n"
        f"Best params: {best_params}\n"
        f"Best CV acc: {best_cv:.3f}\n"
        f"Test acc: {test_acc:.3f}\n\n"
        "=== Classification Report ===\n"
        + report
        + "\nConfusion Matrix:\n"
        + str(cm)
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "svm": clf,
                "scaler": scaler,
                "classes": classes,
                "best_params": best_params,
                "best_cv_accuracy": best_cv,
                "test_accuracy": test_acc,
                "feature_names": feature_names,
                "feature_version": FEATURE_VERSION,
                "config": config,
            },
            handle,
        )

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Best params: {best_params}")
    print(f"Best CV acc: {best_cv:.3f}")
    print(f"Test acc   : {test_acc:.3f}")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
