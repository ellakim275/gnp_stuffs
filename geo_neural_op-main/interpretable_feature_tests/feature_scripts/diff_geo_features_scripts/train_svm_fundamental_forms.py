"""
train_svm_fundamental_forms.py
------------------------------
Train/test an SVM using GNP first and second fundamental form coefficients.

For each point cloud, GNP estimates:
    metric = [[E, F], [F, G]]
    shape  = [[L, M], [M, N]]

The script trains on 21 shape-level features by default: distribution
statistics of the invariant curvature signals K, H, k1, k2, plus the
Gaussian curvature integral. It can also train on only that integral.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from path_config import roots_for

ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
TEST_ROOT = ROOTS["TEST_ROOT"]
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "modelnet10_smooth_shapes"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOTS["FEATURE_SCRIPTS_ROOT"]))

from features import fundamental_form_feature_vector
from gnp.estimator import GeometryEstimator

CURVATURE_SIGNALS = ["K", "H", "k1", "k2"]
CURVATURE_STATS = ["mean", "std", "skewness", "kurtosis", "iqr"]
FEATURE_SIGNALS = [
    f"{signal}_{stat}"
    for signal in CURVATURE_SIGNALS
    for stat in CURVATURE_STATS
] + ["gaussian_curvature_integral"]


def load_points(csv_path: Path, max_points: int | None) -> np.ndarray:
    points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[:, :3]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[indices]
    return points


def estimate_metric_and_shape(points: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    xyz_t = torch.tensor(points, dtype=torch.float32, device=device)
    normals_t = torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device=device)
    estimator = GeometryEstimator(xyz_t, orientation=normals_t, model="clean_30k", device=device)
    output = estimator.estimate_quantities(["metric", "shape"])
    return xyz_t, output["metric"], output["shape"]


def select_feature_mode(feature: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "all":
        return feature
    if feature_mode == "gaussian_curvature_integral":
        return feature[-1:].copy()
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def feature_signals_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == "all":
        return FEATURE_SIGNALS
    if feature_mode == "gaussian_curvature_integral":
        return ["gaussian_curvature_integral"]
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def process_points(points: np.ndarray, device: torch.device, feature_mode: str, area_k: int) -> np.ndarray:
    xyz_t, metric, shape = estimate_metric_and_shape(points, device)
    feature = fundamental_form_feature_vector(metric, shape, xyz_t, area_k=area_k).detach().cpu().numpy().astype(np.float32)
    return select_feature_mode(feature, feature_mode)


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


def cache_is_fresh(
    cache_path: Path,
    manifest_path: Path,
    rows: list[dict[str, str]],
    dataset_dir: Path,
    feature_signals: list[str],
    area_k: int,
) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = np.load(cache_path, allow_pickle=True)
        train_width = cached["X_train"].shape[1]
        test_width = cached["X_test"].shape[1]
        if train_width != test_width or train_width != len(feature_signals):
            return False
        if "feature_version" not in cached.files or str(cached["feature_version"]) != "gaussian_curvature_integral_v1":
            return False
        if "area_k" not in cached.files or int(cached["area_k"]) != int(area_k):
            return False
        if "feature_signals" not in cached.files:
            return False
        cached_signals = [str(value) for value in cached["feature_signals"]]
        if cached_signals != feature_signals:
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
    device: torch.device,
    max_points: int | None,
    feature_mode: str,
    area_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    X_train, y_train, X_test, y_test = [], [], [], []

    for row in rows:
        split = row["split"]
        class_name = row["class_name"]
        csv_path = resolve_csv_path(dataset_dir, row)
        print(f"  [{split}] {class_name:8s} {csv_path.name}", flush=True)
        feature = process_points(
            load_points(csv_path, max_points=max_points),
            device,
            feature_mode=feature_mode,
            area_k=area_k,
        )
        if split == "train":
            X_train.append(feature)
            y_train.append(class_to_idx[class_name])
        elif split == "test":
            X_test.append(feature)
            y_test.append(class_to_idx[class_name])
        else:
            raise ValueError(f"Unknown split {split!r} in {csv_path}")

    return np.vstack(X_train), np.array(y_train), np.vstack(X_test), np.array(y_test)


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
    parser = argparse.ArgumentParser(description="Train an SVM using GNP fundamental-form features.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-points", type=int, default=None, help="Optional deterministic point subsample per shape.")
    parser.add_argument(
        "--feature-mode",
        choices=["all", "gaussian_curvature_integral"],
        default="all",
        help="Use all fundamental-form summary features or only the Gaussian curvature integral.",
    )
    parser.add_argument("--area-k", type=int, default=16, help="Neighbor count for local point-area weights.")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest_path, rows = read_rows(dataset_dir)
    classes = sorted({row["class_name"] for row in rows})
    output_dir = args.output_dir or dataset_dir / "svm_fundamental_form_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_signals = feature_signals_for_mode(args.feature_mode)
    cache_name = "features.npz" if args.feature_mode == "all" else f"features_{args.feature_mode}.npz"
    cache_path = output_dir / cache_name
    device = torch.device(args.device)

    print(f"Dataset: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Device : {device}")
    print(f"Classes: {classes}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Area k: {args.area_k}")
    print("Features: " + ", ".join(feature_signals))

    if not args.recompute and cache_is_fresh(cache_path, manifest_path, rows, dataset_dir, feature_signals, args.area_k):
        cached = np.load(cache_path, allow_pickle=True)
        X_train = cached["X_train"]
        y_train = cached["y_train"]
        X_test = cached["X_test"]
        y_test = cached["y_test"]
        print(f"Loaded cached features: {cache_path}")
    else:
        X_train, y_train, X_test, y_test = build_features(
            rows=rows,
            dataset_dir=dataset_dir,
            classes=classes,
            device=device,
            max_points=args.max_points,
            feature_mode=args.feature_mode,
            area_k=args.area_k,
        )
        np.savez(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            classes=np.array(classes),
            feature_signals=np.array(feature_signals),
            feature_mode=args.feature_mode,
            feature_version="gaussian_curvature_integral_v1",
            area_k=args.area_k,
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
        f"Feature mode: {args.feature_mode}\n"
        f"Feature signals: {', '.join(feature_signals)}\n"
        f"Feature aggregation: {'Gaussian curvature integral only' if args.feature_mode == 'gaussian_curvature_integral' else 'mean/std/skewness/kurtosis/IQR of K, H, k1, k2, plus Gaussian curvature integral'}\n"
        f"Area weighting: local tangent-disk estimate with area_k={args.area_k}\n"
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
                "feature_mode": args.feature_mode,
                "feature_signals": feature_signals,
                "feature_version": "gaussian_curvature_integral_v1",
                "area_k": args.area_k,
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
