"""
train_svm_gaussian_dataset.py
-----------------------------
Train an SVM on the generated Gaussian-smoothed deformation dataset.

Expected input folder:
    output/gaussian_augmented_dataset/

Expected files:
    <class_name>_variantXX.csv

Outputs:
    output/gaussian_augmented_dataset/svm_gaussian_features.npz
    output/gaussian_augmented_dataset/svm_gaussian_model.pkl
    output/gaussian_augmented_dataset/svm_gaussian_results.txt
    output/gaussian_augmented_dataset/svm_gaussian_pca_plot.png
"""

from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output" / "gaussian_augmented_dataset"
SPLIT_MANIFEST = OUTPUT_DIR / "train_test_split.csv"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from point_sampling.curvature import estimate_curvatures
from features import extract_features


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANT_RE = re.compile(r"^(?P<class_name>[a-zA-Z0-9_]+)_variant(?P<idx>\d+)\.csv$")

TRAIN_PER_CLASS = 24
TEST_PER_CLASS = 6
INDICATOR_BINS = 4
VOXEL_BINS = 3
N_FOURIER = 64
FEATURES_CACHE = OUTPUT_DIR / "svm_gaussian_features.npz"


def process_points(points: np.ndarray) -> np.ndarray:
    xyz_t, curvatures = estimate_curvatures(points, DEVICE)
    feats = extract_features(
        xyz_t,
        curvatures,
        indicator_bins=INDICATOR_BINS,
        voxel_bins=VOXEL_BINS,
        n_fourier=N_FOURIER,
    )
    curvature_vec = feats["combined"]

    ones = torch.ones(xyz_t.shape[0], device=xyz_t.device)
    density_feats = extract_features(
        xyz_t,
        {"density": ones},
        indicator_bins=INDICATOR_BINS,
        voxel_bins=VOXEL_BINS,
        n_fourier=N_FOURIER,
    )
    return torch.cat([curvature_vec, density_feats["combined"]]).cpu().numpy()


def load_dataset_files() -> tuple[list[str], dict[str, list[Path]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for csv_path in sorted(OUTPUT_DIR.glob("*.csv")):
        match = VARIANT_RE.match(csv_path.name)
        if not match:
            continue
        grouped[match.group("class_name")].append(csv_path)

    classes = sorted(grouped.keys())
    if not classes:
        raise FileNotFoundError(f"No variant CSV files found in {OUTPUT_DIR}")
    return classes, grouped


def cache_is_fresh(cache_path: Path, grouped: dict[str, list[Path]]) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for files in grouped.values():
        for csv_path in files:
            if csv_path.stat().st_mtime > cache_mtime:
                return False
    return True


def build_split(
    classes: list[str],
    grouped: dict[str, list[Path]],
    train_per_class: int,
    test_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = [], [], [], []

    for cls_idx, cls_name in enumerate(classes):
        files = sorted(grouped[cls_name])
        needed = train_per_class + test_per_class
        if len(files) < needed:
            raise ValueError(
                f"Class {cls_name} has only {len(files)} files, need at least {needed}"
            )

        train_files = files[:train_per_class]
        test_files = files[train_per_class:train_per_class + test_per_class]

        for split_name, split_files, X_out, y_out in [
            ("train", train_files, X_train, y_train),
            ("test", test_files, X_test, y_test),
        ]:
            for csv_path in split_files:
                print(f"  [{split_name}] {cls_name} | {csv_path.name}", flush=True)
                points = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                feats = process_points(points)
                X_out.append(feats)
                y_out.append(cls_idx)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
    )


def main() -> None:
    classes, grouped = load_dataset_files()
    print(f"Device  : {DEVICE}")
    print(f"Classes : {classes}")
    print(f"Dataset : {OUTPUT_DIR}")

    if cache_is_fresh(FEATURES_CACHE, grouped):
        cached = np.load(FEATURES_CACHE, allow_pickle=True)
        X_train = cached["X_train"]
        y_train = cached["y_train"]
        X_test = cached["X_test"]
        y_test = cached["y_test"]
        cached_classes = list(cached["classes"])
        if cached_classes != classes:
            raise ValueError(
                f"Cached classes {cached_classes} do not match dataset classes {classes}"
            )
        print(f"\nLoaded cached features from {FEATURES_CACHE}")
        print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    else:
        print("\n=== Building training set ===")
        X_train, y_train, X_test, y_test = build_split(
            classes,
            grouped,
            train_per_class=TRAIN_PER_CLASS,
            test_per_class=TEST_PER_CLASS,
        )
        print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

        np.savez(
            str(FEATURES_CACHE),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            classes=np.array(classes),
        )
        print(f"Saved: {FEATURES_CACHE}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = [
        {"kernel": ["linear"], "C": [0.1, 1.0, 10.0, 100.0]},
        {
            "kernel": ["poly"],
            "C": [0.1, 1.0, 10.0, 100.0],
            "degree": [2, 3, 4],
            "gamma": [1e-4, 1e-3, 1e-2, "scale", "auto"],
            "coef0": [0.0, 1.0],
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": [1e-4, 1e-3, 1e-2, "scale", "auto"],
        },
        {
            "kernel": ["sigmoid"],
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": [1e-4, 1e-3, 1e-2, "scale", "auto"],
            "coef0": [0.0, 1.0],
        },
    ]
    base_clf = SVC(decision_function_shape="ovr", random_state=0)
    grid = GridSearchCV(base_clf, param_grid, cv=5, scoring="accuracy", n_jobs=1)
    grid.fit(X_train_s, y_train)
    clf = grid.best_estimator_

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV acc: {grid.best_score_:.3f}")

    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=classes)
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Classification Report ===")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    out_txt = OUTPUT_DIR / "svm_gaussian_results.txt"
    out_txt.write_text(
        f"Best params: {grid.best_params_}\n"
        f"Best CV acc: {grid.best_score_:.3f}\n\n"
        "=== Classification Report ===\n"
        + report
        + "\nConfusion Matrix:\n"
        + str(cm)
        + "\n"
    )
    print(f"Saved: {out_txt}")

    out_pkl = OUTPUT_DIR / "svm_gaussian_model.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "svm": clf,
                "scaler": scaler,
                "classes": classes,
                "best_params": grid.best_params_,
                "best_cv_accuracy": grid.best_score_,
            },
            f,
        )
    print(f"Saved: {out_pkl}")

    import plotly.graph_objects as go
    import plotly.express as px

    n_tr = len(X_train)
    X_all_s = np.vstack([X_train_s, X_test_s])
    pca = PCA(n_components=3, random_state=0)
    X_3d = pca.fit_transform(X_all_s)

    colors = px.colors.qualitative.Plotly  # 10-color palette, extend if needed

    fig = go.Figure()

    for ci, cls_name in enumerate(classes):
        col = colors[ci % len(colors)]  # same color for both traces
        tr_mask = y_train == ci
        te_mask = y_test == ci

        tr = X_3d[:n_tr][tr_mask]
        te = X_3d[n_tr:][te_mask]

        fig.add_trace(go.Scatter3d(
            x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
            mode="markers",
            marker=dict(size=5, color=col, symbol="circle", line=dict(width=0.5, color="black")),
            name=f"{cls_name} train",
            legendgroup=cls_name,
        ))
        fig.add_trace(go.Scatter3d(
            x=te[:, 0], y=te[:, 1], z=te[:, 2],
            mode="markers",
            marker=dict(size=8, color=col, symbol="cross", line=dict(width=0.5, color="black")),
            name=f"{cls_name} test",
            legendgroup=cls_name,
            showlegend=True,
        ))

    fig.update_layout(
        title="Gaussian Augmented Dataset — PCA projection",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f} %)",
        ),
        legend=dict(groupclick="toggleitem"),
        width=900, height=700,
    )

    fig.show()                                     
    


if __name__ == "__main__":
    main()
