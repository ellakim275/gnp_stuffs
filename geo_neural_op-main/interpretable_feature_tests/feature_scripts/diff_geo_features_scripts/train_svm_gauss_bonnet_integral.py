"""
train_svm_gauss_bonnet_integral.py
----------------------------------
Train/test an SVM using only the area-weighted Gaussian curvature integral
computed by compute_gauss_bonnet_dataset.py.
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
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from path_config import roots_for

ROOTS = roots_for(__file__)
DEFAULT_DATASET_DIR = ROOTS["OUTPUT_ROOT"] / "gaussian" / "datasets" / "only_smoothmeshes_gaussian_dataset"
FEATURE_NAME = "gaussian_curvature_integral"
LEGACY_FEATURE_NAMES = ("total_gaussian_curvature",)


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")
    columns = set(rows[0])
    required = {"class_name", "split"}
    missing = required - columns
    if missing:
        raise ValueError(f"{results_csv} is missing columns: {sorted(missing)}")
    if FEATURE_NAME not in columns:
        legacy_name = next((name for name in LEGACY_FEATURE_NAMES if name in columns), None)
        if legacy_name is None:
            raise ValueError(f"{results_csv} is missing columns: {[FEATURE_NAME]}")
        for row in rows:
            row[FEATURE_NAME] = row[legacy_name]
    return rows


def build_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    classes = sorted({row["class_name"] for row in rows})
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    X_train, y_train, X_test, y_test = [], [], [], []

    for row in rows:
        feature = [float(row[FEATURE_NAME])]
        label = class_to_idx[row["class_name"]]
        if row["split"] == "train":
            X_train.append(feature)
            y_train.append(label)
        elif row["split"] == "test":
            X_test.append(feature)
            y_test.append(label)
        else:
            raise ValueError(f"Unknown split {row['split']!r}")

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.int64),
        classes,
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
    return (
        scaler,
        clf,
        grid.best_params_,
        float(grid.best_score_),
        float(accuracy_score(y_test, y_pred)),
        report,
        cm,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM using the Gaussian curvature integral feature.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    results_csv = args.results_csv or dataset_dir / "gauss_bonnet_results.csv"
    output_dir = args.output_dir or dataset_dir / "svm_gaussian_curvature_integral_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(results_csv)
    X_train, y_train, X_test, y_test, classes = build_arrays(rows)
    scaler, clf, best_params, best_cv, test_acc, report, cm = train_and_eval(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
    )

    results_path = output_dir / "results.txt"
    results_path.write_text(
        f"Dataset: {dataset_dir}\n"
        f"Source CSV: {results_csv}\n"
        f"Classes: {classes}\n"
        f"Train shape: {X_train.shape}\n"
        f"Test shape: {X_test.shape}\n"
        f"Feature signals: {FEATURE_NAME}\n"
        f"Feature aggregation: area-weighted sum(K_i dA_i), from gauss_bonnet_results.csv {FEATURE_NAME}\n"
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
                "feature_signals": [FEATURE_NAME],
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
