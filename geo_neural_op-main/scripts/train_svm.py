"""
train_svm.py
------------
End-to-end shape classification pipeline using GNP curvature features + SVM.

Pipeline per mesh:
  1. Sample mesh surface to a point cloud (Open3D Poisson disk sampling)
  2. Smooth the point cloud with KNN Laplacian diffusion
  3. Estimate normals and normalize coordinates
  4. Estimate mean & Gaussian curvature via a pre-trained GNP model
  5. Project curvature signals onto indicator and random Fourier kernels
     to obtain a fixed-length feature vector

Classifier:
  sklearn SVC with RBF kernel, trained on feature vectors from 4 ModelNet10
  classes (bathtub, chair, sofa, monitor), 5 training + 3 test meshes each.

Outputs (written to output/ in the repo root):
  svm_features.npz    — raw feature arrays for later inspection
  svm_model.pkl       — trained SVM + StandardScaler + class names
  svm_results.txt     — sklearn classification report + confusion matrix
  feature_pca_plot.png — 2D PCA scatter of the feature space
"""

import sys
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import GridSearchCV

import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')          # headless — no display required
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
OUTPUT_DIR  = REPO_ROOT / 'output'
sys.path.insert(0, str(SCRIPT_DIR))   # for features.py (same dir)
sys.path.insert(0, str(REPO_ROOT))    # for gnp package

from features import extract_features
from output_CSV import sample_mesh_to_points
from knn import laplacian_smooth
from curvature import estimate_curvatures

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSES           = ['bathtub', 'chair', 'toilet', 'desk']
MODELNET_ROOT     = REPO_ROOT.parent / 'ModelNet10'
N_TRAIN_PER_CLASS = 30
N_TEST_PER_CLASS  = 5
N_POINTS_SAMPLE   = 5000
SMOOTH_ITERATIONS = 1
SMOOTH_LAM        = 0.5
KNN_K             = 12
INDICATOR_BINS    = 1       
VOXEL_BINS        = 1       
N_FOURIER         = 64     

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def process_mesh(mesh_path):
    pts = sample_mesh_to_points(mesh_path, N_POINTS_SAMPLE)
    pts = laplacian_smooth(pts, KNN_K, SMOOTH_ITERATIONS, SMOOTH_LAM)
    xyz_t, curvatures = estimate_curvatures(pts, DEVICE)
    feats = extract_features(xyz_t, curvatures,
                             indicator_bins=INDICATOR_BINS,
                             voxel_bins=VOXEL_BINS,
                             n_fourier=N_FOURIER)
    curvature_vec = feats['combined']

    ones = torch.ones(xyz_t.shape[0], device=xyz_t.device)
    density_feats = extract_features(xyz_t, {'density': ones},
                                     indicator_bins=INDICATOR_BINS,
                                     voxel_bins=VOXEL_BINS,
                                     n_fourier=N_FOURIER)
    return torch.cat([curvature_vec, density_feats['combined']]).cpu().numpy()


def build_dataset(split, n_per_class):
    X, y, labels = [], [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = MODELNET_ROOT / cls_name / split
        mesh_files = sorted(cls_dir.glob('*.off'))[:n_per_class]
        if not mesh_files:
            warnings.warn(f"No .off files found in {cls_dir}")
            continue
        for mesh_path in mesh_files:
            print(f"  [{split}] {cls_name} | {mesh_path.name}", flush=True)
            try:
                fv = process_mesh(mesh_path)
                X.append(fv)
                y.append(cls_idx)
                labels.append(cls_name)
            except Exception as exc:
                warnings.warn(f"  Skipped {mesh_path.name}: {exc}")
    return np.array(X), np.array(y), labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device  : {DEVICE}")
    print(f"Classes : {CLASSES}")
    print(f"ModelNet: {MODELNET_ROOT}")
    feat_dim = (3 * INDICATOR_BINS + VOXEL_BINS ** 3 + N_FOURIER) * 2
    print(f"Feature dims per shape: {feat_dim}\n")

    print("=== Building training set ===")
    X_train, y_train, _ = build_dataset('train', N_TRAIN_PER_CLASS)
    print("\n=== Building test set ===")
    X_test, y_test, _ = build_dataset('test', N_TEST_PER_CLASS)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    out_npz = OUTPUT_DIR / 'svm_features.npz'
    np.savez(str(out_npz), X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test, classes=np.array(CLASSES))
    print(f"Saved: {out_npz}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n=== Training SVM (RBF kernel) ===")
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [1e-4, 1e-3, 1e-2, 'scale', 'auto']}
    base_clf = SVC(kernel='rbf', decision_function_shape='ovr', random_state=0)
    grid = GridSearchCV(base_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_s, y_train)
    clf = grid.best_estimator_
    print(f"  Best params : {grid.best_params_}")
    print(f"  CV accuracy : {grid.best_score_:.3f}")


    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    cm     = confusion_matrix(y_test, y_pred)
    print("\n=== Classification Report ===")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    out_txt = OUTPUT_DIR / 'svm_results.txt'
    out_txt.write_text("=== Classification Report ===\n" + report +
                       "\n\nConfusion Matrix:\n" + str(cm) + "\n")
    print(f"Saved: {out_txt}")

    out_pkl = OUTPUT_DIR / 'svm_model.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'svm': clf, 'scaler': scaler, 'classes': CLASSES}, f)
    print(f"Saved: {out_pkl}")

    n_tr = len(X_train)
    X_all_s = np.vstack([X_train_s, X_test_s])
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X_all_s)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, (cls_name, col) in enumerate(zip(CLASSES, colors)):
        tr_mask = y_train == ci
        te_mask = y_test  == ci
        ax.scatter(X_2d[:n_tr][tr_mask, 0], X_2d[:n_tr][tr_mask, 1],
                   c=col, marker='o', s=80, label=f'{cls_name} train',
                   edgecolors='k', linewidths=0.4)
        ax.scatter(X_2d[n_tr:][te_mask, 0], X_2d[n_tr:][te_mask, 1],
                   c=col, marker='*', s=200, edgecolors='k', linewidths=0.4)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)')
    ax.set_title('GNP Curvature Features — PCA projection\n(circles = train, stars = test)')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    out_png = OUTPUT_DIR / 'feature_pca_plot.png'
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    print(f"Saved: {out_png}")
    print("\nDone.")


if __name__ == '__main__':
    main()
