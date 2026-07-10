# Eigendecomposition Features

This folder contains a separate graph-Laplacian eigendecomposition pipeline for
point-cloud shape classification. The feature vector is only the raw graph
Laplacian eigenvalues.

Pipeline:

```text
N x 3 point cloud
center and scale normalize
k-NN graph
Gaussian kernel weights
density-corrected weights
symmetric normalized graph Laplacian
sparse eigsh, K smallest eigenpairs
eigenvalues only
SVM classifier
```

Default run:

```bash
python geo_neural_op-main/interpretable_feature_tests/eigendecomposition/train_svm_eigendecomposition.py --recompute
```

The default Laplacian is the sampling-robust symmetric normalized form:

```text
L = I - D^{-1/2} W_tilde D^{-1/2}
```

The raw unnormalized form `D - W_tilde` is available with:

```bash
--laplacian unnormalized
```

Eigenvalue barcode plots:

```bash
python geo_neural_op-main/interpretable_feature_tests/eigendecomposition/plot_feature_barcodes.py \
  --dataset-dir geo_neural_op-main/interpretable_feature_tests/output/gaussian/datasets/only_smoothmeshes_gaussian_dataset
```

Each plot uses eigenvalue feature index on the x-axis, shared x ticks, and
shared y-axis limits across the whole dataset.
