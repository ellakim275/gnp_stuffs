# DINO on Gaussian ModelNet10 Snapshots

This folder now has a DINO-ready snapshot dataset rendered from the Gaussian ModelNet10 CSV point clouds:

```text
data/only_modelnet10_gaussian_dataset/
  train/{bathtub,desk,sofa,toilet}/*.png
  val/{bathtub,desk,sofa,toilet}/*.png
```

The `val` split is rendered from the original dataset's `test` split because DINO's ImageFolder eval convention expects `train` and `val`.

The default snapshots use a flat white background with no 3D axes, panes, ticks, title, or bounding box. This avoids giving DINO a background shortcut in the attention maps.

## Refresh the Dataset Copy

```bash
python geo_neural_op-main/dino_tests/prepare_only_modelnet10_dataset.py
```

Useful rendering options:

```bash
python geo_neural_op-main/dino_tests/prepare_only_modelnet10_dataset.py \
  --background white \
  --point-size 0.18 \
  --elev 20 \
  --azim 35
```

If you ever want the old Matplotlib preview PNGs instead:

```bash
python geo_neural_op-main/dino_tests/prepare_only_modelnet10_dataset.py --copy-existing-previews
```

## Run Pretrained DINO Feature Extraction

From the repo root:

```bash
python geo_neural_op-main/dino_tests/run_only_modelnet10_dino.py
```

By default this uses pretrained `vit_small` DINO weights with patch size 16. The first run may download the weights from the official DINO release. To use a local checkpoint instead:

```bash
python geo_neural_op-main/dino_tests/run_only_modelnet10_dino.py \
  --pretrained-weights /path/to/dino_deitsmall16_pretrain.pth
```

Outputs are written to:

```text
geo_neural_op-main/dino_tests/outputs/only_modelnet10_gaussian_dataset/
  features.npz
  knn_metrics.csv
  pca_features.png
  attention/
```

Use `features.npz` for downstream probes, `knn_metrics.csv` to check whether the frozen DINO representation separates classes, `pca_features.png` for a quick visual cluster check, and `attention/` for qualitative inspection of what image regions the final-layer attention heads emphasize.

## Original DINO kNN Script

The copied data also works with the original DINO `eval_knn.py` if you want to use their distributed evaluator:

```bash
cd geo_neural_op-main/dino_tests/dino
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
  --data_path ../data/only_modelnet10_gaussian_dataset \
  --arch vit_small \
  --patch_size 16 \
  --nb_knn 1 3 5 10 20 \
  --dump_features ../outputs/original_eval_knn_features
```
