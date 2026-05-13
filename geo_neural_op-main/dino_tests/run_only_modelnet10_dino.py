#!/usr/bin/env python3
"""
Extract pretrained DINO features from the Gaussian ModelNet10 snapshot dataset.

Outputs:
  - features.npz: train/val embeddings, labels, class names, and image paths
  - knn_metrics.csv: weighted kNN accuracy for requested k values
  - pca_features.png: 2D PCA scatter of train and val embeddings
  - attention/<image_stem>/...: optional last-layer CLS attention maps
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

MPL_CACHE = Path(__file__).resolve().parent / "outputs" / ".matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
XDG_CACHE = Path(__file__).resolve().parent / "outputs" / ".cache"
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import datasets
from torchvision import transforms as pth_transforms


DINO_DIR = Path(__file__).resolve().parent / "dino"
sys.path.insert(0, str(DINO_DIR))

import utils  # noqa: E402
import vision_transformer as vits  # noqa: E402


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "only_modelnet10_gaussian_dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "only_modelnet10_gaussian_dataset"


PRETRAINED_URLS = {
    ("vit_small", 16): "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    ("vit_small", 8): "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    ("vit_base", 16): "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    ("vit_base", 8): "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
}


def build_transform(image_size: int) -> pth_transforms.Compose:
    return pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    model.eval()
    model.to(device)

    if args.pretrained_weights:
        weights_path = Path(args.pretrained_weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        url = PRETRAINED_URLS.get((args.arch, args.patch_size))
        if url is None:
            raise ValueError(
                f"No default DINO weights for arch={args.arch} patch_size={args.patch_size}. "
                "Pass --pretrained-weights."
            )
        print(f"Downloading/loading DINO weights: {url}")
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

    if args.checkpoint_key and args.checkpoint_key in state_dict:
        state_dict = state_dict[args.checkpoint_key]
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded DINO weights with msg: {msg}")
    return model


@torch.no_grad()
def extract_split_features(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images)
        feats = nn.functional.normalize(feats, dim=1, p=2)
        features.append(feats.cpu().numpy())
        labels.append(batch_labels.numpy())

    paths = [sample[0] for sample in dataset.samples]
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), paths


def weighted_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    k: int,
    temperature: float,
    num_classes: int,
) -> float:
    k = min(k, len(train_features))
    sims = val_features @ train_features.T
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    correct = 0

    for row_idx, neighbors in enumerate(topk_idx):
        votes = np.zeros(num_classes, dtype=np.float64)
        neighbor_sims = sims[row_idx, neighbors]
        weights = np.exp(neighbor_sims / temperature)
        for label, weight in zip(train_labels[neighbors], weights):
            votes[label] += weight
        correct += int(votes.argmax() == val_labels[row_idx])

    return 100.0 * correct / len(val_labels)


def write_knn_metrics(
    output_dir: Path,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    k_values: list[int],
    temperature: float,
    class_names: list[str],
) -> None:
    rows = []
    for k in k_values:
        accuracy = weighted_knn(
            train_features,
            train_labels,
            val_features,
            val_labels,
            k=k,
            temperature=temperature,
            num_classes=len(class_names),
        )
        rows.append({"k": k, "top1_accuracy": f"{accuracy:.4f}"})
        print(f"{k}-NN top-1 accuracy: {accuracy:.2f}%")

    with (output_dir / "knn_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "top1_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def plot_pca(
    output_dir: Path,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    class_names: list[str],
) -> None:
    all_features = np.vstack([train_features, val_features])
    centered = all_features - all_features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    split_names = np.array(["train"] * len(train_features) + ["val"] * len(val_features))
    labels = np.concatenate([train_labels, val_labels])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(class_names)))
    for class_idx, class_name in enumerate(class_names):
        for split, marker in [("train", "o"), ("val", "^")]:
            mask = (labels == class_idx) & (split_names == split)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=36 if split == "train" else 60,
                marker=marker,
                color=colors[class_idx],
                edgecolor="black" if split == "val" else "none",
                linewidth=0.5,
                alpha=0.85,
                label=f"{class_name} {split}",
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Pretrained DINO features on Gaussian ModelNet10 snapshots")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_features.png", dpi=180)
    plt.close(fig)


def save_preview_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=image_tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=image_tensor.dtype).view(1, 3, 1, 1)
    preview = (image_tensor.cpu() * std + mean).clamp(0, 1)
    torchvision.utils.save_image(
        preview,
        output_path,
    )


@torch.no_grad()
def save_attention_maps(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    output_dir: Path,
    image_size: int,
    patch_size: int,
    sample_count: int,
    device: torch.device,
) -> None:
    if sample_count <= 0:
        return

    attention_root = output_dir / "attention"
    attention_root.mkdir(parents=True, exist_ok=True)
    transform = build_transform(image_size)

    selected_samples = []
    seen_classes = set()
    for image_path, class_idx in dataset.samples:
        if class_idx not in seen_classes:
            selected_samples.append((image_path, class_idx))
            seen_classes.add(class_idx)
        if len(selected_samples) >= sample_count:
            break
    if len(selected_samples) < sample_count:
        selected_samples.extend(dataset.samples[: sample_count - len(selected_samples)])

    for image_path, class_idx in selected_samples:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        h = tensor.shape[1] - tensor.shape[1] % patch_size
        w = tensor.shape[2] - tensor.shape[2] % patch_size
        tensor = tensor[:, :h, :w].unsqueeze(0).to(device)

        attentions = model.get_last_selfattention(tensor)
        num_heads = attentions.shape[1]
        h_featmap = tensor.shape[-2] // patch_size
        w_featmap = tensor.shape[-1] // patch_size
        attentions = attentions[0, :, 0, 1:].reshape(num_heads, h_featmap, w_featmap)
        attentions = nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        )[0].cpu().numpy()

        sample_dir = attention_root / f"{Path(image_path).stem}__{dataset.classes[class_idx]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_preview_image(tensor, sample_dir / "image.png")
        for head_idx in range(num_heads):
            plt.imsave(sample_dir / f"attn_head_{head_idx:02d}.png", attentions[head_idx], format="png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained-weights", default="", help="Optional local DINO .pth checkpoint.")
    parser.add_argument("--checkpoint-key", default="teacher")
    parser.add_argument("--arch", default="vit_small", choices=["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--knn", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--attention-samples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.data_path / "train"
    val_dir = args.data_path / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected ImageFolder dataset with train/ and val/ under {args.data_path}. "
            "Run prepare_only_modelnet10_dataset.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = build_transform(args.image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(f"Train/val classes differ: {train_dataset.classes} vs {val_dataset.classes}")
    print(f"Loaded {len(train_dataset)} train and {len(val_dataset)} val images.")
    print(f"Classes: {', '.join(train_dataset.classes)}")

    model = load_model(args, device)
    train_features, train_labels, train_paths = extract_split_features(
        model, train_dataset, args.batch_size, args.num_workers, device
    )
    val_features, val_labels, val_paths = extract_split_features(
        model, val_dataset, args.batch_size, args.num_workers, device
    )

    np.savez_compressed(
        args.output_dir / "features.npz",
        train_features=train_features,
        train_labels=train_labels,
        train_paths=np.array(train_paths),
        val_features=val_features,
        val_labels=val_labels,
        val_paths=np.array(val_paths),
        class_names=np.array(train_dataset.classes),
    )
    write_knn_metrics(
        args.output_dir,
        train_features,
        train_labels,
        val_features,
        val_labels,
        args.knn,
        args.temperature,
        train_dataset.classes,
    )
    plot_pca(args.output_dir, train_features, train_labels, val_features, val_labels, train_dataset.classes)
    save_attention_maps(
        model,
        val_dataset,
        args.output_dir,
        args.image_size,
        args.patch_size,
        args.attention_samples,
        device,
    )
    print(f"Done. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
