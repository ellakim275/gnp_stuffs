"""
shape_features.py
-----------------
Computes shape classification feature vectors from point cloud curvature data.

Given:
    xyz       : (N, 3) tensor of surface point coordinates (normalized)
    curvatures: dict with 'mean_curvature' and/or 'gaussian_curvature' (N,) tensors

Two kernel families are implemented:

  1. IndicatorKernel  — half-space / axis-aligned slab indicators
                        k(x) = 1 if a^T x < threshold, else 0
                        Produces spatially localized averages.

  2. FourierKernel    — random truncated Fourier series
                        k(x) = sum_j alpha_j * phi_j(x)
                        where phi_j are random 3D sinusoids.
                        Produces smooth random projections.

Usage (standalone, after running curvature.py):
    features = extract_features(xyz, curvatures)
    # features['indicator'] : (n_indicator_features,)
    # features['fourier']   : (n_fourier_features,)
    # features['combined']  : concatenation of both
"""

import torch
import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# Kernel 1: Indicator / half-space kernel
# ---------------------------------------------------------------------------

def indicator_features(xyz: torch.Tensor,
                       signal: torch.Tensor,
                       n_bins: int = 4) -> torch.Tensor:
    """
    For each axis and each quantile-spaced threshold along that axis,
    compute:
        F = mean( s(x') * 1[a^T x' < t] )

    This gives the average signal value over each spatial half-space slab,
    sweeping along x1, x2, x3 independently.

    Parameters
    ----------
    xyz    : (N, 3) point coordinates, assumed normalized to [-1, 1]
    signal : (N,)  scalar surface signal (e.g. mean curvature)
    n_bins : number of threshold cuts per axis

    Returns
    -------
    features : (3 * n_bins,) tensor
    """
    device = xyz.device
    thresholds = torch.linspace(-0.75, 0.75, n_bins, device=device)

    feats = []
    for axis in range(3):
        coords = xyz[:, axis]           # (N,)
        for t in thresholds:
            mask = (coords < t).float() # 1 where x_axis < t
            weight_sum = mask.sum()
            if weight_sum > 0:
                F = (mask * signal).sum() / weight_sum
            else:
                F = torch.tensor(0.0, device=device)
            feats.append(F)

    return torch.stack(feats)           # (3 * n_bins,)


def indicator_features_3d(xyz: torch.Tensor,
                           signal: torch.Tensor,
                           n_bins: int = 3) -> torch.Tensor:
    """
    Axis-aligned 3D voxel grid version.
    Divides space into n_bins^3 voxels and computes the mean signal in each.

    Parameters
    ----------
    xyz    : (N, 3)
    signal : (N,)
    n_bins : number of divisions per axis (total voxels = n_bins^3)

    Returns
    -------
    features : (n_bins^3,) tensor  (zeros for empty voxels)
    """
    device = xyz.device
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    spans = (maxs - mins).clamp(min=1e-6)
    padding = spans * 1e-6
    edges = [
        torch.linspace(mins[axis] - padding[axis], maxs[axis] + padding[axis], n_bins + 1, device=device)
        for axis in range(3)
    ]

    feats = []
    for i, j, k in product(range(n_bins), repeat=3):
        mask = (
            (xyz[:, 0] >= edges[0][i]) & (xyz[:, 0] < edges[0][i+1]) &
            (xyz[:, 1] >= edges[1][j]) & (xyz[:, 1] < edges[1][j+1]) &
            (xyz[:, 2] >= edges[2][k]) & (xyz[:, 2] < edges[2][k+1])
        ).float()
        weight_sum = mask.sum()
        F = (mask * signal).sum() / weight_sum if weight_sum > 0 else torch.tensor(0.0, device=device)
        feats.append(F)

    return torch.stack(feats)           # (n_bins^3,)


# ---------------------------------------------------------------------------
# Kernel 2: Random truncated Fourier kernel
# ---------------------------------------------------------------------------

def fourier_features(xyz: torch.Tensor,
                     signal: torch.Tensor,
                     n_features: int = 128,
                     max_freq: float = 4.0,
                     seed: int = 42) -> torch.Tensor:
    """
    For each of n_features random smooth kernels k_r(x), compute:

        F_r = (1/N) * sum_i k_r(x_i) * s(x_i)

    where k_r(x) = sum_j alpha_j * [cos(omega_j^T x) or sin(omega_j^T x)]
    is a truncated random Fourier series in R^3.

    This is equivalent to projecting the surface signal field onto a random
    element of a Fourier RKHS — a random feature approximation of a
    translation-invariant kernel inner product.

    Parameters
    ----------
    xyz        : (N, 3) point coordinates
    signal     : (N,)  scalar surface signal
    n_features : number of random projection features
    max_freq   : maximum frequency magnitude for random Fourier modes
    seed       : random seed for reproducibility

    Returns
    -------
    features : (n_features,) tensor
    """
    device = xyz.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    n_harmonics = 16                    # Fourier terms per kernel

    # Random frequencies: (n_features, n_harmonics, 3)
    omega = (torch.rand(n_features, n_harmonics, 3,
                        generator=rng, device=device) * 2 - 1) * max_freq

    # Random amplitudes: (n_features, n_harmonics)
    alpha = torch.randn(n_features, n_harmonics, generator=rng, device=device)
    alpha = alpha / alpha.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Phase offsets (adds both sin and cos character): (n_features, n_harmonics)
    phase = torch.rand(n_features, n_harmonics,
                       generator=rng, device=device) * 2 * np.pi

    # Compute kernel values at all points
    # xyz: (N, 3) -> project onto frequencies -> (N, n_features, n_harmonics)
    proj = torch.einsum('nd, fhd -> nfh', xyz, omega)   # (N, n_features, n_harmonics)
    phi  = torch.cos(proj + phase.unsqueeze(0))          # (N, n_features, n_harmonics)

    # k_r(x_i) = sum_h alpha_rh * phi_rh(x_i)  -> (N, n_features)
    k_vals = (phi * alpha.unsqueeze(0)).sum(dim=-1)      # (N, n_features)

    # F_r = mean_i k_r(x_i) * s(x_i)
    features = (k_vals * signal.unsqueeze(1)).mean(dim=0)  # (n_features,)

    return features


# ---------------------------------------------------------------------------
# Multi-signal feature extraction
# ---------------------------------------------------------------------------

def extract_features(xyz: torch.Tensor,
                     curvatures: dict,
                     indicator_bins: int = 4,
                     voxel_bins: int = 4,
                     n_fourier: int = 128,
                     max_freq: float = 4.0,
                     seed: int = 42) -> dict:
    """
    Full feature extraction pipeline.

    Parameters
    ----------
    xyz         : (N, 3) normalized point coordinates
    curvatures  : dict with keys 'mean_curvature' and/or 'gaussian_curvature',
                  each an (N,) tensor. (Direct output from GeometryEstimator.)
    indicator_bins : threshold cuts per axis for half-space features
    voxel_bins     : divisions per axis for 3D voxel features
    n_fourier      : number of random Fourier projection features
    max_freq       : max frequency for Fourier kernels
    seed           : RNG seed (same seed = same random kernels for all signals,
                     so features are comparable across shapes)

    Returns
    -------
    features : dict with keys:
        '<signal>_indicator_halfspace' : (3 * indicator_bins,)
        '<signal>_indicator_voxel'     : (voxel_bins^3,)
        '<signal>_fourier'             : (n_fourier,)
        'combined'                     : concatenation of all the above
    """
    device = xyz.device
    all_vecs = []
    result = {}

    for signal_name, signal_raw in curvatures.items():
        signal = signal_raw.to(device).float()

        # Clamp extreme curvature values (outlier robustness)
        p5, p95 = torch.quantile(signal, torch.tensor([0.05, 0.95], device=device))
        signal_clamped = signal.clamp(p5, p95)

        # Normalize signal to zero mean, unit std
        mu, sigma = signal_clamped.mean(), signal_clamped.std().clamp(min=1e-8)
        signal_norm = (signal_clamped - mu) / sigma

        # --- Indicator features ---
        hs = indicator_features(xyz, signal_norm, n_bins=indicator_bins)
        vx = indicator_features_3d(xyz, signal_norm, n_bins=voxel_bins)

        # --- Fourier features ---
        ff = fourier_features(xyz, signal_norm,
                              n_features=n_fourier,
                              max_freq=max_freq,
                              seed=seed)

        result[f'{signal_name}_indicator_halfspace'] = hs
        result[f'{signal_name}_indicator_voxel']     = vx
        result[f'{signal_name}_fourier']             = ff

        all_vecs.extend([hs, vx, ff])

    result['combined'] = torch.cat(all_vecs)
    return result


def fundamental_form_coefficients(metric: torch.Tensor,
                                  shape: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Extract first and second fundamental form coefficient signals.

    Parameters
    ----------
    metric : (N, 2, 2) tensor
        First fundamental form from GNP SurfacePatch.metric.
    shape : (N, 2, 2) tensor
        Second fundamental form from GNP SurfacePatch.shape.

    Returns
    -------
    dict
        Seven per-point scalar signals: E, F, G, e, f1, f2, g.
    """
    return {
        "E": metric[:, 0, 0],
        "F": metric[:, 0, 1],
        "G": metric[:, 1, 1],
        "e": shape[:, 0, 0],
        "f1": shape[:, 0, 1],
        "f2": shape[:, 1, 0],
        "g": shape[:, 1, 1],
    }


def extract_fundamental_form_features(xyz: torch.Tensor,
                                      metric: torch.Tensor,
                                      shape: torch.Tensor,
                                      indicator_bins: int = 4,
                                      voxel_bins: int = 4,
                                      n_fourier: int = 128,
                                      max_freq: float = 4.0,
                                      seed: int = 42) -> dict:
    """
    Build fixed-length shape features from the seven fundamental-form signals.

    The seven raw signals are E, F, G from the first fundamental form and
    e, f1, f2, g from the second fundamental form. Each signal is passed through
    the same indicator, voxel, and random Fourier feature pipeline used by the
    curvature features, then concatenated into one SVM-ready vector.
    """
    signals = fundamental_form_coefficients(metric, shape)
    return extract_features(
        xyz=xyz,
        curvatures=signals,
        indicator_bins=indicator_bins,
        voxel_bins=voxel_bins,
        n_fourier=n_fourier,
        max_freq=max_freq,
        seed=seed,
    )


def raw_fundamental_form_feature_vector(metric: torch.Tensor,
                                        shape: torch.Tensor) -> torch.Tensor:
    """
    Return the literal seven fundamental-form features for a whole shape.

    Since E, F, G, e, f1, f2, g are per-point coefficient fields, this produces
    one fixed-length descriptor by averaging each coefficient over the point
    cloud. The output order is:
        E, F, G, e, f1, f2, g
    """
    signals = fundamental_form_coefficients(metric, shape)
    return torch.stack([signals[name].float().mean() for name in ["E", "F", "G", "e", "f1", "f2", "g"]])

if __name__ == '__main__':
    # --- Paste or import from curvature.py outputs ---
    # Assumes the following variables are already in scope from curvature.py:
    #   data['xyz']       : (N, 3) torch tensor on device
    #   mean_clean        : (N,)   torch tensor
    #   gauss_clean       : (N,)   torch tensor

    # If running standalone, mock with random data for testing:
    try:
        xyz_in = data['xyz']
        curvatures_in = {
            'mean_curvature':     mean_clean,
            'gaussian_curvature': gauss_clean,
        }
        print("Using live curvature data from curvature.py.")
    except NameError:
        print("No live data found — running with synthetic mock data.")
        N = 2048
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xyz_in = (torch.rand(N, 3, device=device) * 2 - 1)
        xyz_in = xyz_in / xyz_in.norm(dim=1, keepdim=True)   # project to sphere
        curvatures_in = {
            'mean_curvature':     torch.randn(N, device=device),
            'gaussian_curvature': torch.randn(N, device=device),
        }

    features = extract_features(
        xyz_in,
        curvatures_in,
        indicator_bins = 4,     # → 3*4 = 12 half-space features per signal
        voxel_bins     = 4,     # → 4^3 = 64 voxel features per signal
        n_fourier      = 128,   # → 128 Fourier projection features per signal
        max_freq       = 4.0,
    )

    print("\n--- Feature vector summary ---")
    total = 0
    for k, v in features.items():
        if k == 'combined':
            continue
        print(f"  {k:45s}: shape {tuple(v.shape)}")
        total += v.numel()

    print(f"\n  {'combined':45s}: shape {tuple(features['combined'].shape)}")
    print(f"  Total feature dimensions: {features['combined'].numel()}")

    # Save to disk as a numpy array for use in a classifier
    out = features['combined'].cpu().numpy()
    np.save('shape_feature_vector.npy', out)
    print("\nSaved: shape_feature_vector.npy")
