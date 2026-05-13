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
        First-form signals E, F, G and second-form signals L, M, N.
    """
    return {
        "E": metric[:, 0, 0],
        "F": metric[:, 0, 1],
        "G": metric[:, 1, 1],
        "L": shape[:, 0, 0],
        "M": 0.5 * (shape[:, 0, 1] + shape[:, 1, 0]),
        "N": shape[:, 1, 1],
    }


def fundamental_form_feature_vector(metric: torch.Tensor,
                                    shape: torch.Tensor) -> torch.Tensor:
    """
    Return invariant curvature distribution features for a whole shape.

    Instead of averaging the frame-dependent fundamental-form coefficients,
    this derives intrinsic curvature signals from the first and second forms:
        K  = det(II) / det(I)
        H  = 0.5 * trace(I^-1 II)
        k1 = H + sqrt(H^2 - K)
        k2 = H - sqrt(H^2 - K)

    Each signal is summarized by five distribution statistics:
        mean, std, skewness, excess kurtosis, IQR.

    The final feature is the discrete total Gaussian curvature, sum(K).

    The output order is:
        K_mean, K_std, K_skewness, K_kurtosis, K_iqr,
        H_mean, H_std, H_skewness, H_kurtosis, H_iqr,
        k1_mean, k1_std, k1_skewness, k1_kurtosis, k1_iqr,
        k2_mean, k2_std, k2_skewness, k2_kurtosis, k2_iqr,
        total_gaussian_curvature
    """
    signals = fundamental_form_coefficients(metric, shape)
    first_form_det = signals["E"] * signals["G"] - signals["F"] * signals["F"]
    second_form_det = signals["L"] * signals["N"] - signals["M"] * signals["M"]
    det_sign = torch.where(first_form_det < 0, -1.0, 1.0)
    safe_det = det_sign * first_form_det.abs().clamp(min=1e-6)

    gaussian_curvature = second_form_det / safe_det
    mean_curvature = 0.5 * (
        signals["G"] * signals["L"]
        - 2.0 * signals["F"] * signals["M"]
        + signals["E"] * signals["N"]
    ) / safe_det
    discriminant = (mean_curvature * mean_curvature - gaussian_curvature).clamp(min=0.0)
    root = torch.sqrt(discriminant)
    principal_1 = mean_curvature + root
    principal_2 = mean_curvature - root

    curvature_signals = [gaussian_curvature, mean_curvature, principal_1, principal_2]
    feature_parts = [_distribution_stats(signal.float()) for signal in curvature_signals]
    total_gaussian_curvature = _winsorized_values(gaussian_curvature.float()).sum().view(1)
    return torch.cat(feature_parts + [total_gaussian_curvature])


def _distribution_stats(signal: torch.Tensor) -> torch.Tensor:
    """
    Summarize a per-point scalar field with five stable distribution statistics.
    """
    values = _winsorized_values(signal)
    if values.numel() == 0:
        return signal.new_zeros(5)

    mean = values.mean()
    centered = values - mean
    std = torch.sqrt((centered * centered).mean()).clamp(min=1e-8)
    skewness = (centered.pow(3).mean()) / std.pow(3)
    kurtosis = (centered.pow(4).mean()) / std.pow(4) - 3.0
    q25, q75 = torch.quantile(values, torch.tensor([0.25, 0.75], device=values.device))
    stats = torch.stack([mean, std, skewness, kurtosis, q75 - q25])
    return torch.nan_to_num(stats, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)


def _winsorized_values(signal: torch.Tensor) -> torch.Tensor:
    """
    Drop non-finite values and cap singular-estimate tails before aggregation.
    """
    values = signal[torch.isfinite(signal)]
    if values.numel() == 0:
        return values
    if values.numel() < 4:
        return torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    q01, q99 = torch.quantile(values, torch.tensor([0.01, 0.99], device=values.device))
    return values.clamp(q01, q99).clamp(-1e6, 1e6)

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
