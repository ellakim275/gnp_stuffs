import torch
import numpy as np


def _as_1d_tensor(values, device=None) -> torch.Tensor:
    """Convert a curvature/weight input to a flat float tensor."""
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.float().reshape(-1)


def estimate_point_area_weights(
    xyz: torch.Tensor,
    k: int = 16,
) -> torch.Tensor:
    """
    Estimate one local surface-area weight per point from k-nearest neighbors.

    The GNP gives pointwise Gaussian curvature K_i. To approximate the
    Gauss-Bonnet integral, sum K_i dA_i. For a point cloud without mesh faces,
    dA_i is estimated as the area of the tangent disk reaching the kth neighbor,
    divided across the k local samples.

    Parameters
    ----------
    xyz : (N, 3) tensor
        Point cloud coordinates.
    k : int
        Number of neighbors used for the local area estimate.

    Returns
    -------
    area_weights : (N,) tensor
        Approximate surface-area contribution for each point.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape (N, 3).")

    n_points = xyz.shape[0]
    if n_points == 0:
        raise ValueError("xyz must contain at least one point.")
    if n_points == 1:
        return torch.ones(1, dtype=xyz.dtype, device=xyz.device)

    n_neighbors = min(max(int(k), 1), n_points - 1)
    distances = torch.cdist(xyz.float(), xyz.float())

    # Include self at distance 0, so kth neighbor is index n_neighbors.
    kth_distances = torch.topk(
        distances,
        k=n_neighbors + 1,
        largest=False,
        dim=1,
    ).values[:, -1]

    return (np.pi * kth_distances.square() / n_neighbors).to(xyz.device)


def total_gaussian_curvature(
    gaussian_curvature,
    area_weights=None,
    xyz: torch.Tensor | None = None,
    area_k: int = 16,
) -> torch.Tensor:
    """
    Approximate the total Gaussian curvature integral, int_M K dA.

    If explicit per-point area weights are unavailable, pass ``xyz`` and this
    function will estimate weights from local point spacing.
    """
    device = xyz.device if isinstance(xyz, torch.Tensor) else None
    gaussian_curvature = _as_1d_tensor(gaussian_curvature, device=device)

    if area_weights is None:
        if xyz is None:
            area_weights = torch.ones_like(gaussian_curvature)
        else:
            area_weights = estimate_point_area_weights(xyz, k=area_k)
    else:
        area_weights = _as_1d_tensor(area_weights, device=gaussian_curvature.device)

    if area_weights.shape != gaussian_curvature.shape:
        raise ValueError(
            "area_weights and gaussian_curvature must have the same number of points."
        )

    return torch.sum(gaussian_curvature * area_weights)


def euler_characteristic_from_gauss_bonnet(
    gaussian_curvature,
    area_weights=None,
    xyz: torch.Tensor | None = None,
    area_k: int = 16,
) -> torch.Tensor:
    """
    Estimate Euler characteristic using Gauss-Bonnet: chi = int_M K dA / 2pi.

    This assumes the sampled surface is closed and orientable. For open surfaces,
    Gauss-Bonnet also has boundary terms, which are not estimated here.
    """
    total_curvature = total_gaussian_curvature(
        gaussian_curvature=gaussian_curvature,
        area_weights=area_weights,
        xyz=xyz,
        area_k=area_k,
    )
    return total_curvature / (2.0 * np.pi)


def compute_gauss_bonnet_features(
    xyz: torch.Tensor,
    curvatures: dict | torch.Tensor,
    area_weights=None,
    area_k: int = 16,
    return_total_curvature: bool = False,
) -> torch.Tensor:
    """
    Compute the single Gauss-Bonnet feature for one input point cloud.

    Parameters
    ----------
    xyz : (N, 3) tensor
        Point cloud coordinates used to estimate local area weights when
        explicit ``area_weights`` are not provided.
    curvatures : dict or (N,) tensor
        Either the GNP output dict containing ``'gaussian_curvature'`` or the
        Gaussian curvature tensor itself.
    area_weights : optional (N,) tensor
        Per-point surface area weights. Use this when mesh-derived areas are
        available; otherwise they are estimated from the point cloud.
    area_k : int
        Neighbor count for point-cloud area estimation.
    return_total_curvature : bool
        If False, return Euler characteristic estimate chi. If True, return
        total Gaussian curvature int_M K dA.

    Returns
    -------
    feature : (1,) tensor
        One scalar feature for this point cloud.
    """
    if isinstance(curvatures, dict):
        if "gaussian_curvature" not in curvatures:
            raise KeyError("curvatures must contain 'gaussian_curvature'.")
        gaussian_curvature = curvatures["gaussian_curvature"]
    else:
        gaussian_curvature = curvatures

    if return_total_curvature:
        feature = total_gaussian_curvature(
            gaussian_curvature=gaussian_curvature,
            area_weights=area_weights,
            xyz=xyz,
            area_k=area_k,
        )
    else:
        feature = euler_characteristic_from_gauss_bonnet(
            gaussian_curvature=gaussian_curvature,
            area_weights=area_weights,
            xyz=xyz,
            area_k=area_k,
        )

    return feature.reshape(1)
