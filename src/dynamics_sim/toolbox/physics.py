"""Force models used by the particle simulation."""

from dataclasses import dataclass

import torch


@dataclass
class DynamicsParams:
    friction: float
    attract_strength: float
    attract_range: float
    repel_strength: float
    repel_range: float
    external_strength: float
    external_bandwidth: float


def compute_pairwise_force(positions: torch.Tensor, params: DynamicsParams) -> torch.Tensor:
    """Compute attraction/repulsion forces for all particle pairs."""
    delta = positions[:, None, :] - positions[None, :, :]
    dist2 = (delta * delta).sum(dim=-1) + 1e-6
    dist = torch.sqrt(dist2)
    inv_dist = torch.rsqrt(dist2)
    direction = delta * inv_dist.unsqueeze(-1)

    attract_kernel = params.attract_strength * torch.exp(-dist / max(params.attract_range, 1e-3))
    repel_kernel = params.repel_strength * torch.exp(-dist2 / max(params.repel_range**2, 1e-3))
    interaction = repel_kernel - attract_kernel

    eye = torch.eye(positions.size(0), device=positions.device, dtype=positions.dtype)
    interaction = interaction * (1.0 - eye)

    return (interaction.unsqueeze(-1) * direction).sum(dim=1)


def compute_external_force(
    positions: torch.Tensor,
    centers: torch.Tensor,
    center_signs: torch.Tensor,
    params: DynamicsParams,
) -> torch.Tensor:
    """Compute radial Gaussian external forces from user-controlled centers."""
    bandwidth = max(params.external_bandwidth, 1e-3)
    delta = positions[:, None, :] - centers[None, :, :]
    dist2 = (delta * delta).sum(dim=-1)
    dist = torch.sqrt(dist2 + 1e-6)
    weight = torch.exp(-dist2 / (2.0 * bandwidth * bandwidth))

    # Use a radial Gaussian profile: force magnitude decays with distance, direction is
    # center-seeking for +1 and center-repelling for -1.
    direction_to_center = -delta / dist.unsqueeze(-1)
    signed_gain = center_signs * params.external_strength
    center_forces = signed_gain[None, :, None] * weight[:, :, None] * direction_to_center
    return center_forces.sum(dim=1)
