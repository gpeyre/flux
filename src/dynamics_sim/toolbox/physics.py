"""Force models used by the particle simulation."""

from dataclasses import dataclass

import torch


@dataclass
class DynamicsParams:
    friction: float
    a_a: float
    s_a: float
    a_r: float
    s_r: float
    external_strength: float
    external_bandwidth: float
    unity_normalization: bool = True
    attention_normalization: bool = False


def compute_pairwise_force(positions: torch.Tensor, params: DynamicsParams) -> torch.Tensor:
    """Compute pairwise forces with a 2-head Gaussian mixture and 1/n normalization."""
    delta = positions[:, None, :] - positions[None, :, :]
    dist2 = (delta * delta).sum(dim=-1) + 1e-6

    # v_1 = +a_r (repulsive), v_2 = -a_a (attractive)
    g_r = torch.exp(-dist2 / max(params.s_r**2, 1e-3))
    g_a = torch.exp(-dist2 / max(params.s_a**2, 1e-3))

    if params.unity_normalization:
        direction = delta * torch.rsqrt(dist2).unsqueeze(-1)
        pair_term = direction
    else:
        pair_term = delta

    if params.attention_normalization:
        # Per-head attention normalization over neighbors l for each i.
        denom_r = g_r.sum(dim=1, keepdim=True).clamp_min(1e-8)
        denom_a = g_a.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weight_r = params.a_r * (g_r / denom_r)
        weight_a = -params.a_a * (g_a / denom_a)
        interaction = weight_r + weight_a
        return (interaction.unsqueeze(-1) * pair_term).sum(dim=1)

    interaction = params.a_r * g_r - params.a_a * g_a
    n = max(positions.size(0), 1)
    return (interaction.unsqueeze(-1) * pair_term).sum(dim=1) / float(n)


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
