import torch

from dynamics_sim.toolbox.physics import DynamicsParams, compute_external_force, compute_pairwise_force


def test_pairwise_force_shape_and_conservation_like_symmetry() -> None:
    params = DynamicsParams(
        friction=0.2,
        a_a=10.0,
        s_a=100.0,
        a_r=20.0,
        s_r=30.0,
        external_strength=50.0,
        external_bandwidth=80.0,
    )
    positions = torch.tensor([[10.0, 30.0], [40.0, 30.0], [25.0, 55.0]])
    forces = compute_pairwise_force(positions, params)
    assert forces.shape == positions.shape
    assert torch.isfinite(forces).all()


def test_external_force_is_zero_at_center_when_single_particle() -> None:
    params = DynamicsParams(
        friction=0.0,
        a_a=0.0,
        s_a=100.0,
        a_r=0.0,
        s_r=10.0,
        external_strength=10.0,
        external_bandwidth=40.0,
    )
    center = torch.tensor([[50.0, 70.0]])
    signs = torch.tensor([1.0])
    positions = torch.tensor([[50.0, 70.0]])
    force = compute_external_force(positions, center, signs, params)
    assert torch.allclose(force, torch.zeros_like(force), atol=1e-6)


def test_external_force_points_toward_attractive_center() -> None:
    params = DynamicsParams(
        friction=0.0,
        a_a=0.0,
        s_a=100.0,
        a_r=0.0,
        s_r=10.0,
        external_strength=20.0,
        external_bandwidth=80.0,
    )
    center = torch.tensor([[0.0, 0.0]])
    signs = torch.tensor([1.0])
    positions = torch.tensor([[10.0, 0.0]])
    force = compute_external_force(positions, center, signs, params)
    assert force[0, 0] < 0.0
