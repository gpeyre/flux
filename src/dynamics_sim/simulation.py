"""Simulation state and ODE integration for particles."""

from __future__ import annotations

import torch

from .toolbox import (
    DynamicsParams,
    compute_external_force,
    compute_pairwise_force,
    random_positions,
    random_positions_in_bounds,
    random_velocities,
)


class ParticleSimulation:
    def __init__(
        self,
        particle_count: int,
        box_size: float,
        centers: torch.Tensor,
        center_signs: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.box_size = float(box_size)
        self.device = device
        self.dtype = dtype
        self.centers = centers.to(device=self.device, dtype=self.dtype)
        self.center_signs = center_signs.to(device=self.device, dtype=self.dtype)

        self.positions = random_positions(particle_count, self.box_size, self.device, self.dtype)
        self.velocities = random_velocities(particle_count, speed_scale=25.0, device=self.device, dtype=self.dtype)

    @property
    def particle_count(self) -> int:
        return int(self.positions.shape[0])

    def set_centers(self, centers: torch.Tensor) -> None:
        self.centers = centers.to(device=self.device, dtype=self.dtype)

    def set_particle_count(self, target_count: int) -> None:
        target_count = max(1, int(target_count))
        current = self.particle_count
        if target_count == current:
            return

        if target_count < current:
            keep = torch.randperm(current, device=self.device)[:target_count]
            self.positions = self.positions[keep]
            self.velocities = self.velocities[keep]
            return

        add_count = target_count - current
        new_pos = random_positions(add_count, self.box_size, self.device, self.dtype)
        new_vel = random_velocities(add_count, speed_scale=20.0, device=self.device, dtype=self.dtype)
        self.positions = torch.cat([self.positions, new_pos], dim=0)
        self.velocities = torch.cat([self.velocities, new_vel], dim=0)

    def emit_particles(self, count: int, min_corner: torch.Tensor, max_corner: torch.Tensor) -> None:
        count = int(max(0, count))
        if count <= 0:
            return
        new_pos = random_positions_in_bounds(count, min_corner, max_corner, self.device, self.dtype)
        new_vel = random_velocities(count, speed_scale=18.0, device=self.device, dtype=self.dtype)
        self.positions = torch.cat([self.positions, new_pos], dim=0)
        self.velocities = torch.cat([self.velocities, new_vel], dim=0)

    def absorb_particles(self, min_corner: torch.Tensor, max_corner: torch.Tensor) -> int:
        if self.particle_count <= 0:
            return 0
        inside_x = (self.positions[:, 0] >= min_corner[0]) & (self.positions[:, 0] <= max_corner[0])
        inside_y = (self.positions[:, 1] >= min_corner[1]) & (self.positions[:, 1] <= max_corner[1])
        inside = inside_x & inside_y
        absorbed = int(inside.sum().item())
        if absorbed == 0:
            return 0
        keep = ~inside
        self.positions = self.positions[keep]
        self.velocities = self.velocities[keep]
        return absorbed

    def step(self, params: DynamicsParams, dt: float) -> None:
        if self.particle_count <= 0:
            return

        pair_force = compute_pairwise_force(self.positions, params)
        external_force = compute_external_force(self.positions, self.centers, self.center_signs, params)

        acceleration = pair_force + external_force - params.friction * self.velocities
        self.velocities = self.velocities + dt * acceleration
        self.positions = self.positions + dt * self.velocities

        # Reflective boundaries.
        below_zero = self.positions < 0.0
        self.positions = torch.where(below_zero, -self.positions, self.positions)
        self.velocities = torch.where(below_zero, torch.abs(self.velocities), self.velocities)

        above_box = self.positions > self.box_size
        self.positions = torch.where(above_box, 2.0 * self.box_size - self.positions, self.positions)
        self.velocities = torch.where(above_box, -torch.abs(self.velocities), self.velocities)

    def positions_cpu(self) -> torch.Tensor:
        return self.positions.detach().to(device="cpu")
