"""Configuration values for the particle dynamics app."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    playground_size: int = 760
    panel_width: int = 320
    margin: int = 20
    fps: int = 60
    particle_radius: int = 3
    max_particles: int = 1000
    min_particles: int = 1
    initial_particles: int = 120

    @property
    def window_width(self) -> int:
        return self.playground_size + self.panel_width + 3 * self.margin

    @property
    def window_height(self) -> int:
        return self.playground_size + 2 * self.margin


CONFIG = AppConfig()
