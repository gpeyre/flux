"""Interactive PyTorch + pygame particle dynamics sandbox."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
import torch
from .config import CONFIG
from .simulation import ParticleSimulation
from .toolbox import DynamicsParams, clamp
from .ui import Slider


@dataclass
class UIState:
    friction: Slider
    time_scale: Slider
    particle_count: Slider
    attract_strength: Slider
    attract_range: Slider
    repel_strength: Slider
    repel_range: Slider
    external_strength: Slider
    external_bandwidth: Slider


def build_sliders(panel_left: int, top: int) -> UIState:
    width = CONFIG.panel_width - 2 * 18
    gap = 54
    y = top

    def add(label: str, lo: float, hi: float, value: float, decimals: int = 2) -> Slider:
        nonlocal y
        slider = Slider(panel_left + 18, y, width, label, lo, hi, value, decimals=decimals)
        y += gap
        return slider

    return UIState(
        friction=add("Friction", 0.0, 4.0, 0.35),
        time_scale=add("Time Scale", 0.1, 4.0, 1.0),
        particle_count=add("Particles", CONFIG.min_particles, CONFIG.max_particles, CONFIG.initial_particles, decimals=0),
        attract_strength=add("Attract Str.", 0.0, 200.0, 30.0),
        attract_range=add("Attract Range", 10.0, 450.0, 180.0),
        repel_strength=add("Repel Str.", 0.0, 400.0, 220.0),
        repel_range=add("Repel Range", 4.0, 180.0, 45.0),
        external_strength=add("External Str.", 0.0, 320.0, 130.0),
        external_bandwidth=add("External BW", 20.0, 260.0, 110.0),
    )


def collect_params(ui: UIState) -> DynamicsParams:
    return DynamicsParams(
        friction=ui.friction.value,
        attract_strength=ui.attract_strength.value,
        attract_range=ui.attract_range.value,
        repel_strength=ui.repel_strength.value,
        repel_range=ui.repel_range.value,
        external_strength=ui.external_strength.value,
        external_bandwidth=ui.external_bandwidth.value,
    )


def world_to_screen(position: tuple[float, float], origin: tuple[int, int]) -> tuple[int, int]:
    return int(position[0] + origin[0]), int(position[1] + origin[1])


def screen_to_world(position: tuple[int, int], origin: tuple[int, int]) -> tuple[float, float]:
    return float(position[0] - origin[0]), float(position[1] - origin[1])


def draw_scene(
    screen: pygame.Surface,
    origin: tuple[int, int],
    panel_rect: pygame.Rect,
    sim: ParticleSimulation,
    centers: torch.Tensor,
    center_signs: torch.Tensor,
    emitter_rect: pygame.Rect,
    target_rect: pygame.Rect,
    score: int,
    ui: UIState,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    device_label: str,
) -> None:
    screen.fill((19, 23, 31))
    play_rect = pygame.Rect(origin[0], origin[1], CONFIG.playground_size, CONFIG.playground_size)
    pygame.draw.rect(screen, (28, 34, 45), play_rect)
    pygame.draw.rect(screen, (84, 94, 114), play_rect, width=2)

    pygame.draw.rect(screen, (239, 209, 88), emitter_rect, width=2)
    pygame.draw.rect(screen, (170, 120, 255), target_rect, width=2)

    bandwidth = int(ui.external_bandwidth.value)
    centers_cpu = centers.detach().cpu()
    signs_cpu = center_signs.detach().cpu()
    for idx in range(centers_cpu.shape[0]):
        cx, cy = centers_cpu[idx].tolist()
        sx, sy = world_to_screen((cx, cy), origin)
        attractive = signs_cpu[idx].item() > 0
        color = (66, 170, 255) if attractive else (240, 90, 90)
        pygame.draw.circle(screen, color, (sx, sy), bandwidth, width=1)
        pygame.draw.circle(screen, color, (sx, sy), 8)

    particles = sim.positions_cpu().numpy()
    for x, y in particles:
        px, py = world_to_screen((float(x), float(y)), origin)
        pygame.draw.circle(screen, (232, 241, 255), (px, py), CONFIG.particle_radius)

    pygame.draw.rect(screen, (24, 29, 39), panel_rect, border_radius=8)
    title = font.render("Controls", True, (244, 246, 251))
    screen.blit(title, (panel_rect.left + 16, panel_rect.top + 10))

    score_text = font.render(f"Score: {score}", True, (239, 209, 88))
    screen.blit(score_text, (panel_rect.left + 16, panel_rect.top + 40))

    for slider in ui.__dict__.values():
        slider.draw(screen, small_font)

    info_lines = [
        f"Device: {device_label}",
        "Yellow = emitter, Purple = absorber",
        "Drag blue/red centers in playground",
        "Blue = attract, Red = repel",
    ]
    info_y = panel_rect.bottom - 80
    for line in info_lines:
        surf = small_font.render(line, True, (172, 184, 201))
        screen.blit(surf, (panel_rect.left + 16, info_y))
        info_y += 22


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Flux")
    screen = pygame.display.set_mode((CONFIG.window_width, CONFIG.window_height))
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = "cuda" if device.type == "cuda" else "cpu"

    origin = (CONFIG.margin, CONFIG.margin)
    panel_left = origin[0] + CONFIG.playground_size + CONFIG.margin
    panel_rect = pygame.Rect(panel_left, origin[1], CONFIG.panel_width, CONFIG.playground_size)
    ui = build_sliders(panel_left, origin[1] + 42)

    centers = torch.tensor(
        [
            [CONFIG.playground_size * 0.3, CONFIG.playground_size * 0.3],
            [CONFIG.playground_size * 0.72, CONFIG.playground_size * 0.65],
        ],
        device=device,
        dtype=torch.float32,
    )
    center_signs = torch.tensor([1.0, -1.0], device=device, dtype=torch.float32)

    sim = ParticleSimulation(
        particle_count=int(ui.particle_count.value),
        box_size=float(CONFIG.playground_size),
        centers=centers,
        center_signs=center_signs,
        device=device,
    )

    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 18)

    emitter_size = 50
    target_size = 90
    emitter_world = pygame.Rect(
        int(CONFIG.playground_size * 0.08),
        int(CONFIG.playground_size * 0.1),
        emitter_size,
        emitter_size,
    )
    target_world = pygame.Rect(
        int(CONFIG.playground_size * 0.72),
        int(CONFIG.playground_size * 0.7),
        target_size,
        target_size,
    )
    emitter_rect = emitter_world.move(origin[0], origin[1])
    target_rect = target_world.move(origin[0], origin[1])

    score = 0
    emitter_rate = 28.0  # particles per second

    active_center: int | None = None
    running = True

    while running:
        frame_dt = clock.tick(CONFIG.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            slider_changed = any(slider.handle_event(event) for slider in ui.__dict__.values())
            if slider_changed:
                max_particles = int(round(ui.particle_count.value))
                if sim.particle_count > max_particles:
                    sim.set_particle_count(max_particles)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                play_rect = pygame.Rect(origin[0], origin[1], CONFIG.playground_size, CONFIG.playground_size)
                if play_rect.collidepoint(event.pos):
                    wx, wy = screen_to_world((mx, my), origin)
                    world = torch.tensor([wx, wy], device=device, dtype=torch.float32)
                    dists = torch.norm(centers - world, dim=1)
                    nearest = int(torch.argmin(dists).item())
                    if dists[nearest].item() <= max(ui.external_bandwidth.value * 0.2, 18.0):
                        active_center = nearest

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                active_center = None

            elif event.type == pygame.MOUSEMOTION and active_center is not None:
                wx, wy = screen_to_world(event.pos, origin)
                wx = clamp(wx, 0.0, float(CONFIG.playground_size))
                wy = clamp(wy, 0.0, float(CONFIG.playground_size))
                centers[active_center, 0] = wx
                centers[active_center, 1] = wy

        params = collect_params(ui)
        sim.set_centers(centers)

        max_particles = int(round(ui.particle_count.value))
        if sim.particle_count < max_particles:
            lam = emitter_rate * frame_dt
            spawn = int(torch.poisson(torch.tensor(lam)).item()) if lam > 0 else 0
            spawn = min(spawn, max_particles - sim.particle_count)
            if spawn > 0:
                min_corner = torch.tensor([emitter_world.left, emitter_world.top], device=device, dtype=torch.float32)
                max_corner = torch.tensor([emitter_world.right, emitter_world.bottom], device=device, dtype=torch.float32)
                sim.emit_particles(spawn, min_corner, max_corner)

        sim_time = frame_dt * ui.time_scale.value
        max_step = 1.0 / 240.0
        substeps = max(1, int(math.ceil(sim_time / max_step)))
        dt = sim_time / substeps
        for _ in range(substeps):
            sim.step(params, dt)

        min_corner = torch.tensor([target_world.left, target_world.top], device=device, dtype=torch.float32)
        max_corner = torch.tensor([target_world.right, target_world.bottom], device=device, dtype=torch.float32)
        absorbed = sim.absorb_particles(min_corner, max_corner)
        if absorbed:
            score += absorbed

        draw_scene(
            screen,
            origin,
            panel_rect,
            sim,
            centers,
            center_signs,
            emitter_rect,
            target_rect,
            score,
            ui,
            font,
            small_font,
            device_label,
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
