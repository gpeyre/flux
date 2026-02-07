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
    a_r: Slider
    a_a: Slider
    s_r: Slider
    s_a: Slider
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
        a_r=add("a_R = |v_1|", 0.0, 400.0, 220.0),
        a_a=add("a_A = |v_2|", 0.0, 200.0, 30.0),
        s_r=add("s_R (repulsive radius)", 4.0, 180.0, 45.0),
        s_a=add("s_A (attractive radius)", 10.0, 450.0, 180.0),
        external_strength=add("External Str.", 0.0, 320.0, 130.0),
        external_bandwidth=add("External BW", 20.0, 260.0, 110.0),
    )


def collect_params(ui: UIState, unity_on: bool, attention_on: bool) -> DynamicsParams:
    return DynamicsParams(
        friction=ui.friction.value,
        a_a=ui.a_a.value,
        s_a=ui.s_a.value,
        a_r=ui.a_r.value,
        s_r=ui.s_r.value,
        external_strength=ui.external_strength.value,
        external_bandwidth=ui.external_bandwidth.value,
        unity_normalization=unity_on,
        attention_normalization=attention_on,
    )


def world_to_screen(position: tuple[float, float], origin: tuple[int, int]) -> tuple[int, int]:
    return int(position[0] + origin[0]), int(position[1] + origin[1])


def screen_to_world(position: tuple[int, int], origin: tuple[int, int]) -> tuple[float, float]:
    return float(position[0] - origin[0]), float(position[1] - origin[1])


def compute_panel_layout(ui: UIState, panel_rect: pygame.Rect) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    sliders = list(ui.__dict__.values())
    last_slider = sliders[-1]
    info_top = panel_rect.bottom - 80
    toggle_height = 26
    gap = 12
    start = last_slider.rect.bottom + 16
    max_plot_height = max(40, info_top - start - toggle_height - gap - 8)
    plot_height = min(110, max_plot_height)
    plot_rect = pygame.Rect(panel_rect.left + 16, int(start), panel_rect.width - 32, int(plot_height))
    toggle_y = plot_rect.bottom + gap
    unity_rect = pygame.Rect(panel_rect.left + 16, int(toggle_y), 140, toggle_height)
    attention_rect = pygame.Rect(unity_rect.right + 12, int(toggle_y), 170, toggle_height)
    return plot_rect, unity_rect, attention_rect


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
    unity_on: bool,
    unity_toggle_rect: pygame.Rect,
    attention_on: bool,
    attention_toggle_rect: pygame.Rect,
    phi_plot_rect: pygame.Rect,
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

    trails_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    history = sim.history_cpu().numpy()
    history_len = history.shape[0]
    if history_len > 1:
        alpha_step = 255 / (history_len - 1)
        for particle_idx in range(history.shape[1]):
            for idx in range(1, history_len):
                x0, y0 = history[idx - 1, particle_idx]
                x1, y1 = history[idx, particle_idx]
                p0 = world_to_screen((float(x0), float(y0)), origin)
                p1 = world_to_screen((float(x1), float(y1)), origin)
                alpha = int(alpha_step * idx)
                pygame.draw.line(trails_surface, (232, 241, 255, alpha), p0, p1, width=1)
        screen.blit(trails_surface, (0, 0))

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

    pygame.draw.rect(screen, (20, 24, 33), phi_plot_rect, border_radius=6)
    pygame.draw.rect(screen, (58, 68, 86), phi_plot_rect, width=1, border_radius=6)
    plot_label = small_font.render("phi(r)", True, (200, 208, 220))
    screen.blit(plot_label, (phi_plot_rect.left + 8, phi_plot_rect.top + 4))

    if phi_plot_rect.width > 2 and phi_plot_rect.height > 2:
        plot_left = phi_plot_rect.left + 6
        plot_right = phi_plot_rect.right - 6
        plot_top = phi_plot_rect.top + 18
        plot_bottom = phi_plot_rect.bottom - 28
        plot_width = max(2, plot_right - plot_left)
        plot_height = max(2, plot_bottom - plot_top)

        r_max = max(ui.s_a.value * 3.0, ui.s_r.value * 3.0, 20.0)
        values = []
        for idx in range(plot_width):
            r = r_max * (idx / max(plot_width - 1, 1))
            attract = ui.a_a.value * math.exp(-(r * r) / max(ui.s_a.value**2, 1e-3))
            repel = ui.a_r.value * math.exp(-(r * r) / max(ui.s_r.value**2, 1e-3))
            values.append(repel - attract)

        min_val = min(values)
        max_val = max(values)
        max_abs = max(abs(min_val), abs(max_val), 1e-6)
        mid_y = plot_top + plot_height // 2
        scale = (plot_height / 2 - 2) / max_abs

        pygame.draw.line(
            screen,
            (92, 104, 126),
            (plot_left, mid_y),
            (plot_left + plot_width - 1, mid_y),
            width=1,
        )

        points = []
        for idx, val in enumerate(values):
            x = plot_left + idx
            y = int(mid_y - val * scale)
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.aalines(screen, (232, 241, 255), False, points)
    if unity_on:
        eq_text = "w_ij=(x_i-x_j)/|x_i-x_j|"
    else:
        eq_text = "w_ij=(x_i-x_j)"
    eq_surface = pygame.font.SysFont("timesnewroman", 16).render(eq_text, True, (188, 198, 214))
    screen.blit(eq_surface, (phi_plot_rect.left + 8, phi_plot_rect.bottom - 20))

    toggle_color = (76, 170, 120) if unity_on else (86, 96, 110)
    pygame.draw.rect(screen, toggle_color, unity_toggle_rect, border_radius=6)
    pygame.draw.rect(screen, (28, 34, 45), unity_toggle_rect, width=1, border_radius=6)
    toggle_label = small_font.render("Unity: On" if unity_on else "Unity: Off", True, (244, 246, 251))
    label_pos = toggle_label.get_rect(center=unity_toggle_rect.center)
    screen.blit(toggle_label, label_pos)

    attention_color = (76, 170, 120) if attention_on else (86, 96, 110)
    pygame.draw.rect(screen, attention_color, attention_toggle_rect, border_radius=6)
    pygame.draw.rect(screen, (28, 34, 45), attention_toggle_rect, width=1, border_radius=6)
    attention_label = small_font.render("Attention: On" if attention_on else "Attention: Off", True, (244, 246, 251))
    attention_pos = attention_label.get_rect(center=attention_toggle_rect.center)
    screen.blit(attention_label, attention_pos)

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = "cuda"
    else:
        device = torch.device("cpu")
        device_label = "cpu"

    origin = (CONFIG.margin, CONFIG.margin)
    panel_left = origin[0] + CONFIG.playground_size + CONFIG.margin
    panel_rect = pygame.Rect(panel_left, origin[1], CONFIG.panel_width, CONFIG.playground_size)
    ui = build_sliders(panel_left, origin[1] + 42)
    phi_plot_rect, unity_toggle_rect, attention_toggle_rect = compute_panel_layout(ui, panel_rect)

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
    unity_on = True
    attention_on = False
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
                elif unity_toggle_rect.collidepoint(event.pos):
                    unity_on = not unity_on
                elif attention_toggle_rect.collidepoint(event.pos):
                    attention_on = not attention_on

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                active_center = None

            elif event.type == pygame.MOUSEMOTION and active_center is not None:
                wx, wy = screen_to_world(event.pos, origin)
                wx = clamp(wx, 0.0, float(CONFIG.playground_size))
                wy = clamp(wy, 0.0, float(CONFIG.playground_size))
                centers[active_center, 0] = wx
                centers[active_center, 1] = wy

        params = collect_params(ui, unity_on, attention_on)
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
            unity_on,
            unity_toggle_rect,
            attention_on,
            attention_toggle_rect,
            phi_plot_rect,
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
