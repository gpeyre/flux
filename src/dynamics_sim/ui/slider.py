"""Simple horizontal slider widget for pygame UI."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from ..toolbox import clamp


@dataclass
class SliderStyle:
    track_color: tuple[int, int, int] = (70, 78, 95)
    fill_color: tuple[int, int, int] = (82, 172, 255)
    knob_color: tuple[int, int, int] = (236, 245, 255)
    label_color: tuple[int, int, int] = (240, 240, 245)


class Slider:
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        label: str,
        min_value: float,
        max_value: float,
        value: float,
        decimals: int = 2,
        style: SliderStyle | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, width, 8)
        self.label = label
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.value = clamp(float(value), self.min_value, self.max_value)
        self.decimals = decimals
        self.style = style or SliderStyle()
        self.dragging = False

    def _value_from_mouse(self, mouse_x: int) -> float:
        ratio = (mouse_x - self.rect.left) / max(self.rect.width, 1)
        ratio = clamp(ratio, 0.0, 1.0)
        return self.min_value + ratio * (self.max_value - self.min_value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0, 14).collidepoint(event.pos):
                self.dragging = True
                self.value = self._value_from_mouse(event.pos[0])
                changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._value_from_mouse(event.pos[0])
            changed = True
        return changed

    def normalized(self) -> float:
        return (self.value - self.min_value) / max(self.max_value - self.min_value, 1e-8)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        track_rect = self.rect
        fill_rect = pygame.Rect(track_rect.left, track_rect.top, int(track_rect.width * self.normalized()), track_rect.height)
        knob_x = track_rect.left + int(track_rect.width * self.normalized())
        knob_center = (knob_x, track_rect.centery)

        pygame.draw.rect(surface, self.style.track_color, track_rect, border_radius=5)
        pygame.draw.rect(surface, self.style.fill_color, fill_rect, border_radius=5)
        pygame.draw.circle(surface, self.style.knob_color, knob_center, 7)

        value_str = f"{self.value:.{self.decimals}f}"
        label = font.render(f"{self.label}: {value_str}", True, self.style.label_color)
        surface.blit(label, (track_rect.left, track_rect.top - 20))
