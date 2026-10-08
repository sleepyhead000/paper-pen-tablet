from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import FilterConfig


def _alpha(cutoff: float, dt: float) -> float:
    if cutoff <= 0 or dt <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One Euro filter (Casiez et al. 2012): adaptive low-pass whose cutoff
    rises with speed - low jitter at rest, low lag in motion."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_hat: float | None = None
        self.dx_hat = 0.0

    def reset(self) -> None:
        self.x_hat = None
        self.dx_hat = 0.0

    def filter(self, x: float, t: float) -> float:
        if self.x_hat is None:
            self.x_hat = x
            self.dx_hat = 0.0
            return x
        dt = t - getattr(self, "_t", t)
        self._t = t
        if dt <= 0:
            return self.x_hat
        dx = (x - self.x_hat) / dt
        ad = _alpha(self.d_cutoff, dt)
        self.dx_hat = self.dx_hat + ad * (dx - self.dx_hat)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_hat)
        a = _alpha(cutoff, dt)
        self.x_hat = self.x_hat + a * (x - self.x_hat)
        return self.x_hat


@dataclass
class SmoothedPoint:
    x: float
    y: float
    valid: bool  # safe to output (draw / move pointer)
    coasting: bool  # predicted through a short occlusion
    observed: bool  # True when this frame had a real tip detection


class Smoother:
    """One Euro on observed tips + constant-velocity coast through short
    occlusions; goes invalid after max_coast_frames so stale positions
    never drive the mouse (PRD §5)."""

    def __init__(self, cfg: FilterConfig | None = None):
        self.cfg = cfg or FilterConfig()
        self.fx = OneEuro(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self.fy = OneEuro(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self.x = 0.0
        self.y = 0.0
        self.t: float | None = None
        self.vx = 0.0
        self.vy = 0.0
        self.coast = 0
        self.last: SmoothedPoint | None = None

    def update(self, xy: tuple[float, float] | None, t: float) -> SmoothedPoint:
        if xy is not None:
            x = self.fx.filter(xy[0], t)
            y = self.fy.filter(xy[1], t)
            if self.t is not None and t > self.t:
                dt = t - self.t
                # observed velocity over the last frame, damped for coasting
                self.vx = 0.6 * (x - self.x) / dt + 0.4 * self.vx
                self.vy = 0.6 * (y - self.y) / dt + 0.4 * self.vy
            self.x, self.y, self.t = x, y, t
            self.coast = 0
            out = SmoothedPoint(x, y, valid=True, coasting=False, observed=True)
        else:
            self.coast += 1
            if self.last is None or self.coast > self.cfg.max_coast_frames or self.t is None:
                out = SmoothedPoint(
                    self.x, self.y, valid=False, coasting=False, observed=False
                )
            else:
                dt = t - self.t
                px = self.x + self.vx * dt
                py = self.y + self.vy * dt
                out = SmoothedPoint(px, py, valid=True, coasting=True, observed=False)
        self.last = out
        return out

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)
