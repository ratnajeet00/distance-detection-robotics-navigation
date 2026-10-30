"""Camera model and intrinsics handling.

All distance estimation in this project is geometry-based (as the assignment
requires): every distance comes from the pinhole model
``distance = (real_height * focal_length_px) / pixel_height``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """Pinhole camera with intrinsics and mounting pose."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.4      # camera above ground plane
    pitch_deg: float = -10.0   # negative = looking slightly down

    @classmethod
    def from_config(cls, cfg: dict) -> "Camera":
        return cls(
            width=int(cfg["image_width"]),
            height=int(cfg["image_height"]),
            fx=float(cfg["fx"]),
            fy=float(cfg["fy"]),
            cx=float(cfg["cx"]),
            cy=float(cfg["cy"]),
            height_m=float(cfg.get("height_m", 1.4)),
            pitch_deg=float(cfg.get("pitch_deg", -10.0)),
        )

    @property
    def intrinsics(self) -> np.ndarray:
        """3x3 intrinsics matrix K (row-major image convention)."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)
