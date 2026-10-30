"""Monocular distance estimation from bounding boxes (pure geometry).

Method 1 -- known real height (default)
---------------------------------------
From the pinhole model, an object of real height ``H`` projecting to ``h``
pixels at image distance ``Z`` satisfies (similar triangles):

    h = fy * H / Z   ->   Z = fy * H / h

where ``fy`` is the vertical focal length in pixels and ``H`` the known
physical height of the object class (traffic cone ≈ 0.5 m, barrier ≈ 0.8 m,
stop sign ≈ 0.6 m sign face). Unknowns that cancel: the object does not need
to stand on any particular part of the image; only its full height must be
visible.

"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from src.camera import Camera

DEFAULT_REAL_HEIGHTS_M: Dict[str, float] = {
    "traffic_cone": 0.50,
    "barrier": 0.80,
    "stop_sign": 0.60,
    # alias: COCO's class name is "stop sign" (with a space), so pretrained
    # models work out of the box for that one class until fine-tuning is done
    "stop sign": 0.60,
}


def distance_from_real_height(real_height_m: float, pixel_height: float, fy: float) -> float:
    """Pinhole-model distance in meters: Z = fy * H / h (h > 0)."""
    if pixel_height <= 1:
        raise ValueError("pixel_height must be >= 1 px")
    return float(fy * real_height_m / pixel_height)


def estimate_distance(
    class_name: str,
    bbox_xyxy: Iterable[float],
    camera: Camera,
    real_heights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Distance in meters for one detection, or None if unknown/invalid."""
    heights = real_heights if real_heights is not None else DEFAULT_REAL_HEIGHTS_M
    real_h = heights.get(class_name)
    if real_h is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    pixel_h = y2 - y1
    if pixel_h < 1:  # degenerate box -> distance estimate would explode
        return None
    # Use the vertical (fy) axis: bounding-box height is measured vertically.
    return distance_from_real_height(real_h, pixel_h, camera.fy)


def estimate_distances(
    class_names: Iterable[str],
    boxes: np.ndarray,
    camera: Camera,
    real_heights: Optional[Dict[str, float]] = None,
) -> List[Optional[float]]:
    """Vector convenience wrapper over :func:`estimate_distance`."""
    if len(boxes) == 0:
        return []
    return [
        estimate_distance(name, box, camera, real_heights)
        for name, box in zip(class_names, np.asarray(boxes).reshape(-1, 4))
    ]
