"""Annotation utilities: draw boxes like "Cone, 1.5m"."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# Interface colors, kept minimal and readable.
CLASS_COLORS: Dict[str, tuple] = {
    "traffic_cone": (40, 150, 255),   # orange-ish BGR
    "barrier": (255, 90, 0),
    "stop_sign": (60, 60, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_names: Sequence[str],
    distances: Sequence[Optional[float]],
    scores: Optional[Sequence[Optional[float]]] = None,
    display_names: Optional[Dict[str, str]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw each box with a "<Display Name> (<conf>), <d>m" label.

    `class_names` are the raw dataset names (used for the color table);
    `display_names` maps them to the label text, e.g. traffic_cone -> Cone.
    `distances`/`scores` may contain None entries (label parts are omitted).
    """
    out = frame
    if scores is None:
        scores = [None] * len(boxes)
    for i, (box, name, dist, score) in enumerate(
            zip(boxes, class_names, distances, scores)):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = CLASS_COLORS.get(name, (200, 200, 200))
        pretty = (display_names or {}).get(name, name)
        parts = [pretty]
        if score is not None:
            parts.append(f"({score * 100:.0f}%)")
        if dist is not None:
            parts.append(f"{dist:.1f}m")
        label = ", ".join(parts)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        _fill_label(out, label, (x1, max(0, y1 - 22)), color)
    return out


def _fill_label(img, text: str, origin: tuple, color: tuple) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, 0.55, 1)
    x, y = origin
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - base), FONT, 0.55, (20, 20, 20), 1, cv2.LINE_AA)


def draw_track_ids(frame: np.ndarray, tracks) -> np.ndarray:
    """Overlay track id on boxes given as list of (box, id, name)."""
    for box, tid, name in tracks:
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(frame, f"id{tid}", (x1, int(box[3]) + 18), FONT,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_legend(frame: np.ndarray, lines: List[str], origin=(8, 24)) -> np.ndarray:
    x, y = origin
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.5, 1)
        cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 6, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        y += th + 10
    return frame
