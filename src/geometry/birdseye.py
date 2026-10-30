"""Homography / perspective transform: warping the scene to a bird's-eye view.

Theory (one line): if four image points of a *plane* are known in both the
image and a metric top-down frame, the mapping between the two planes is a
3x3 homography H (up to scale), H Appl (x, y, 1)^T = (x', y', 1)^T. Because H
is computed against a *metric* target rectangle, warping a point through H
gives its ground position in meters.

Two ways to get H are provided:
  * manual four-point calibration (``four_point_homography``) - robust when a
    user clicks a reasonable patch of road,
  * derived from intrinsics + flat-ground assumption
    (``ground_homography_from_intrinsics``) - no manual corners needed.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from src.camera import Camera

Order = ("TL", "TR", "BR", "BL")


def four_point_homography(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> np.ndarray:
    """Homography mapping 4 source image points (TL, TR, BR, BL) -> dst points."""
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("need exactly 4 point correspondences (TL, TR, BR, BL)")
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst))
    return H


def ground_homography_from_intrinsics(camera: Camera) -> np.ndarray:
    """H for the flat-ground plane at height ``camera.height_m`` below the camera.

    Standard result: for camera height h and pitch theta (looking down), a
    point on the ground at (X, Y_forward) maps as

        v = cy + fy_tan(theta) is the horizon row; image row y corresponds to
        forward distance Y = h * fy / (y - v_proj)  (along optical axis),
        lateral X = (x - cx) * Y / fx.

    We build H directly from these relations. Works without clicking any points.
    """
    f = camera.fy
    hpx = camera.height
    horizon = camera.cy + f * np.tan(-camera.pitch_rad())  # horizon row
    # Sample the relations: rows of ground grid -> image, invert to get H.
    ys = np.linspace(0.35, 0.98, 6) * hpx
    meters_per_px_x = camera.height_m * 0.01  # arbitrary lateral scale
    src, dst = [], []
    for y in ys:
        y = min(y, horizon - 2)  # must stay below horizon
        fwd = camera.height_m * f / (horizon - y)
        for x_px in (0.5 * camera.fx, 1.0 * camera.fx, 1.5 * camera.fx):
            x_img = camera.cx + (x_px - camera.cx) / 2
            lat = (x_img - camera.cx) * camera.height_m / max(f, 1)
            src.append([x_img, y])
            dst.append([lat * 20, fwd * 20])  # scale to px for bev image
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst))
    return H


def warp_bev(img: np.ndarray, H: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Warp image through H into a top-down (bird's-eye) view of size (w, h)."""
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR)


def project_point(H: np.ndarray, pt_xy: Sequence[float]) -> Tuple[float, float]:
    """Project one image point through H into BEV coordinates (pixels)."""
    v = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    w = H @ v
    return float(w[0] / w[2]), float(w[1] / w[2])


def bbox_to_ground_point(bbox_xyxy: Sequence[float]) -> Tuple[float, float]:
    """The point of a vehicle-adjacent object that touches the ground:
    bottom-center of its bounding box."""
    x1, _y1, x2, y2 = bbox_xyxy
    return (float(x1) + float(x2)) / 2.0, float(y2)


def estimate_distance_bev(
    H: np.ndarray,
    bbox_xyxy: Sequence[float],
    meters_per_pixel: float,
) -> Optional[float]:
    """Ground-plane distance: warp the box's bottom-center through H into the
    metric BEV frame and read off its distance from the robot (origin at the
    bottom-center of the BEV image)."""
    gp = bbox_to_ground_point(bbox_xyxy)
    bx, by = project_point(H, gp)
    return float(meters_per_pixel * np.hypot(bx, by))


def draw_bev_points(bev: np.ndarray, pts: Sequence[Tuple[float, float]],
                    labels: Optional[Sequence[str]] = None) -> np.ndarray:
    """Draw ground positions (with optional labels) on a BEV image."""
    out = bev
    for i, (x, y) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 5, (0, 90, 255), -1)
        if labels is not None:
            cv2.putText(out, labels[i], (p[0] + 6, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


__all__ = [
    "four_point_homography", "ground_homography_from_intrinsics", "warp_bev",
    "project_point", "bbox_to_ground_point", "estimate_distance_bev",
    "draw_bev_points",
]
