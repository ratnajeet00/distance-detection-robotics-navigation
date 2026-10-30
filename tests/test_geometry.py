"""Quick geometric sanity tests (no model weights needed)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Camera
from src.distance.geometry import distance_from_real_height, estimate_distance


def _cam():
    return Camera(width=1280, height=720, fx=800, fy=800, cx=640, cy=360)


def test_pinhole_distance():
    # cone of 0.5 m appearing 50 px tall with fy=800 -> 8 m
    d = distance_from_real_height(0.5, 50, 800)
    assert abs(d - 8.0) < 1e-6


def test_estimate_distance_clamps():
    cam = _cam()
    d = estimate_distance("traffic_cone", [100, 300, 150, 350], cam)
    assert d == pytest.approx(800 * 0.5 / 50)
    assert estimate_distance("unknown_word", [0, 0, 10, 10], cam) is None
    # degenerate box
    assert estimate_distance("traffic_cone", [0, 0, 10, 0.5], cam) is None


def test_intrinsics_matrix():
    cam = _cam()
    K = cam.intrinsics
    assert K.shape == (3, 3)
    assert K[0, 0] == 800 and K[2, 2] == 1


def test_disparity_depth():
    from src.geometry.epipolar import depth_from_disparity
    # Z = fx * B / d
    assert depth_from_disparity(40, 800, 0.30) == pytest.approx(6.0)


def test_bev_homography_maps_bottom_to_near():
    from src.geometry.birdseye import four_point_homography, project_point
    src = [[540, 470], [740, 470], [1180, 660], [100, 660]]
    dst = [[0, 0], [400, 0], [400, 600], [0, 600]]
    H = four_point_homography(src, dst)
    x, y = project_point(H, src[2])  # maps exactly
    assert np.allclose([x / 400, y / 600], [1, 1], atol=1e-3)


def test_optical_flow_track_matches_by_iou():
    from src.geometry.optical_flow import iou
    a = np.array([0, 0, 10, 10], float)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, np.array([20, 20, 30, 30], float)) == pytest.approx(0.0)
