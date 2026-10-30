"""End-to-end pipeline test with a stub detector (no model download needed),
plus the metric ground-truth test for BEV homography and epipolar geometry.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Camera
from src.detection.model import Detections
from src.distance.geometry import estimate_distance
from src.pipeline import DistancePipeline

ROOT = Path(__file__).resolve().parents[1]


class _FakeDetector:
    """Always sees one traffic cone, 60 px tall, in the middle of the frame."""

    def __call__(self, frame):
        h, w = frame.shape[:2]
        return Detections(
            boxes=np.array([[w // 2 - 30, 300, w // 2 + 30, 360]], np.float32),
            scores=np.array([0.92], np.float32),
            class_ids=np.array([0]),
            names=["traffic_cone"],
        )


def test_pipeline_annotates_and_writes():
    pipe = DistancePipeline(detector=_FakeDetector())
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:] = (40, 40, 40)
    out, lines = pipe.process_frame(frame)
    assert out.shape == frame.shape
    # box drawn at [610, 300, 670, 360] in color (40, 150, 255) BGR
    assert tuple(frame[300, 611]) == (40, 150, 255)
    assert any("detections: 1" in l for l in lines)

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "lane.jpg"
        cv2.imwrite(str(img), frame)
        out_path = Path(tmp) / "annotated.jpg"
        pipe.run_source(str(img), output=str(out_path), show=False)
        assert out_path.exists() and out_path.stat().st_size > 1000


def test_bev_distance_matches_ground_truth():
    """Synthetic pinhole scene, camera at 1.5 m above flat ground, looking
    straight ahead. BEV-projected distance of a ground point must reproduce
    hypot(X, Y)."""
    fx = fy = 800.0
    cx, cy = 320.0, 240.0
    cam_h = 1.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(X, Y):
        return cx + fx * X / Y, cy + fy * cam_h / Y  # straight-ahead looking

    # build H from 4 ground plane corners (X in [-0.5, 0.5], Y in [4, 8])
    corners = [(-0.5, 4), (0.5, 4), (0.5, 8), (-0.5, 8)]
    src = [project(*c) for c in corners]           # TL TR BR BL
    dst = [[c[0] * 10, c[1] * 10] for c in corners]  # 10 px per meter

    from src.geometry.birdseye import estimate_distance_bev, four_point_homography
    H = four_point_homography(src, dst)

    for X, Y in [(0.25, 6.0), (-0.3, 7.5), (0.4, 5.0)]:
        px, py = project(X, Y)
        box = [px - 2, py - 50, px + 2, py]  # bottom-center == ground point
        est = estimate_distance_bev(H, box, meters_per_pixel=0.1)
        truth = np.hypot(X, Y)
        assert est == pytest.approx(truth, abs=0.35), f"{est} vs {truth}"


def test_epipolar_constraint_on_synthetic_stereo():
    """Two cameras with pure x-translation: F estimated via our own pipeline
    must satisfy p2^T F p1 = 0 (up to a few px noise)."""
    rng = np.random.RandomState(42)
    n = 60
    pts3d = rng.uniform(-2, 2, (n, 3)) + np.array([0, 0, 6])  # in front of cams
    fx, fy, cx, cy = 400.0, 400.0, 320.0, 240.0

    def project(pts, shift):
        x = fx * pts[:, 0] / pts[:, 2] + cx + shift * fx / pts[:, 2]
        y = fy * pts[:, 1] / pts[:, 2] + cy
        return x, y

    x1, y1 = project(pts3d, 0)
    x2, y2 = project(pts3d, 0.3)  # baseline 0.3 m, camera 2 shifted +x
    p1 = np.stack([x1, y1], 1)
    p2 = np.stack([x2, y2], 1)

    from src.geometry.epipolar import estimate_fundamental
    F, mask = estimate_fundamental(p1, p2)
    assert mask.sum() >= n - 20  # nearly all points are inliers

    # epipolar residual on inliers: homogeneous (p2^T F p1)
    P1 = np.hstack([p1[mask], np.ones((mask.sum(), 1))])
    P2 = np.hstack([p2[mask], np.ones((mask.sum(), 1))])
    num = np.abs((P2 @ F @ P1.T).diagonal())
    denom = np.abs(P2 @ F[:, :2]).sum(1)  # type-1 normalization
    assert np.median(num / (denom + 1e-9)) < 1e-3


def test_camera_keeps_config_order():
    cam = Camera(width=1280, height=720, fx=800, fy=800, cx=640, cy=360)
    K = cam.intrinsics
    assert K[0, 2] == 640 and K[1, 2] == 360
    assert cam.pitch_rad() < 0  # slightly looking down
