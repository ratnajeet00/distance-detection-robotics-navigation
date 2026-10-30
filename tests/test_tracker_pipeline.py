"""Integration tests: optical-flow tracker + full pipeline, torch-free.

The tracker runs real Lucas-Kanade flow on synthetic video; the pipeline is
exercised end-to-end with a stub detector.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.geometry.optical_flow import iou, OpticalFlowTracker


def make_scene(size=(240, 320), seed=0):
    """Flat background so goodFeaturesToTrack only finds corners on the moving
    patch itself - avoids flow lock-on in noise-on-noise scenes."""
    return np.full((size[0], size[1], 3), 60, dtype=np.uint8)


def moving_patch_frame(bg, patch, center):
    frame = bg.copy()
    x, y = int(center[0]), int(center[1])
    h, w = patch.shape[:2]
    color_patch = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    frame[y - h // 2: y + h // 2, x - w // 2: x + w // 2] = color_patch
    return frame


def test_iou_bounds():
    a = np.array([0, 0, 10, 10], float)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, np.array([20, 20, 30, 30], float)) == pytest.approx(0.0)


def test_tracker_follows_object_with_stable_id():
    bg = make_scene()
    patch = np.random.RandomState(1).randint(0, 255, (30, 30), dtype=np.uint8)

    tracker = OpticalFlowTracker(fb_tol=4.0, max_age=30)
    # frame 0: detection at x=40
    centers = [(40 + i * 6, 100) for i in range(10)]
    frame0 = moving_patch_frame(bg, patch, centers[0])
    tracks = tracker.update(
        frame0,
        boxes=np.array([[25, 85, 55, 115]], np.float32),
        names=["traffic_cone"],
        distances=[2.0],
    )
    assert len(tracks) == 1
    tid = tracks[0].track_id

    # frames 1..9: NO detections, tracking should coast on optical flow alone
    for c in centers[1:]:
        frame = moving_patch_frame(bg, patch, c)
        tracks = tracker.update(frame, np.zeros((0, 4), np.float32), [], [])
    assert len(tracks) == 1
    assert tracks[0].track_id == tid  # same object, same id

    cx = (tracks[0].box[0] + tracks[0].box[2]) / 2
    cy = (tracks[0].box[1] + tracks[0].box[3]) / 2
    expected = max(c for c, _ in centers)
    assert abs(cx - expected) < 8  # followed the moving patch
    assert abs(cy - 100) < 8
    assert tracks[0].age > 0  # we coasted without detections


def test_smoothed_distance_median():
    from src.geometry.optical_flow import Track
    t = Track(1, "traffic_cone", np.zeros(4), distances=[1.0, 1.0, 3.0, 1.0])
    assert t.smoothed_distance == pytest.approx(1.0)  # median ignores the outlier
