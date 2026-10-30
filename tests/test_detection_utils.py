"""Unit tests for detection-model math utilities (no torch/ultralytics needed)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.model import letterbox, nms, scale_boxes_back


def test_letterbox_roundtrip():
    img = np.zeros((720, 1280, 3), np.uint8)
    lb, scale, pad = letterbox(img, 640)
    assert lb.shape == (640, 640, 3)
    # scale for wide image: 640/1280 = 0.5
    assert abs(scale - 0.5) < 1e-6

    # forward: box in original coords -> letterboxed coords -> back to original
    orig = np.array([[100, 200, 300, 500]], np.float32)
    lb_space = orig * scale + pad  # x' = x*s + pad, y' = y*s + pad
    back = scale_boxes_back(lb_space.copy(), scale, pad, (720, 1280))
    assert np.allclose(back, orig, atol=1e-3)


def test_nms_suppresses_overlap():
    boxes = np.array([
        [10, 10, 50, 50],    # high score
        [12, 12, 52, 52],    # heavy overlap with first -> suppressed
        [200, 200, 260, 260],  # far away -> kept
    ], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, iou_thr=0.5)
    assert keep == [0, 2]
