"""Optical flow: track moving cones (and other detections) across frames.

Method
------
For each detection we seed sparse feature points inside its bounding box
(goodFeaturesToTrack restricted to the box) and propagate them with the
Lucas-Kanade pyramidal optical flow (``cv2.calcOpticalFlowPyrLK``).

A forward-backward consistency check rejects badly tracked points:
p is valid only if  ||LK(2nd pass, F(p)) - p|| < fb_tol.

Tracks keep identity by matching a new detection to the track's flow-predicted
box (IoU assignment), which pairs detections frame-to-frame while optical flow
carries the boxes between frames. Distances per track are median-filtered
over a small window so single jittery frames do not flash on screen.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    name: str
    box: np.ndarray                 # xyxy, last updated (possibly coasted)
    age: int = 0                    # frames since last matched detection
    distances: List[float] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    smooth_k: int = 5               # distance smoothing window (frames)

    @property
    def smoothed_distance(self) -> Optional[float]:
        if not self.distances:
            return None
        return statistics.median(self.distances[-max(1, self.smooth_k):])


def features_in_box(gray: np.ndarray, box: np.ndarray, max_corners: int,
                    corners_quality: float = 0.01, min_dist: int = 4) -> np.ndarray:
    """Good features to track, sampled only inside a detection box."""
    x1, y1, x2, y2 = box.astype(int)
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return np.zeros((0, 2), np.float32)
    roi = gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(
        roi, maxCorners=max_corners, qualityLevel=corners_quality,
        minDistance=min_dist, blockSize=7)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    pts = pts.reshape(-1, 2).astype(np.float32)
    pts += np.float32([x1, y1])  # back to full-frame coordinates
    return pts


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


class OpticalFlowTracker:
    """Sparse LK optical-flow tracker over detector outputs."""

    def __init__(self, max_corners: int = 40, fb_tol: float = 2.0,
                 match_iou: float = 0.35, max_age: int = 12,
                 smooth_k: int = 5):
        self.fb_tol = fb_tol
        self.max_corners = max_corners
        self.match_iou = match_iou
        self.max_age = max_age
        self.smooth_k = smooth_k
        self._tracks: List[Track] = []
        self._next_id = 1
        self._prev_gray: Optional[np.ndarray] = None

    # internal
    def _flow_step(self, gray: np.ndarray) -> None:
        """Propagate every track's box through LK flow (prev_gray -> gray)."""
        if self._prev_gray is None or not self._tracks:
            return
        all_pts = []
        owners = []
        for ti, tr in enumerate(self._tracks):
            pts = features_in_box(self._prev_gray, tr.box, self.max_corners)
            all_pts.append(pts)
            owners.extend([ti] * len(pts))
        if not all_pts:
            return
        p0 = np.concatenate(all_pts).reshape(-1, 1, 2).astype(np.float32)
        if p0.shape[0] == 0:
            return
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None)
        fb_err = np.linalg.norm((p0.reshape(-1, 2) - p0b.reshape(-1, 2)), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < self.fb_tol)

        movers: Dict[int, List[np.ndarray]] = {}
        for idx, ti in enumerate(owners):
            if ok[idx]:
                movers.setdefault(ti, []).append(p1.reshape(-1, 2)[idx])
        for ti, tr in enumerate(self._tracks):
            pts = movers.get(ti, [])
            if pts:
                c = np.mean(np.stack(pts), axis=0)
                prev_c = np.array([(tr.box[0] + tr.box[2]) / 2,
                                   (tr.box[1] + tr.box[3]) / 2], np.float32)
                tr.velocity = (c - prev_c).astype(np.float32)  # px/frame
                cx, cy = c
                hw = (tr.box[2] - tr.box[0]) / 2
                hh = (tr.box[3] - tr.box[1]) / 2
                tr.box = np.float32([cx - hw, cy - hh, cx + hw, cy + hh])

    def update(self, frame_bgr: np.ndarray, boxes: np.ndarray,
               names: List[str], distances: List[Optional[float]]) -> List[Track]:
        """Advance one frame. Returns active tracks (id-stable)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._flow_step(gray)
        self._prev_gray = gray

        # age everything; matching below resets the ones we see again
        for tr in self._tracks:
            tr.age += 1

        # Assign each new detection to at most one existing track (IoU greedy).
        used: set = set()
        matches: List[Tuple[bool, Track, int]] = []
        for bi in range(len(boxes)):
            best_iou, best_ti = self.match_iou, None
            for ti, tr in enumerate(self._tracks):
                if ti in used:
                    continue
                v = iou(boxes[bi], tr.box)
                if v > best_iou:
                    best_iou, best_ti = v, ti
            if best_ti is not None:
                used.add(best_ti)
                matches.append((True, self._tracks[best_ti], bi))
            else:
                matches.append((False, None, bi))

        for is_match, tr, bi in matches:
            if is_match:
                tr.box = boxes[bi].astype(np.float32)
                tr.age = 0
                if distances[bi] is not None:
                    tr.distances.append(float(distances[bi]))
            else:
                tr = Track(self._next_id, names[bi], boxes[bi].astype(np.float32),
                           smooth_k=self.smooth_k)
                self._tracks.append(tr)
                self._next_id += 1
                if distances[bi] is not None:
                    tr.distances.append(float(distances[bi]))

        # cull stale tracks; unmatched tracks keep coasting on their last box
        self._tracks = [tr for tr in self._tracks if tr.age <= self.max_age]
        return list(self._tracks)
