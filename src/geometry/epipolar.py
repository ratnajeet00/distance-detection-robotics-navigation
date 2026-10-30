"""Epipolar geometry & stereo depth.

Derivation of the disparity-depth relation:
-------------------------------------------
Consider a calibrated stereo pair with the same intrinsics K, baseline ``B``
(horizontal camera separation), and no relative rotation. A 3D point at depth
``Z`` projects at ``x_l = fx * X / Z`` in the left image and at
``x_r = fx * (X - B) / Z`` in the right image. The disparity is

    d = x_l - x_r = fx * B / Z     =>     Z = fx * B / d

So depth is inversely proportional to disparity: far objects have small
disparity, near objects have large disparity. With arbitrary poses, the
epipolar constraint ``p_l^T F p_r = 0`` (fundamental matrix) and
``p_l'^T E p_r = 0`` (essential matrix, E = [t]_x R when pose is normalized)
lets us rectify the pair and recover the same relation after rectification.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def match_features(
    img1: np.ndarray, img2: np.ndarray,
    ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SIFT + FLANN + Lowe ratio test.

    Returns (keypoints1, keypoints2, matched_points1, matched_points2)
    with matched points in homogeneous image coords.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None:
        raise ValueError("no features found in one of the images")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    knn = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return k1, k2, p1, p2


def estimate_fundamental(p1: np.ndarray, p2: np.ndarray, ransac: float = 3.0):
    """F with RANSAC anc inlier mask, satisfying p_l^T F p_r = 0."""
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac, 0.99)
    return F, mask.ravel().astype(bool)


def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray, p1: np.ndarray,
                        p2: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Draw epipolar lines of points from img1 onto img2 (and vice versa)."""
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]
    lines_left = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines_right = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    out1, out2 = img1.copy(), img2.copy()
    for r in lines_left[:: max(1, len(lines_left) // 30)]:
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1_, y1_ = map(int, [w1, -(r[2] + r[0] * w1) / r[1]])
        cv2.line(out1, (x0, y0), (x1_, y1_), (0, 200, 0), 1, cv2.LINE_AA)
    for r in lines_right[:: max(1, len(lines_right) // 30)]:
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1_, y1_ = map(int, [w2, -(r[2] + r[0] * w2) / r[1]])
        cv2.line(out2, (x0, y0), (x1_, y1_), (0, 200, 0), 1, cv2.LINE_AA)
    return out1, out2


def depth_from_disparity(disparity_px: float, fx: float, baseline_m: float) -> float:
    """Z = fx * B / d (the disparity-depth relation derived above)."""
    if abs(disparity_px) < 1e-6:
        return float("inf")
    return float(fx * baseline_m / disparity_px)


def stereo_rectify(K: np.ndarray, R: np.ndarray, t_baseline: np.ndarray,
                   size: Tuple[int, int]):
    """Return rectification maps for a stereo pair (horizontal baseline)."""
    t = t_baseline.reshape(3, 1).astype(np.float64)
    R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(
        K, None, K, None, size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1l, map2l = cv2.initUndistortRectifyMap(K, None, R1, P1, size, cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(K, None, R2, P2, size, cv2.CV_32FC1)
    return (map1l, map2l), (map1r, map2r)


def recover_pose_and_triangulate(
    p1: np.ndarray, p2: np.ndarray, K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Essential-matrix pose (R, t) + triangulated 3D points (in cam1 frame).

    Returns (points_3d Nx3, R, t). Depth of each 3D point along +Z is its
    distance from camera 1 - usable for metric scale if t is scaled by |t|.
    """
    F, mask = estimate_fundamental(p1, p2)
    E = K.T @ F @ K
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    mask = mask.ravel().astype(bool)
    _, R, t, pose_mask = cv2.recoverPose(E, p1[mask], p2[mask], K)
    pm = pose_mask.ravel().astype(bool) & mask

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    pts4 = cv2.triangulatePoints(P1, P2,
                                 p1[pm].reshape(-1, 1, 2).T,
                                 p2[pm].reshape(-1, 1, 2).T)
    pts3 = (pts4 / pts4[3]).T[:, :3]
    return pts3, R, t
