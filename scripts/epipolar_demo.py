#!/usr/bin/env python
"""Epipolar geometry demo on a stereo image pair.

python scripts/epipolar_demo.py --left left.jpg --right right.jpg \
    --fx 800 --baseline 0.30 --out results/epipolar.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--fx", type=float, default=800.0)
    ap.add_argument("--baseline", type=float, default=0.30,
                    help="known camera separation in meters")
    ap.add_argument("--out", type=str, default="results/epipolar.png")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img1 = cv2.imread(args.left)
    img2 = cv2.imread(args.right)

    from src.geometry.epipolar import (draw_epipolar_lines, estimate_fundamental,
                                       match_features, recover_pose_and_triangulate)
    _, _, p1, p2 = match_features(img1, img2)
    print(f"[demo] {len(p1)} matched points (SIFT + ratio test)")

    F, mask = estimate_fundamental(p1, p2)
    print(f"[demo] fundamental matrix:\n{np.round(F, 5)}")
    print(f"[demo] epipolar inliers: {mask.sum()}/{len(mask)}")

    K = np.array([[args.fx, 0, img1.shape[1] / 2],
                  [0, args.fx, img1.shape[0] / 2],
                  [0, 0, 1]])
    pts3, R, t = recover_pose_and_triangulate(p1[mask], p2[mask], K)
    zdepths = pts3[:, 2][pts3[:, 2] > 0]
    if len(zdepths):
        print(f"[demo] triangulated depth (cam1 frame): "
              f"median={np.median(zdepths):.2f} m (relative scale)")

    left, right = draw_epipolar_lines(img1, img2, p1[mask], p2[mask], F)
    canvas = np.hstack([left, right])
    cv2.imwrite(args.out, canvas)
    print(f"[demo] saved -> {args.out}")


if __name__ == "__main__":
    main()
