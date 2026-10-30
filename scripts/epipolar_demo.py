#!/usr/bin/env python
"""Run epipolar geometry on a supplied real stereo image pair.

This command never creates a stereo pair. Both images and the camera baseline
must be supplied by the user.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--fx", type=float, default=800.0)
    ap.add_argument("--baseline", type=float, required=True)
    ap.add_argument("--out", default="results/epipolar_real.png")
    args = ap.parse_args()

    left_path, right_path = Path(args.left), Path(args.right)
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError("both stereo input images must exist")
    img1, img2 = cv2.imread(str(left_path)), cv2.imread(str(right_path))
    if img1 is None or img2 is None:
        raise ValueError("could not decode both stereo input images")

    from src.geometry.epipolar import (draw_epipolar_lines,
                                       estimate_fundamental, match_features)
    _, _, p1, p2 = match_features(img1, img2)
    if len(p1) < 8:
        raise ValueError("at least 8 matched features are required")
    F, mask = estimate_fundamental(p1, p2)
    left, right = draw_epipolar_lines(img1, img2, p1[mask], p2[mask], F)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.hstack([left, right]))
    print(f"[epipolar] matches={len(p1)} inliers={int(mask.sum())}")
    print(f"[epipolar] baseline_m={args.baseline:.4f}")
    print(f"[epipolar] output={args.out}")


if __name__ == "__main__":
    main()
