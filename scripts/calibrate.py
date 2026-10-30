#!/usr/bin/env python
"""Calibrate the real camera intrinsics from checkerboard photos.

    python scripts/calibrate.py --images data/calib_imgs --pattern 12,9 --square 0.03

- shoot 15-30 photos of a printed checkerboard (keep it completely flat),
  varying distance/angle, with the FIXED camera setup you will deploy;
- the tool finds corners in every photo, runs cv2.calibrateCamera and prints
  the results + real reprojection error;
- the paste-ready yaml block lands in results/calibration.yaml

fx/fy are near-equal for most lenses; use fy for the distance formula
(bbox height is a vertical measure) and paste the distortion coefficients
into configs/config.yaml under camera: for undistortion work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of checkerboard photos")
    ap.add_argument("--pattern", default="12,9",
                    help="inner corners per row,col (cols,rows)")
    ap.add_argument("--square", type=float, default=0.03, help="square size in meters")
    ap.add_argument("--out", default="results/calibration.yaml")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.pattern.split(","))
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square

    objpoints, imgpoints = [], []
    files = sorted(Path(args.images).glob("*.jp*g")) + \
        sorted(Path(args.images).glob("*.png"))
    used = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"[calib] skipped {f.name} (no {cols}x{rows} pattern)")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 5:
        raise SystemExit(f"[calib] only {used} usable photos - need >= 5 "
                         "(15-30 is better). nothing calibrated.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"\n[calib] photos used: {used}, reprojection RMS error: {rms:.3f} px "
          "(< 0.5 is good, < 1.0 is usable)")
    print(f"[calib] fx = {fx:.2f} px   fy = {fy:.2f} px   "
          f"cx = {cx:.2f}   cy = {cy:.2f}")
    print(f"[calib] distortion: {dist.ravel().tolist()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# paste into configs/config.yaml (camera:)\n"
        f"# NOTE: recalibrate when you change lens/camera/FOV\n"
        f"fx: {fx:.2f}            # {fx:.2f}\n"
        f"fy: {fy:.2f}            # vertical focal - used by distance formula\n"
        f"cx: {cx:.2f}            # principal point x\n"
        f"cy: {cy:.2f}            # principal point y\n"
        f"distortion: [{', '.join(f'{v:.6f}' for v in dist.ravel())}]\n"
        f"# reprojection RMS error: {rms:.3f} px (over {used} photos)\n")
    print(f"[calib] paste-ready config -> {out}")


if __name__ == "__main__":
    main()
