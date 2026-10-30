#!/usr/bin/env python
"""Generate the INT8 calibration image set (synthetic, deterministic).

Usage: python scripts/gen_calib.py [--out data/calib] [--count 24]

Any images work in principle; we use scenes from the same renderer as the demo
outputs so calibration sees thin cone/barrier-like shapes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cv2
import numpy as np

import make_demo_output as demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/calib")
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(3)
    for i in range(args.count):
        frame = np.zeros((720, 1280, 3), np.uint8)
        demo.draw_road(frame)
        frame = cv2.convertScaleAbs(frame, alpha=0.8 + 0.4 * rng.rand(),
                                    beta=rng.randint(-20, 20))
        x = rng.randint(60, 900)
        alt = rng.rand()
        if alt < 0.7:
            demo.draw_cone(frame, x, rng.randint(480, 680), rng.randint(50, 150))
            if alt >= 0.4:
                demo.draw_cone(frame, min(1100, x + rng.randint(100, 400)),
                               rng.randint(480, 660), rng.randint(40, 120))
        else:
            demo.draw_barrier(frame, x, rng.randint(430, 560),
                              rng.randint(90, 200), rng.randint(40, 90))
        if i % 3 == 0:
            demo.draw_stop_sign(frame, rng.randint(50, 220), rng.randint(430, 540),
                                rng.randint(50, 90))
        cv2.imwrite(str(out / f"calib_{i:02d}.jpg"), frame)
    print(f"wrote {args.count} calibration images -> {out}")


if __name__ == "__main__":
    main()
