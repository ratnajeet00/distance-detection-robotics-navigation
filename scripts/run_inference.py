#!/usr/bin/env python
"""Run detection + distance annotation on an image / video / webcam.

Examples
--------
python scripts/run_inference.py --source data/samples/street.jpg --output out.jpg
python scripts/run_inference.py --source 0 --bev --track        # webcam, extras on
python scripts/run_inference.py --source video.mp4 --backend onnx \
    --weights models/yolov8n_int8.onnx --output annotated.mp4 --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import DistancePipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=str, default="0",
                    help="image path, video path, directory, or camera index")
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--backend", type=str, default=None, choices=["torch", "onnx"])
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--bev", action="store_true", help="bird's-eye-view overlay")
    ap.add_argument("--track", action="store_true", help="optical-flow tracking")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--max-frames", type=int, default=-1)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = DistancePipeline(args.config, weights_override=args.weights,
                                backend=args.backend)
    if args.bev:
        pipeline.enable_bev()
    if args.track:
        pipeline.enable_tracking()
    pipeline.run_source(source, output=args.output, show=args.show,
                        max_frames=args.max_frames)


if __name__ == "__main__":
    main()
