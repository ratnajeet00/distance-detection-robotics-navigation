#!/usr/bin/env python
"""Fine-tune (transfer learning) the detector on the merged dataset.

Example
-------
python scripts/train.py --data data/dataset/data.yaml \
    --base yolov8n.pt --epochs 50 --batch 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.train import resolve_checkpoint, train, validate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/dataset/data.yaml")
    ap.add_argument("--base", type=str, default="yolov8n.pt",
                    help="pretrained backbone for transfer learning")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--name", type=str, default="bdd_finetune")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--metrics-out", type=str,
                    default="results/detection_metrics.json",
                    help="JSON path; Markdown is written beside it")
    args = ap.parse_args()

    if args.validate_only:
        weights = resolve_checkpoint(args.name)
        validate(weights, args.data,
                 args.imgsz, args.metrics_out)
    else:
        model = train(args.data, base_weights=args.base, epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch, device=args.device,
                      name=args.name)
        weights = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else resolve_checkpoint(args.name)
        validate(weights, args.data,
                 args.imgsz, args.metrics_out)


if __name__ == "__main__":
    main()
