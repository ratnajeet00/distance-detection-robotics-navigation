#!/usr/bin/env python
"""Prune a trained model (magnitude pruning) and optionally fine-tune.

Examples
--------
python scripts/prune.py --weights runs/train/bdd_finetune/weights/best.pt --amount 0.3
python scripts/prune.py --weights best.pt --amount 0.3 --fine-tune --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.prune import prune_yolo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--amount", type=float, default=0.3,
                    help="fraction of smallest-magnitude weights to zero")
    ap.add_argument("--fine-tune", action="store_true",
                    help="recover accuracy with short fine-tuning")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--data", type=str, default="data/dataset/data.yaml")
    args = ap.parse_args()

    pruned = prune_yolo(args.weights, amount=args.amount)
    print(f"[prune] saved: {pruned}")

    if args.fine_tune:
        from ultralytics import YOLO
        model = YOLO(pruned)
        model.train(data=args.data, epochs=args.epochs, imgsz=640,
                    project="runs/train", name="pruned_finetune", amp=True)
        print("[prune] fine-tuned -> runs/train/pruned_finetune/weights/best.pt")


if __name__ == "__main__":
    main()
