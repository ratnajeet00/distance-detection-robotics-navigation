#!/usr/bin/env python
"""Compare validation accuracy for detector variants on the same split.

Example:
    python scripts/compare_accuracy.py --data data/dataset/data.yaml ^
      --model fp32=runs/train/bdd_finetune/weights/best.pt ^
      --model int8=models/bdd_finetune_int8.onnx

Every model is evaluated independently with the same Ultralytics validation
settings. Missing model files are reported as errors rather than omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.train import metrics_dict


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/dataset/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--model", action="append", required=True,
                    metavar="LABEL=PATH")
    ap.add_argument("--out", default="results/accuracy_comparison.json")
    args = ap.parse_args()

    rows = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            raise FileNotFoundError(f"model not found: {path}")
        results = YOLO(path).val(data=args.data, imgsz=args.imgsz,
                                 conf=args.conf, split="val", plots=False)
        row = {"variant": label, "weights": path, **metrics_dict(results)}
        rows.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    md = output.with_suffix(".md")
    lines = ["# Detection accuracy comparison", "",
             "| Variant | Precision | Recall | mAP50 | mAP50-95 |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return f"{value:.4f}" if value is not None else "unavailable"
        lines.append(f"| {row['variant']} | {fmt('precision')} | "
                     f"{fmt('recall')} | {fmt('map50')} | {fmt('map50_95')} |")
    md.write_text("\n".join(lines) + "\n")
    print(f"[compare] JSON: {output}")
    print(f"[compare] Markdown: {md}")


if __name__ == "__main__":
    main()
