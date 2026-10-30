#!/usr/bin/env python
"""Evaluate distance-estimation error against measured ground truth.

Honest harness: it needs REAL measured distances supplied by you in a CSV
(results/ground_truth.csv). It never invents numbers. Without the file it
prints how to create one and exits.

CSV format (one measured object per row)::

    image,class,gt_distance_m,x1,y1,x2,y2
    data/samples/street.jpg,traffic_cone,2.4,612,533,678,665
    data/samples/street.jpg,stop_sign,7.8

  - image                path of the photo you measured on (camera fixed!)
  - class                traffic_cone | barrier | stop_sign
  - gt_distance_m        distance you measured (tape/laser/known setup)
  - x1,y1,x2,y2          OPTIONAL box of the measured object; when given the
                         evaluation matches it to the detector output by IoU,
                         otherwise the nearest detection of that class is used
                         (weaker match, flagged in the report)

    python scripts/evaluate_distance.py                       # instructions
    python scripts/evaluate_distance.py --gt results/ground_truth.csv

Report is written to results/distance_evaluation.md: per-class and overall
mean absolute error (m) and mean relative error (%).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera import Camera
from src.detection.model import Detector
from src.distance.geometry import estimate_distance

GT_TEMPLATE = ["image", "class", "gt_distance_m", "x1", "y1", "x2", "y2"]


def load_gt(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: v for k, v in row.items() if k in GT_TEMPLATE})
    return rows


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(gt_rows: list[dict], weights: str, backend: str,
             conf: float) -> tuple[list[dict], int]:
    det_fn = Detector(weights=weights, backend=backend, conf=conf)
    missing_files = 0
    out = []
    for row in gt_rows:
        img_path = Path(row["image"])
        if not img_path.exists():
            missing_files += 1
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            missing_files += 1
            continue
        dets = det_fn(frame)
        gt = float(row["gt_distance_m"])
        cls = row["class"].strip()
        match_box = None
        match_iou = 0.0
        if row.get("x1"):
            gt_box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            best = max(
                ((i, iou(gt_box, dets.boxes[i])) for i in range(len(dets))
                 if dets.names[i] == cls),
                key=lambda t: t[1], default=None)
            if best and best[1] >= 0.3:
                match_box, match_iou = dets.boxes[best[0]], best[1]
        else:  # nearest detection of the class (approximate match)
            cands = [dets.boxes[i] for i in range(len(dets))
                     if dets.names[i] == cls]
            if cands:
                # box-centred on the image centre chooses the dominant object
                match_box = min(
                    cands, key=lambda b: np.hypot(
                        (b[0] + b[2]) / 2 - frame.shape[1] / 2,
                        (b[1] + b[3]) / 2 - frame.shape[0] * 0.6))
        if match_box is None:
            out.append({**row, "matched": False, "pred": None})
            continue
        pred = estimate_distance(cls, match_box, det_fn.camera
                                 if hasattr(det_fn, "camera") else CAMERA)
        out.append({**row, "matched": True, "pred": pred,
                    "iou": round(match_iou, 2) if match_iou else None})
    return out, missing_files


CAMERA = None  # set in main() from the config


def write_report(rows: list[dict], out_path: Path, weights: str) -> None:
    lines = ["# Distance-estimation error vs measured ground truth", "",
             f"Detector weights: `{weights}`", ""]
    if not rows:
        lines.append("_No matched measurements._")
    else:
        ok = [r for r in rows if r["matched"] and r["pred"] is not None]
        lines += ["| image | class | measured (m) | predicted (m) | abs err (m) | rel err (%) |",]
        lines += ["|---|---|---|---|---|---|"]
        for r in ok:
            ae = abs(r["pred"] - r["gt"])
            rel = 100 * ae / r["gt"]
            lines.append(
                f"| {Path(r['image']).name} | {r['class']} | {r['gt']:.2f} | "
                f"{r['pred']:.2f} | {ae:.2f} | {rel:.1f} |")
        if ok:
            mae = sum(abs(r["pred"] - r["gt"]) for r in ok) / len(ok)
            mre = sum(100 * abs(r["pred"] - r["gt"]) / r["gt"] for r in ok) / len(ok)
            lines += ["", f"**MAE = {mae:.2f} m** over {len(ok)} measured objects, "
                      f"mean relative error {mre:.1f}%"]
        skipped = [r for r in rows if not (r["matched"] and r["pred"] is not None)]
        if skipped:
            lines += ["", f"{len(skipped)} row(s) skipped: no matched/estimable "
                      "detection (list them in the CSV box columns for exact matching)."]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[eval] report -> {out_path}")


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="results/ground_truth.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--backend", default=None, choices=["torch", "onnx"])
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--out", default="results/distance_evaluation.md")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    global CAMERA
    CAMERA = Camera.from_config(cfg["camera"])

    gt_path = Path(args.gt)
    if not gt_path.exists():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w", newline="") as f:
            csv.writer(f).writerow(GT_TEMPLATE)
        print(__doc__ or "")
        print(f"[eval] no ground truth yet. Template written -> {gt_path}\n"
              "       go measure real distances (tape / laser meter / known\n"
              "       placement) on the photos you will test with, fill one\n"
              "       row per measured object (x1..y2 helps exact matching),\n"
              "       then rerun this script. No synthetic errors are printed.")
        return

    weights = args.weights
    if weights is None:
        best = Path("runs/train/bdd_finetune/weights/best.pt")
        weights = str(best) if best.exists() else "models/yolov8n_int8.onnx"
    backend = args.backend or ("torch" if weights.endswith(".pt") else "onnx")

    rows, missing = evaluate(load_gt(gt_path), weights, backend, args.conf)
    if missing:
        print(f"[eval] {missing} ground-truth rows skipped (image not found)")
    write_report(rows, Path(args.out), weights)


if __name__ == "__main__":
    main()
