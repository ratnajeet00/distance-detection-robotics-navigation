#!/usr/bin/env python
"""Prepare the merged dataset (BDD100K stop signs + extra cones/barriers).

Examples
--------
python scripts/prepare_bdd100k.py \
    --bdd-labels data/bdd/train_labels.json --bdd-images data/bdd/images \
    --out data/dataset --split train

python scripts/prepare_bdd100k.py \
    --extra data/cones_kaggle --out data/dataset --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.dataset import bdd_to_yolo, build_data_yaml, extra_yolo_to_split, print_class_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bdd-labels", type=str, help="BDD100K label JSON file")
    ap.add_argument("--bdd-images", type=str, help="BDD100K images folder")
    ap.add_argument("--extra", type=str, action="append", default=[],
                    help="extra YOLO-formatted dataset root (cones/barriers); repeatable")
    ap.add_argument("--out", type=str, default="data/dataset")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = ap.parse_args()

    out_split = Path(args.out) / args.split
    if args.bdd_labels:
        n = bdd_to_yolo(Path(args.bdd_labels), Path(args.bdd_images),
                        out_split, copy_images=True)
        print(f"[prepare] BDD100K -> {args.split}: wrote {n} label files")
    for extra in args.extra:
        root = Path(extra)
        n = extra_yolo_to_split(root, out_split)
        print(f"[prepare] extra {root.name} -> {args.split}: wrote {n} label files")

    build_data_yaml(Path(args.out), Path(args.out) / "data.yaml")
    print(f"[prepare] data.yaml -> {args.out}/data.yaml")
    print_class_stats(args.out)


if __name__ == "__main__":
    main()
