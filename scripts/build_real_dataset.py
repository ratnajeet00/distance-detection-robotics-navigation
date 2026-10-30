#!/usr/bin/env python
"""Build the merged REAL training dataset (reproducible, one command):

  BDD100K stop signs   data/bdd/<split> label jsons; images already pulled
                       into data/bdd/images/{train,val} by fetch_bdd_images.py
  TraCon cones         real traffic-cone dataset (YOLO txt, single cone class)
  DORIE guardrails     real roadside-barrier subset, class 8 -> our 1
                       (scripts/build_dorie_barrier_extra.py builds the sets)

    python scripts/build_real_dataset.py --negatives 1500
    -> data/dataset/{train,val} + data/dataset/data.yaml
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.dataset import (
    bdd_to_yolo, build_data_yaml, extra_yolo_to_split, print_class_stats)

BDD = Path("data/bdd")
DATASET = Path("data/dataset")
SPLITS = ("train", "val")


def copy_negatives(split: str, negatives: int) -> int:
    """Add sampled stop-sign-free BDD frames as clean background images."""
    neg_file = BDD / f"negatives_{split}.txt"
    if not neg_file.exists():
        return 0
    negs = neg_file.read_text().split()
    out_img = DATASET / split / "images"
    out_lbl = DATASET / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in negs[:negatives]:
        stem = Path(name).stem
        src = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = BDD / "images" / split / f"{stem}{ext}"
            if cand.exists():
                src = cand
                break
        if not src:
            continue
        shutil.copy(src, out_img / f"{stem}{src.suffix}")
        (out_lbl / f"{stem}.txt").write_text("")
        copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=1500)
    ap.add_argument("--cone-root", default="data/extra_raw/tracon/TraCon_dataset")
    ap.add_argument("--barrier-root", default="data/dorie_barrier")
    ap.add_argument("--skip-bdd", action="store_true")
    args = ap.parse_args()

    if DATASET.exists():
        shutil.rmtree(DATASET)

    for split in SPLITS:
        if not args.skip_bdd and (BDD / f"wanted_{split}.txt").exists():
            n = bdd_to_yolo(BDD / f"{split}.json", BDD / "images" / split,
                            DATASET / split, copy_images=True)
            print(f"[bdd] {split}: {n} labeled stop-sign frames (from real json)")
        n_neg = copy_negatives(split, args.negatives)
        print(f"[bdd] {split}: {n_neg} negative frames added")

        for root, label in ((args.cone_root, "tracon"), (args.barrier_root, "dorie_barrier")):
            if not Path(root).exists():
                print(f"[merge] {label} {split}: skipped ({root} missing)")
                continue
            n = extra_yolo_to_split(Path(root), DATASET / split, src_split=split)
            print(f"[merge] {label} {split}: {n} frames")

    build_data_yaml(DATASET, DATASET / "data.yaml")
    print_class_stats(DATASET)


if __name__ == "__main__":
    main()
