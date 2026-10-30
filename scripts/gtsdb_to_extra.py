#!/usr/bin/env python
"""Convert the REAL GTSDB (German Traffic Sign Detection Benchmark) to our
extra-YOLO layout, keeping stop signs as class 2 (stop_sign).

GTSDB: genuine dashcam scenes (1360x800, German motorway/urban), 43 sign
classes in the RoI annotations (`name;x1;y1;x2;y2;ClassId`, ClassId 14 =
stop sign). We keep only the stop-sign boxes for the stop_sign class; the
remaining real sign classes are not needed downstream (BDD100K frames supply
the background images).

    python scripts/gtsdb_to_extra.py --root data/extra_raw/gtsdb \
        --out data/gtsdb_stop --split train

Expected --root layout (after unpacking archive.zip):
    <root>/full_images/*.png  (+  test_images/ for unlabeled split - skipped)
    <root>/annotations csv: any file where lines look like
        image.png;x1;y1;x2;y2;ClassId
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

# GTSDB RoI ClassId -> our unified class id (-1 = drop). GTSDB taxonomy is
# shared with GTSRB; class 14 is `stop`.
STOP_CLASS_ID = 14


def find_annotation_file(root: Path) -> Path | None:
    for p in sorted(root.rglob("*")):
        if not p.suffix.lower() in (".csv", ".txt", ".gt"):
            continue
        try:
            first = p.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        except OSError:
            continue
        for line in first:
            parts = line.split(";")
            if len(parts) >= 6:
                try:
                    int(parts[5])
                    int(parts[1])
                    return p
                except ValueError:
                    break
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="unpacked GTSDB dir")
    ap.add_argument("--out", required=True, help="extra-set output dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction held out as val when --split auto")
    ap.add_argument("--auto-split", action="store_true",
                    help="split positives 80/20 train/val")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    gt_file = find_annotation_file(root)
    if gt_file is None:
        raise SystemExit(f"[gtsdb] no RoI annotation file found under {root}")
    print(f"[gtsdb] annotations: {gt_file}")

    images = {p.stem: p for p in root.rglob("*.png")} | \
             {p.stem: p for p in root.rglob("*.jpg")}
    if not images:
        raise SystemExit(f"[gtsdb] no images under {root}")
    print(f"[gtsdb] images available: {len(images)}")

    rows = list(csv.reader(gt_file.read_text().splitlines(), delimiter=";"))
    positive = {}
    total, kept = 0, 0
    for parts in rows:
        if not parts or len(parts) < 6:
            continue
        stem = Path(parts[0]).stem
        cls_id = int(parts[5])
        total += 1
        if cls_id != STOP_CLASS_ID:
            continue
        if stem not in images:
            continue
        kept += 1
        positive.setdefault(stem, []).append(parts)

    stems = sorted(positive)
    print(f"[gtsdb] {total} RoI boxes total, {kept} stop-sign boxes in "
          f"{len(stems)} images")
    if args.auto_split:
        n_val = max(1, int(len(stems) * args.val_frac))
        val_stems, train_stems = set(stems[-n_val:]), set(stems[:-n_val])
    else:
        train_stems, val_stems = set(stems), set()
    parts = {"train": train_stems, "val": val_stems}

    for split, stems_in in parts.items():
        if not stems_in:
            continue
        dst = out / split
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        for stem in sorted(stems_in):
            img = images[stem]
            w = None
            # image dims from the PNG itself (GTSDB is fixed 1360x800 but be safe)
            import cv2
            im = cv2.imread(str(img))
            h_im, w_im = (im.shape[0], im.shape[1])
            lines = []
            for p in positive[stem]:
                x1, y1, x2, y2 = (float(v) for v in p[1:5])
                cx, cy = (x1 + x2) / 2 / w_im, (y1 + y2) / 2 / h_im
                bw, bh = (x2 - x1) / w_im, (y2 - y1) / h_im
                lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (dst / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            shutil.copy(img, dst / "images" / img.name)
        print(f"[gtsdb] {split}: {len(stems_in)} stop-sign images -> {dst}")


if __name__ == "__main__":
    main()
