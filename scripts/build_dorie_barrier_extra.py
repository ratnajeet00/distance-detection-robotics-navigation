#!/usr/bin/env python
"""Build the real-world barrier extra set from DORIE (Zenodo 17277466).

DORIE class 8 = "Guardrail: roadside safety barriers"; those boxes are
remapped onto our class 1 (barrier). Labels come from the pulled
data/extra_raw/dorie/DORIE/labels tree; the images themselves are fetched
from the remote zip with scripts/fetch_bdd_images.py and must be present
before this script runs (names are written to data/bdd/dorie_barrier_names.txt).

    python scripts/fetch_bdd_images.py \
        --url https://zenodo.org/records/17277466/files/DORIE.zip?download=1 \
        --names data/bdd/dorie_barrier_names.txt \
        --out data/extra_raw/dorie --resize 1280 --workers 3
    python scripts/build_dorie_barrier_extra.py \
        --train 280 --val 80 --out data/dorie_barrier
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SRC = Path("data/extra_raw/dorie/DORIE")


def guardrail_area(path: Path) -> float:
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "8":
            total += float(parts[3]) * float(parts[4])
    return total


def find_image(name: str, split: str) -> Path | None:
    """Fetched images may be staged split-wise or flat next to DORIE/labels."""
    for cand in (SRC / "images" / split / name, SRC / name,
                 SRC.parent / name, SRC.parent / split / name):
        if cand.exists():
            return cand
    return None


def build(split: str, count: int, out: Path, stage: str) -> list[str]:
    lbl_dir = SRC / "labels" / split
    ranked = sorted(lbl_dir.glob("*.txt"), key=guardrail_area, reverse=True)
    picked = []
    for p in ranked:
        if guardrail_area(p) <= 0:
            break
        picked.append(p)
        if len(picked) >= count:
            break

    dst_img = out / "images"
    dst_lbl = out / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    names, copied = [], 0
    for p in picked:
        name = p.stem
        img = find_image(f"{name}.JPG", split) or find_image(f"{name}.jpg", split)
        names.append(f"{name}.JPG")
        lines = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "8":      # Guardrail -> our barrier id
                parts[0] = "1"
                lines.append(" ".join(parts))
        assert lines, f"picked image without guardrail: {p}"
        (dst_lbl / f"{name}.txt").write_text("\n".join(lines) + "\n")
        if stage == "labels":
            continue
        if img is not None and img.exists():
            shutil.copy(img, dst_img / f"{name}.jpg")
            copied += 1
        elif stage == "images":
            print(f"[dorie] WARN: {name} image not found locally")
    if stage == "labels":
        (Path("data/bdd") / f"dorie_barrier_names_{split}.txt").write_text("\n".join(names))
    return names, copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=280)
    ap.add_argument("--val", type=int, default=80)
    ap.add_argument("--out", default="data/dorie_barrier")
    ap.add_argument("--stage", choices=["labels", "images"], required=True)
    args = ap.parse_args()
    out = Path(args.out)

    train_names, t_copy = build("train", args.train, out / "train", args.stage)
    val_names, v_copy = build("val", args.val, out / "val", args.stage)
    if args.stage == "labels":
        print("[dorie] label txts + wanted lists written "
              f"({len(train_names)} train / {len(val_names)} val with guardrails)")
    else:
        print(f"[dorie] images copied: {t_copy} train / {v_copy} val of "
              f"{len(train_names)}/{len(val_names)} expected")


if __name__ == "__main__":
    main()
