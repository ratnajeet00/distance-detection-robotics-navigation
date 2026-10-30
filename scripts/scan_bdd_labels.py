#!/usr/bin/env python
"""Scan the REAL BDD100K label release zip for stop-sign frames.

The release (bdd100k_labels.zip, the official working label file per
bdd100k/bdd100k#371) stores one small scalabel-style JSON per frame under
100k/{train,val,test}/<frame-id>.json. In this scalabel release every sign box
is a generic "traffic sign" (the v1 sign *type* attribute was dropped), so
stop signs inside it CANNOT be identified - stop-sign positives must come from
a differently-labelled real dataset (we use GTSDB). What BDD100K labels give
us for real is a clean negative/backgrounds source:

  phase scan -- walk every label json; a frame with ZERO 'traffic sign'
               objects cannot contain any stop sign, so those frames are
               guaranteed sign-free background images. Writes
               data/bdd/negatives_<split>.txt (sampled) and prints counts.

    python scripts/scan_bdd_labels.py --zip data/bdd/bdd100k_labels.zip scan
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import zipfile
from pathlib import Path


def frames_of_interest(z: zipfile.ZipFile, member: str) -> tuple[str, int, int]:
    """Return (frame_id, n_traffic_signs, n_any) parsed from one release json."""
    d = json.loads(z.read(member).decode())
    fid = d.get("name", Path(member).stem)
    n_sign = n_any = 0
    for fr in d.get("frames", []):
        for lbl in fr.get("objects", []):
            n_any += 1
            if lbl.get("category") == "traffic sign":
                n_sign += 1
    return fid, n_sign, n_any


def discover(args) -> None:
    z = zipfile.ZipFile(args.zip)
    members = [n for n in z.namelist()
               if n.startswith("100k/") and n.endswith(".json")]
    print(f"members: {len(members)} jsons; by top dir:",
          dict(collections.Counter(n.split('/')[1] for n in members)))
    rng = random.Random(0)
    sample = rng.sample(members, min(args.sample, len(members)))
    cats = collections.Counter()
    sign_attrs = collections.Counter()      # (key, value) on traffic-sign labels
    for m in sample:
        d = json.loads(z.read(m).decode())
        for fr in d.get("frames", []):
            for lbl in fr.get("labels", []):
                cats[lbl.get("category", "?")] += 1
                if lbl.get("category") == "traffic sign":
                    for k, v in (lbl.get("attributes") or {}).items():
                        sign_attrs[(k, v)] += 1
    print(f"\n[discover] categories in {len(sample)} files:")
    for c, n in cats.most_common():
        print(f"  {c:22s} {n}")
    print("\n[discover] traffic-sign attribute key/value pairs:")
    for (k, v), n in sign_attrs.most_common(20):
        print(f"  {k} = {v!r:20s} {n}")


def scan(args) -> None:
    z = zipfile.ZipFile(args.zip)
    members = [n for n in z.namelist()
               if n.startswith("100k/") and n.endswith(".json")]
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for n in members:
        split = n.split("/")[1]
        if split in by_split:
            by_split[split].append(n)

    base = Path("data/bdd")
    for split, files in by_split.items():
        negatives = []      # frame ids with zero traffic-sign objects
        for m in sorted(files):
            fid, n_sign, n_any = frames_of_interest(z, m)
            if n_sign == 0:
                negatives.append(fid)
        neg = base / f"negatives_{split}.txt"
        rng = random.Random(args.seed)
        rng.shuffle(negatives)
        neg.write_text("\n".join(negatives[: args.negatives]) + "\n")
        print(f"[scan] {split}: {len(files)} label files -> "
              f"{len(negatives)} guaranteed sign-free frames "
              f"(sampled {min(len(negatives), args.negatives)} "
              f"-> negatives_{split}.txt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/bdd/bdd100k_labels.zip")
    ap.add_argument("phase", choices=["discover", "scan"])
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--negatives", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "discover":
        discover(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
