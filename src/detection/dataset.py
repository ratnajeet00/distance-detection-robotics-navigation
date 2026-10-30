"""Dataset preparation: BDD100K -> YOLO format, plus helpers to merge in
extra cone/barrier datasets.

BDD100K ground truth (v1 labels) is a JSON per image with 2D boxes. Traffic
signs carry an ``attributes.type`` field whose possible values include
``"stop sign"``, "no parking", "go" etc. We remap:

    traffic sign with type "stop sign"  -> stop_sign
    everything else                     -> ignored

Cones/barriers are NOT in BDD100K, so we support merging any additional
dataset that is already (or convertible to) YOLO txt format by passing a
``--extra`` directory with a class-id remap, e.g. map "cone"=0 -> traffic_cone=0
in our unified label space:

    0: traffic_cone
    1: barrier
    2: stop_sign
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CLASS_NAMES = ["traffic_cone", "barrier", "stop_sign"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Default remap for extra datasets: their class name -> our unified name.
DEFAULT_EXTRA_REMAP: Dict[str, str] = {
    "cone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "trafficcone": "traffic_cone",
    "barrier": "barrier",
    "stop_sign": "stop_sign",
    "stopsign": "stop_sign",
    "stop": "stop_sign",
}

IMAGES_DIR = "images"
LABELS_DIR = "labels"


def _ensure_dirs(split_root: Path) -> Tuple[Path, Path]:
    img_dir = split_root / IMAGES_DIR
    lbl_dir = split_root / LABELS_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def bdd_to_yolo(labels_json: Path, images_dir: Path, out_split_root: Path,
                copy_images: bool = True) -> int:
    """Convert one BDD100K label JSON set into YOLO txt files.

    Requires a ``bdd100k_labels_images_train.json``-style list of objects:
    [{"name": "a.jpg", "attributes": {...}, "labels": [
        {"category": "traffic sign", "attributes": {"type": "stop sign"},
         "box2d": {"x1":..,"y1":..,"x2":..,"y2":..}}, ...]}]
    """
    labels_json = Path(labels_json)
    img_dir = Path(images_dir)
    out_img, out_lbl = _ensure_dirs(out_split_root)

    data: List[dict] = json.loads(labels_json.read_text())
    written = 0
    for entry in data:
        name: str = entry["name"]
        img_path = img_dir / name
        if not img_path.exists():
            continue

        lines: List[str] = []
        src_img = None  # lazy-read once to get WxH
        for lbl in entry.get("labels", []):
            if lbl.get("category") != "traffic sign":
                continue
            stype = (lbl.get("attributes") or {}).get("type", "")
            if stype != "stop sign":
                continue
            if src_img is None:
                import cv2
                src_img = cv2.imread(str(img_path))
                if src_img is None:
                    break
                H, W = src_img.shape[:2]
            box = lbl["box2d"]
            cx = (box["x1"] + box["x2"]) / 2 / W
            cy = (box["y1"] + box["y2"]) / 2 / H
            w = (box["x2"] - box["x1"]) / W
            h = (box["y2"] - box["y1"]) / H
            if w <= 0 or h <= 0:
                continue
            cls = CLASS_TO_ID["stop_sign"]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:  # keep matching label file only where targets exist
            continue
        out_lbl.joinpath(Path(name).with_suffix(".txt")).write_text("\n".join(lines))
        if copy_images:
            shutil.copy(img_path, out_img / name)
        written += 1
    return written


def extra_yolo_to_split(extra_root: Path, out_split_root: Path,
                        remap: Dict[str, str] | None = None) -> int:
    """Import an extra dataset laid out as::

        extra_root/
            images/*.jpg          # either flat, or nested any level
            labels/*.txt          # YOLO txt, class ids may need remapping

    If ``remap`` is given as {old_id: our_name}, ids are remapped. If extra
    txt class ids already match ours it stays a no-op copy.
    """
    remap = remap or {}
    out_img, out_lbl = _ensure_dirs(out_split_root)
    src_img = sorted(extra_root.glob("images/**/*.jp*g")) + sorted(extra_root.glob("images/**/*.png"))
    src_lbl = sorted(extra_root.glob("labels/**/*.txt"))
    lbl_of = {Path(p).stem: p for p in src_lbl}
    copied = 0
    for img in src_img:
        lbl = lbl_of.get(img.stem)
        if lbl is None:
            continue
        dst_name = f"{extra_root.name}_{img.stem}{img.suffix}"
        out = []
        for line in lbl.read_text().strip().splitlines():
            parts = line.split()
            cls = int(parts[0])
            new_name = remap.get(str(cls))
            if new_name is None and cls < len(CLASS_NAMES):
                new_name = CLASS_NAMES[cls]
            if new_name not in CLASS_TO_ID:
                continue
            parts[0] = str(CLASS_TO_ID[new_name])
            out.append(" ".join(parts))
        if out:
            out_lbl.joinpath(dst_name + ".txt").write_text("\n".join(out))
            shutil.copy(img, out_img / dst_name)
            copied += 1
    return copied


def build_data_yaml(dataset_root: Path, out_yaml: Path) -> Path:
    """Write the YOLO data.yaml for the merged dataset."""
    root = Path(dataset_root).resolve()
    text = (
        f"path: {root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)) + "\n"
    )
    out_yaml.write_text(text)
    return out_yaml


def print_class_stats(dataset_root: Path) -> None:
    print(f"{'class':<15}{'train':>8}{'val':>8}")
    for split in ("train", "val"):
        counts = {n: 0 for n in CLASS_NAMES}
        root = Path(dataset_root) / split / LABELS_DIR
        for txt in root.glob("*.txt"):
            for line in txt.read_text().strip().splitlines():
                cls = int(line.split()[0])
                counts[CLASS_NAMES[max(0, min(cls, len(CLASS_NAMES) - 1))]] += 1
        for n, c in counts.items():
            print(f"{n:<15}{c:>8}" + ("" if split == "train" else ""))
