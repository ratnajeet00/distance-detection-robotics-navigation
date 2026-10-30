"""Validate the dataset preparation path end to end (no downloads needed):

1. BDD100K JSON -> YOLO txt (stop signs kept, other categories/type dropped)
2. extra cone/barrier dataset merge (- class-id remap)
3. data.yaml generation

These mirror what ``scripts/prepare_bdd100k.py`` does, so a green run here is
the "script works" proof for reviewers.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.detection.dataset import (
    CLASS_NAMES,
    bdd_to_yolo,
    build_data_yaml,
    extra_yolo_to_split,
)


def _write_image(path, w=640, h=480):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def bdd_root(tmp_path):
    """A tiny synthetic BDD100K-like dataset: 2 images + one label JSON."""
    img_dir = tmp_path / "bdd" / "images"
    img_dir.mkdir(parents=True)
    _write_image(img_dir / "a.jpg")
    _write_image(img_dir / "b.jpg")

    labels = [
        {
            "name": "a.jpg",
            "labels": [
                {   # kept: stop sign
                    "category": "traffic sign",
                    "attributes": {"type": "stop sign"},
                    "box2d": {"x1": 100, "y1": 200, "x2": 180, "y2": 280},
                },
                {   # dropped: sign but not stop
                    "category": "traffic sign",
                    "attributes": {"type": "go"},
                    "box2d": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
                },
                {   # dropped: cone-flat category not handled by BDD path
                    "category": "traffic light",
                    "box2d": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
                },
            ],
        },
        {"name": "b.jpg", "labels": []},  # nothing to keep -> no label file
        {"name": "missing.jpg", "labels": [   # image absent -> skipped
            {
                "category": "traffic sign",
                "attributes": {"type": "stop sign"},
                "box2d": {"x1": 10, "y1": 10, "x2": 90, "y2": 90},
            },
        ]},
    ]
    return tmp_path / "bdd", write_json(tmp_path / "bdd" / "labels.json", labels)


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def test_bdd_to_yolo_stop_signs(bdd_root, tmp_path):
    (bdd_dir, labels_json) = bdd_root
    out = tmp_path / "dataset" / "train"
    n = bdd_to_yolo(labels_json, bdd_dir / "images", out, copy_images=True)
    assert n == 1  # only a.jpg had targets

    lbl = out / "labels" / "a.txt"
    assert lbl.exists()
    assert not (out / "labels" / "b.txt").exists()
    lines = lbl.read_text().strip().splitlines()
    assert len(lines) == 1

    cls, cx, cy, w, h = lines[0].split()
    assert cls == str(CLASS_NAMES.index("stop_sign"))  # id 2
    # box 100..180 x, 200..280 y on a 640x480 image, normalised
    assert abs(float(cx) - 140 / 640) < 1e-5
    assert abs(float(cy) - 240 / 480) < 1e-5
    assert abs(float(w) - 80 / 640) < 1e-5
    assert abs(float(h) - 80 / 480) < 1e-5
    assert (out / "images" / "a.jpg").exists()  # image copied alongside


def test_extra_cone_barrier_merge(tmp_path):
    """The '--extra' path: cone/barrier dataset with foreign class ids."""
    extra = tmp_path / "cones_kaggle"
    (extra / "images").mkdir(parents=True)
    (extra / "labels").mkdir()
    _write_image(extra / "images" / "c1.jpg")
    (extra / "images" / "c2.png").write_bytes(b"placeholder")

    # foreign ids: 0 = cone, 5 = barrier
    (extra / "labels" / "c1.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n5 0.2 0.7 0.4 0.3\n7 0.1 0.1 0.1 0.1\n")
    (extra / "labels" / "c2.txt").write_text("0 0.3 0.3 0.1 0.1\n")
    (extra / "labels" / "orphan.txt").write_text("0 0.3 0.3 0.1 0.1\n")  # no image

    out = tmp_path / "dataset" / "train"
    n = extra_yolo_to_split(
        extra, out,
        remap={"0": "traffic_cone", "5": "barrier"})
    assert n == 2  # c1 + c2 copied, orphan skipped

    merged = (out / "labels" / "cones_kaggle_c1.txt").read_text().strip().splitlines()
    parts0 = merged[0].split()
    assert merged[1].split()[0] == str(CLASS_NAMES.index("barrier"))     # 5 -> 1
    assert parts0 == [str(CLASS_NAMES.index("traffic_cone")),            # 0 -> 0
                      "0.5", "0.5", "0.2", "0.4"]                        # coords untouched
    assert (out / "images" / "cones_kaggle_c1.jpg").exists()


def test_extra_membership_cutoff(tmp_path):
    """Class ids without a mapping and beyond our 3 classes are dropped."""
    extra = tmp_path / "odd"
    (extra / "images").mkdir(parents=True)
    (extra / "labels").mkdir()
    _write_image(extra / "images" / "x.jpg")
    (extra / "labels" / "x.txt").write_text("99 0.1 0.1 0.1 0.1\n")  # unmapped junk

    out = tmp_path / "dataset" / "train"
    assert extra_yolo_to_split(extra, out) == 0
    assert not list((out / "labels").glob("*.txt"))


def test_build_data_yaml(tmp_path):
    root = tmp_path / "dataset"
    yaml = build_data_yaml(root, root / "data.yaml")
    text = yaml.read_text()
    assert "train: train/images" in text and "val: val/images" in text
    for i, name in enumerate(CLASS_NAMES):
        assert f"{i}: {name}" in text
    assert "path:" in text
