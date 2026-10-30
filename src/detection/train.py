"""Transfer learning: fine-tune a pretrained YOLO backbone on the merged
dataset (BDD100K-derived stop signs + extra cones/barriers).

The pretrained backbone (yolov8n.pt / yolov8s.pt / ...) already knows generic
object structure from COCO; fine-tuning adapts it to our 3 robotics-relevant
classes, which is exactly the "transfer learning" requirement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def train(
    data_yaml: str | Path,
    base_weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: str = "bdd_finetune",
    patience: int = 15,
):
    """Fine-tune and return the trained ultralytics model."""
    from ultralytics import YOLO

    model = YOLO(base_weights)  # loads COCO-pretrained weights (transfer learning)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=(None if device in (None, "auto") else device),
        project=project,
        name=name,
        patience=patience,
        amp=True,           # mixed precision, faster on GPU
        cache=False,
        plots=True,
    )
    best = Path(project) / name / "weights" / "best.pt"
    print(f"[train] best weights: {best}")
    return YOLO(str(best))


def validate(weights: str | Path, data_yaml: str | Path, imgsz: int = 640):
    """Run val split evaluation and print per-class mAP."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.val(data=str(data_yaml), imgsz=imgsz, split="val", plots=True)
