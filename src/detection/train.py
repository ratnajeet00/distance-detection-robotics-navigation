"""Transfer learning: fine-tune a pretrained YOLO backbone on the merged
dataset (BDD100K-derived stop signs + extra cones/barriers).

The pretrained backbone (yolov8n.pt / yolov8s.pt / ...) already knows generic
object structure from COCO; fine-tuning adapts it to our 3 robotics-relevant
classes, which is exactly the "transfer learning" requirement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def resolve_checkpoint(name: str = "bdd_finetune",
                       project: str | Path = "runs/train") -> Path:
    """Find the checkpoint in the run directory selected by Ultralytics."""
    project = Path(project)
    candidates = [
        project / name / "weights" / "best.pt",
        Path("runs") / "detect" / project / name / "weights" / "best.pt",
    ]
    candidates.extend(Path("runs").glob(f"**/{name}/weights/best.pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates[:2])
    raise FileNotFoundError(f"best.pt not found; searched {searched}")


def _metric_value(metrics, *names: str) -> Optional[float]:
    """Read a scalar from Ultralytics metrics across supported releases."""
    for name in names:
        value = metrics
        for part in name.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def metrics_dict(results) -> dict:
    """Return stable, serialisable detection metrics from ``model.val()``."""
    box = getattr(results, "box", results)
    values = {
        "precision": _metric_value(box, "mp", "p"),
        "recall": _metric_value(box, "mr", "r"),
        "map50": _metric_value(box, "map50"),
        "map50_95": _metric_value(box, "map"),
    }
    names = getattr(results, "names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    class_maps = getattr(box, "maps", None)
    if class_maps is not None:
        class_indices = getattr(box, "ap_class_index", None)
        if class_indices is None:
            class_indices = range(len(class_maps))
        values["per_class_map50_95"] = {
            str(names.get(int(class_id), class_id)): float(class_maps[int(class_id)])
            for class_id in class_indices
            if 0 <= int(class_id) < len(class_maps)
        }
    return values


def write_metrics_report(metrics: dict, output: str | Path, title: str = "Detection validation metrics") -> None:
    """Persist JSON and a human-readable Markdown validation report."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    md = output.with_suffix(".md")
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
    for key, label in labels.items():
        value = metrics.get(key)
        lines.append(f"| {label} | {value:.4f} |" if value is not None
                     else f"| {label} | unavailable |")
    per_class = metrics.get("per_class_map50_95", {})
    if per_class:
        lines += ["", "## Per-class mAP50-95", "", "| Class | mAP50-95 |",
                  "|---|---:|"]
        lines += [f"| {name} | {score:.4f} |"
                  for name, score in per_class.items()]
    md.write_text("\n".join(lines) + "\n")
    print(f"[val] metrics: {output}")
    print(f"[val] report: {md}")


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
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    best = (Path(save_dir) if save_dir else Path(project) / name) / "weights" / "best.pt"
    if not best.exists():
        best = resolve_checkpoint(name, project)
    print(f"[train] best weights: {best}")
    return YOLO(str(best))


def validate(weights: str | Path, data_yaml: str | Path, imgsz: int = 640,
             output: str | Path | None = "results/detection_metrics.json"):
    """Run val split evaluation and print per-class mAP."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), imgsz=imgsz, split="val", plots=True)
    metrics = metrics_dict(results)
    metrics["weights"] = str(weights)
    metrics["data"] = str(data_yaml)
    if output:
        write_metrics_report(metrics, output)
    return results
