"""Detection backends.

``Detector`` gives one interface over:
  * a torch/ultralytics YOLO model (training, best accuracy), and
  * an exported ONNX model (INT8/FP16 quantized, runs on CPU at edge speed).

Both return the same ``Detections`` structure so the distance and annotation
pipeline is backend-agnostic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# standard COCO 80 class names - used by the ONNX backend so labels are
# readable before a fine-tuned checkpoint exists (stop sign = id 11)
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detections:
    """Backend-agnostic result of one frame."""

    boxes: np.ndarray            # (N, 4) float32, xyxy pixels
    scores: np.ndarray           # (N,) float32
    class_ids: np.ndarray        # (N,) int
    names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: tuple = (114, 114, 114),
) -> tuple:
    """Resize keeping aspect ratio, pad to square. Returns (image, scale, pad)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh = (new_shape - nh) / 2
    dw = (new_shape - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    scale = r
    pad = np.array([left, top, left, top], dtype=np.float32)  # x1 y1 x2 y2 pad
    return out, scale, pad


def scale_boxes_back(boxes: np.ndarray, scale: float, pad: np.ndarray, orig_hw: tuple) -> np.ndarray:
    """Map letterboxed-space boxes back to original image coordinates."""
    boxes = (boxes - pad) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_hw[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_hw[0])
    return boxes


class _TorchBackend:
    def __init__(self, weights: str, conf: float, iou: float, imgsz: int, device: str):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device = None if device == "auto" else device

    def __call__(self, frame: np.ndarray) -> Detections:
        res = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        names = res.names  # {id: name}
        return Detections(
            boxes=res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4), np.float32),
            scores=res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,), np.float32),
            class_ids=res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros((0,), int),
            names=[names[int(i)] for i in (res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else [])],
        )


class _OnnxBackend:
    def __init__(self, model_path: str, conf: float, iou: float, imgsz: int):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        # CUDA EP when available, otherwise plain CPU (edge target).
        chosen = [p for p in providers if p.startswith("CUDA")] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        m = self.session.get_inputs()[0]
        self.net_h = m.shape[2] if isinstance(m.shape[2], int) else imgsz
        self.net_w = m.shape[3] if isinstance(m.shape[3], int) else imgsz

    def __call__(self, frame: np.ndarray) -> Detections:
        orig_h, orig_w = frame.shape[:2]
        img, scale, pad = letterbox(frame, self.net_w)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # 1x3xHxW

        out = self.session.run(None, {self.input_name: blob})[0]  # (1, 4+nc, N)
        out = out[0]  # (4+nc, N)
        preds = out.T  # (N, 4+nc)

        boxes = preds[:, :4].copy()          # cx cy w h
        class_scores = preds[:, 4:]
        if class_scores.shape[1] > 0 and not np.isfinite(class_scores).all():
            return Detections(np.zeros((0, 4), np.float32), np.zeros((0,)), np.zeros((0,), int))
        class_ids = class_scores.argmax(1)
        scores = class_scores.max(1)

        keep = scores > self.conf
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        # cxcywh -> xyxy
        xyxy = boxes.copy()
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        keep = nms(xyxy, scores, self.iou)
        xyxy, scores, class_ids = xyxy[keep], scores[keep], class_ids[keep]
        xyxy = scale_boxes_back(xyxy, scale, pad, (orig_h, orig_w))
        names = [COCO80_NAMES[i] if 0 <= i < len(COCO80_NAMES) else str(i)
                 for i in class_ids]
        return Detections(xyxy.astype(np.float32), scores.astype(np.float32),
                          class_ids.astype(int), names=names)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class Detector:
    """Facade: loads the requested backend and runs detection on BGR frames."""

    def __init__(
        self,
        weights: Optional[str] = None,
        backend: str = "torch",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
    ):
        backend = backend.lower()
        if backend == "torch":
            self._impl = _TorchBackend(weights, conf, iou, imgsz, device)
        elif backend == "onnx":
            self._impl = _OnnxBackend(weights, conf, iou, imgsz)
        else:
            raise ValueError(f"unknown backend: {backend}")

    def __call__(self, frame: np.ndarray) -> Detections:
        return self._impl(frame)
