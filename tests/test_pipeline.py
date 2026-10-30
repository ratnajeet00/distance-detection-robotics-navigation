import cv2
import numpy as np

from src.detection.model import Detections
from src.pipeline import DistancePipeline


class _FakeDetector:
    def __call__(self, frame):
        h, w = frame.shape[:2]
        return Detections(
            boxes=np.array([[w // 2 - 30, 300, w // 2 + 30, 360]], np.float32),
            scores=np.array([0.92], np.float32),
            class_ids=np.array([0]),
            names=["traffic_cone"],
        )


def test_pipeline_annotates_detection_and_distance():
    pipe = DistancePipeline(detector=_FakeDetector())
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    out, lines = pipe.process_frame(frame)
    assert out.shape == frame.shape
    assert tuple(out[300, 611]) == (40, 150, 255)
    assert any("detections: 1" in line for line in lines)


def test_pipeline_writes_annotated_image(tmp_path):
    pipe = DistancePipeline(detector=_FakeDetector())
    source = tmp_path / "source.jpg"
    output = tmp_path / "annotated.jpg"
    cv2.imwrite(str(source), np.zeros((720, 1280, 3), dtype=np.uint8))
    pipe.run_source(str(source), output=str(output), show=False)
    assert output.exists() and output.stat().st_size > 1000
