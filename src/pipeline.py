"""Main navigation pipeline: detect -> estimate distance -> annotate
-> optional bird's-eye view and optical-flow tracking.

Everything is assembled here from the config file, so scripts stay thin.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.annotation.draw import draw_detections, draw_track_ids
from src.camera import Camera
from src.detection.model import Detector, Detections
from src.distance.geometry import estimate_distances


class _Rate:
    """Soft FPS limiter for live display."""

    def __init__(self, hz: float):
        self.dt = 1.0 / max(hz, 1e-6)
        self._t = time.perf_counter()

    def tick(self) -> None:
        now = time.perf_counter()
        wait = self.dt - (now - self._t)
        if wait > 0:
            time.sleep(wait)
        self._t = time.perf_counter()


@dataclass
class PipelineOptions:
    show: bool = True           # live window / imshow
    stream_read: bool = True    # read file back after writing (validation)
    use_bev: bool = False
    use_tracking: bool = False


class DistancePipeline:
    def __init__(self, config_path: str = "configs/config.yaml",
                 weights_override: Optional[str] = None,
                 backend: Optional[str] = None,
                 detector: Optional[Detector] = None):
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        det_cfg = self.cfg["detection"]
        self.camera = Camera.from_config(self.cfg["camera"])
        self.display_names = self.cfg.get("display_names", {})
        self.real_heights = self.cfg["real_heights_m"]
        backend = (backend or self.cfg.get("runtime", {}).get("backend", "torch")).lower()
        weights = weights_override or det_cfg["weights"]
        # `detector` can be injected (used by tests); otherwise build from config
        self.detector = detector or Detector(
            weights=weights,
            backend=backend,
            conf=det_cfg["conf"],
            iou=det_cfg["iou"],
            imgsz=det_cfg["imgsz"],
            device=det_cfg.get("device", "auto"),
        )
        self.options = PipelineOptions()
        self._bev_H = None
        self._tracker = None

    # optional features ------------------------------------------------------
    def enable_bev(self) -> "DistancePipeline":
        from src.geometry.birdseye import four_point_homography
        cfg = self.cfg["birdseye"]
        w, h = cfg["bev_width"], cfg["bev_height"]
        dst = [[0, 0], [w, 0], [w, h], [0, h]]
        self._bev_H = four_point_homography(cfg["src_points"], dst)
        self._m_per_px = cfg["meters_per_pixel"]
        self.options.use_bev = True
        return self

    def enable_tracking(self) -> "DistancePipeline":
        from src.geometry.optical_flow import OpticalFlowTracker
        of = self.cfg.get("optical_flow", {})
        self._tracker = OpticalFlowTracker(
            max_corners=of.get("max_corners", 40),
            max_age=of.get("max_age", 12),
            smooth_k=of.get("smooth_k", 5),
        )
        self.options.use_tracking = True
        return self

    # core -------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """One full pass. Returns (annotated_frame, info_lines)."""
        dets: Detections = self.detector(frame)
        names = [self._pretty(n) for n in dets.names]  # for track/legend text
        distances = estimate_distances(dets.names, dets.boxes, self.camera,
                                       self.real_heights)

        tracks = None
        if self.options.use_tracking and self._tracking_frames_only:
            tracks = self._tracker.update(frame, dets.boxes, names, distances)

        out = draw_detections(frame, dets.boxes, dets.names, distances,
                              self.display_names)
        if tracks is not None:
            # use smoothed distance when available (stable over time)
            draw_track_ids(out, [(t.box, t.track_id, t.name) for t in tracks])

        lines: List[str] = []
        if self.options.use_bev and self._bev_H is not None:
            out = self._render_bev(out, dets.boxes, names, distances)
            lines.append("BEV on")
        if tracks is not None:
            lines.append(f"tracking: {len(tracks)} track(s)")
        lines.append(f"detections: {len(dets)}")
        return out, lines

    def _render_bev(self, frame, boxes, names, distances):
        from src.geometry.birdseye import (bbox_to_ground_point, draw_bev_points,
                                           estimate_distance_bev, project_point,
                                           warp_bev)
        cfg = self.cfg["birdseye"]
        w, h = cfg["bev_width"], cfg["bev_height"]
        bev = warp_bev(frame, self._bev_H, (w, h))
        pts, labels = [], []
        for box, dist in zip(boxes, distances):
            gp = bbox_to_ground_point(box)
            bx, by = project_point(self._bev_H, gp)
            pts.append((bx, by))
            labels.append(f"{dist:.1f}m" if dist is not None else "")
        bev = draw_bev_points(bev, pts, labels)
        if cfg.get("inset", True):
            ih = int(frame.shape[0] * 0.28)
            scale = ih / h
            iw = int(w * scale)
            small = cv2.resize(bev, (iw, ih))
            frame[frame.shape[0] - ih:, frame.shape[1] - iw:] = small
        return frame

    def run_source(self, source, output: Optional[str] = None,
                   show: bool = True, max_frames: int = -1) -> None:
        """source: camera index, video path, image path, or directory of images."""
        rate = _Rate(30)
        from src.annotation.draw import draw_legend

        p = Path(str(source))
        if p.is_dir():  # batch of flat images
            out_dir = Path(output) if output else p.parent / "annotated"
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(p.glob("*.jp*g")) + sorted(p.glob("*.png")):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                result, lines = self.process_frame(frame)
                draw_legend(result, lines)
                cv2.imwrite(str(out_dir / img_path.name), result)
            return

        is_frame = p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        if is_frame:
            frame = cv2.imread(str(source))
            out, lines = self.process_frame(frame)
            from src.annotation.draw import draw_legend
            draw_legend(out, lines)
            if output:
                cv2.imwrite(output, out)
            if show:
                cv2.imshow("out", out)
                cv2.waitKey(0)
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"cannot open source: {source}")
        writer = None
        if output:
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, (w, h))
        idx = 0
        from src.annotation.draw import draw_legend
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, lines = self.process_frame(frame)
            draw_legend(out, lines)
            if writer is not None:
                writer.write(out)
            if show:
                cv2.imshow("distance detection", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            rate.tick()
            idx += 1
            if max_frames > 0 and idx >= max_frames:
                break
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    def _pretty(self, name: str) -> str:
        return self.display_names.get(name, name)

    @property
    def _tracking_frames_only(self) -> bool:
        return True
