#!/usr/bin/env python
"""Generate demo outputs into results/output_samples/ - what the pipeline
returns once everything runs. Torch-free by design: uses the real geometry /
annotation / BEV / flow / epipolar code with stub detections (same shapes the
real detector produces). On a trained-model machine the outputs look the same,
only the boxes come from YOLO.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.annotation.draw import draw_detections, draw_legend, draw_track_ids
from src.camera import Camera
from src.distance.geometry import estimate_distance, estimate_distances
from src.pipeline import DistancePipeline
from src.detection.model import Detections

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "output_samples"
DISPLAY = {"traffic_cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}
COLORS = {"traffic_cone": (40, 150, 255), "barrier": (255, 90, 0), "stop_sign": (60, 60, 255)}
CAM = Camera(width=1280, height=720, fx=800, fy=800, cx=640, cy=360)


# --------------------------------------------------------------------------
# synthetic scene: road + objects with known pixel heights. the stub detector
# just returns their boxes; distances still come from src.distance.geometry.
# --------------------------------------------------------------------------
def draw_road(frame):
    HIT = 430
    frame[:HIT] = (150, 130, 105)                       # sky-ish (BGR tan)
    frame[HIT:] = (70, 70, 70)                          # asphalt
    for i in range(6):
        x = 90 + i * 200
        cv2.rectangle(frame, (x, 555), (x + 70, 565), (240, 240, 240), -1)
    cv2.rectangle(frame, (0, 300), (int(0.33 * 1280), HIT), (60, 130, 60), -1)
    cv2.rectangle(frame, (int(0.72 * 1280), 300), (1280, HIT), (60, 130, 60), -1)


def draw_cone(frame, cx, base_y, h_px):
    """traffic cone ~0.5m tall"""
    top = base_y - h_px
    w = int(h_px * 0.85)
    pts = np.array([[cx - w // 2, base_y], [cx + w // 2, base_y],
                    [cx, top]], np.int32)
    cv2.fillPoly(frame, [pts], (30, 120, 240))
    cv2.fillPoly(frame, [np.array([[cx - w // 3, base_y - h_px // 4],
                                   [cx + w // 3, base_y - h_px // 4],
                                   [cx + w // 4, base_y - int(h_px * 0.45)],
                                   [cx - w // 4, base_y - int(h_px * 0.45)]], np.int32)],
                 (240, 240, 240))
    cv2.rectangle(frame, (cx - w // 2, base_y - 5), (cx + w // 2, base_y), (30, 120, 240), -1)
    return [cx - w // 2, top, cx + w // 2, base_y]


def draw_stop_sign(frame, cx, base_y, h_px):
    """stop sign face ~0.6m; red octagon on a gray pole"""
    top = base_y - h_px
    r = h_px * 2 / 3
    cy = top + r
    angles = np.deg2rad(np.arange(8) * 45 + 22.5)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], 1).astype(np.int32)
    cv2.fillPoly(frame, [pts], (60, 60, 230))
    cv2.putText(frame, "STOP", (int(cx - r), int(cy + 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (int(cx) - 3, int(cy + r)), (int(cx) + 3, base_y), (180, 180, 190), -1)
    return [int(cx - r), int(top), int(cx + r), int(base_y)]


def draw_barrier(frame, x, y, w, h):
    """barrier ~0.8m tall; red/white striped block"""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 80, 200), -1)
    for i in range(-1, w // 40 + 1):
        pts = np.array([[x + i * 40, y + h], [x + i * 40 + 25, y + h],
                        [x + i * 40 + 25 + h, y], [x + i * 40 + h, y]], np.int32)
        cv2.fillPoly(frame, [pts], (240, 240, 240))
    return [x, y, x + w, y + h]


def scene_objects(state: str = "bend"):
    """Objects placed per scene state. distances below come out of the REAL
    geometry code (Z = fy*H/h), matching the placement we made."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    draw_road(frame)
    det = []
    if state == "bend":
        det.append(("traffic_cone", draw_cone(frame, 1050, 560, 110)))
        det.append(("traffic_cone", draw_cone(frame, 900, 515, 60)))
        det.append(("barrier", draw_barrier(frame, 380, 470, 130, 80)))
        det.append(("stop_sign", draw_stop_sign(frame, 180, 500, 70)))
    else:  # "straight"
        det.append(("traffic_cone", draw_cone(frame, 600, 640, 150)))
        det.append(("barrier", draw_barrier(frame, 760, 500, 150, 95)))
    return frame, det


# --------------------------------------------------------------------------
def save_basic(out_dir: Path):
    """1) plain detection + distance annotation."""
    frame, det = scene_objects("bend")
    boxes = np.array([d[1] for d in det], np.float32)
    names = [d[0] for d in det]
    dists = [estimate_distance(n, b, CAM) for n, b in det]
    out = draw_detections(frame, boxes, names, dists, display_names=DISPLAY)
    draw_legend(out, ["distances from Z = fy*H/h (fy=800)"])
    # sanity: computed distances for our drawn sizes
    truth = [0.5 * 800 / (b[3] - b[1]) for _, b in det[:2]]
    assert abs(dists[0] - truth[0]) < 0.02 and abs(dists[1] - truth[1]) < 0.02
    cv2.imwrite(str(out_dir / "01_detection_distance.jpg"), out)
    return frame, boxes, names, dists


def save_bev(frame, boxes, names, dists, out_dir: Path):
    """2) same frame but with the bird's-eye-view inset on."""
    from src.geometry.birdseye import (bbox_to_ground_point, draw_bev_points,
                                       project_point, warp_bev)
    cfg = {
        "src_points": [[430, 500], [850, 500], [1200, 700], [120, 700]],
        "meters_per_pixel": 0.05,
        "bev_width": 400, "bev_height": 600,
    }
    from src.geometry.birdseye import four_point_homography
    H = four_point_homography(cfg["src_points"],
                              [[0, 0], [400, 0], [400, 600], [0, 600]])
    bev = warp_bev(frame, H, (400, 600))
    pts, labels = [], []
    for b, n, d in zip(boxes, names, dists):
        bx, by = project_point(H, bbox_to_ground_point(b))
        pts.append((bx, by))
        labels.append(f"{d:.1f}m" if d is not None else "")
    bev = draw_bev_points(bev, pts, labels)
    # 1m grid so the view reads as a map (origin = robot at bottom-center)
    for gx in range(0, 400, 20):
        cv2.line(bev, (gx, 0), (gx, 600), (90, 90, 90), 1)
    for gy in range(0, 600, 20):
        cv2.line(bev, (0, gy), (400, gy), (90, 90, 90), 1)
    cv2.circle(bev, (200, 600), 6, (60, 255, 60), -1)  # robot position
    bev = draw_bev_points(bev, pts, labels)
    ih = int(frame.shape[0] * 0.35)
    small = cv2.resize(bev, (int(400 * ih / 600), ih))
    x0, y0 = frame.shape[1] - small.shape[1], frame.shape[0] - ih
    frame[y0:, x0:] = small
    cv2.rectangle(frame, (x0, y0), (x0 + small.shape[1], y0 + ih), (255, 255, 255), 2)
    draw_legend(frame, ["birds-eye view (ground homography)",
                        "orange dots = object foot-points, grid = 1m"])
    cv2.imwrite(str(out_dir / "02_birdeye_inset.jpg"), frame)


def save_tracking(out_dir: Path):
    """3) optical flow tracking: cone approaching the robot, no detections
    after frame 0 - flow carries it and ids stay stable."""
    from src.geometry.optical_flow import OpticalFlowTracker
    tracker = OpticalFlowTracker(max_corners=60, fb_tol=5.0, max_age=40)
    bg = np.zeros((720, 1280, 3), np.uint8)
    draw_road(bg)
    frames = []
    # cone walks right -> left AND grows (getting closer)
    n_frames = 14
    boxes_seq = []
    for i in range(n_frames):
        cx = 980 - i * 42
        h = 70 + i * 9
        base_y = 500 + i * 4
        f = bg.copy()
        b = draw_cone(f, cx, base_y, h)
        boxes_seq.append(b)
        frames.append(f)
    boxes = np.array(boxes_seq, np.float32)
    dist_seq = [estimate_distance("traffic_cone", b, CAM) for b in boxes_seq]

    snap = {}
    tracks = tracker.update(frames[0], boxes[:1], ["Cone"], [dist_seq[0]])
    snap[0] = (frames[0].copy(), list(tracks))
    for i in range(1, n_frames):
        tracks = tracker.update(frames[i], boxes[i:i + 1] if i % 4 == 0 else np.zeros((0, 4), np.float32),
                                ["Cone"] if i % 4 == 0 else [], [dist_seq[i]] if i % 4 == 0 else [])
        if i in (5, 9, 13):
            snap[i] = (frames[i].copy(), list(tracks))

    for i, (f, trs) in snap.items():
        # only annotate the (single) main cone track on top of its box
        main = min(trs, key=lambda t: abs((t.box[0] + t.box[2]) / 2 - 980 + i * 42))
        draw_track_ids(f, [(main.box, main.track_id, main.name)])
        filtered = [t for t in trs if t is main]
        d = main.smoothed_distance
        draw_legend(f, [f"tracking {filtered[0].name} id{main.track_id}"] +
                    ([f"distance {d:.1f}m (smoothed)"] if d else []))
        if i == n_frames - 1:  # keep just the last snapshot on disk
            cv2.imwrite(str(out_dir / "03_tracking_final.jpg"), f)

    # write the whole step as a small mp4
    vw = cv2.VideoWriter(str(out_dir / "04_tracking_clip.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
    wrote = vw.isOpened()
    if wrote:
        tracker2 = OpticalFlowTracker(max_corners=60, fb_tol=5.0, max_age=40)
        trs = tracker2.update(frames[0], boxes[:1], ["Cone"], [dist_seq[0]])
        f = draw_track_ids(frames[0].copy(), [(t.box, t.track_id, t.name) for t in trs])
        draw_legend(f, ["tracking %s id%d" % (trs[0].name, trs[0].track_id)])
        vw.write(f)
        for i in range(1, n_frames):
            show_det = i % 4 == 0
            trs = tracker2.update(frames[i],
                                  boxes[i:i + 1] if show_det else np.zeros((0, 4), np.float32),
                                  ["Cone"] if show_det else [],
                                  [dist_seq[i]] if show_det else [])
            main = min(trs, key=lambda t: abs((t.box[0] + t.box[2]) / 2 - (980 - i * 42)))
            f = draw_track_ids(frames[i].copy(), [(main.box, main.track_id, main.name)])
            d = main.smoothed_distance
            draw_legend(f, ["tracking %s id%d" % (main.name, main.track_id)] +
                        ([f"distance {d:.1f}m (smoothed)"] if d else []))
            vw.write(f)
        vw.release()
    return dist_seq


def save_stereo(out_dir: Path):
    """4) synthetic stereo pair -> real epipolar code path."""
    rng = np.random.RandomState(7)
    n = 320
    pts = rng.uniform(-2, 2, (n, 3))
    pts[:, 2] += 6.0  # all in front of the cameras
    radii = rng.randint(3, 7, n)  # SAME radius in both views (same 3D object)
    fx = 800.0

    def px(pts, shift):
        x = pts[:, 0] - shift
        return ((x / pts[:, 2]) * fx + 640).astype(int), \
               ((pts[:, 1] / pts[:, 2]) * fx + 360).astype(int)

    def render(shift, out_img=None):
        img = out_img if out_img is not None else np.zeros((720, 1280, 3), np.uint8)
        img[:] = (20, 20, 20)
        xs, ys = px(pts, shift)
        for x, y, r in zip(xs, ys, radii):
            cv2.circle(img, (int(x), int(y)), int(r), (255, 255, 255), -1)
        return img

    left = render(0.0)
    right = render(0.3)
    # stereo inputs stay in memory; only the epipolar result goes on disk

    from src.geometry.epipolar import (draw_epipolar_lines, estimate_fundamental,
                                       match_features)
    _, _, p1, p2 = match_features(left, right)
    F, mask = estimate_fundamental(p1, p2)
    l, r = draw_epipolar_lines(left, right, p1[mask], p2[mask], F)
    cv2.imwrite(str(out_dir / "06_epipolar_lines.png"), np.hstack([l, r]))
    return int(mask.sum()), len(p1)


def save_real_model(out_dir: Path):
    """4b) REAL quantized-model inference (not stub boxes). Only runs when the
    exported models exist; the fresh COCO yolov8n natively knows 'stop sign'."""
    weight = "models/yolov8n_int8.onnx"
    photo = "data/samples/bus.jpg"
    if not (Path(weight).exists() and Path(photo).exists()):
        print(f"skip real-model demo ({weight} / {photo} missing)")
        return
    from src.detection.model import Detector
    det_fn = Detector(weights=weight, backend="onnx", conf=0.30)
    frame = cv2.imread(photo)
    dets = det_fn(frame)
    distances = estimate_distances(dets.names, dets.boxes, CAM)
    print("real INT8 onnx model on", photo, ":")
    for n, b, s, d in zip(dets.names, dets.boxes, dets.scores, distances):
        ds = format(d, ".1f") + "m" if d else "n/a (class not in height table)"
        print(f"  {n:12s} score {s:.2f}  box {b.astype(int).tolist()}  dist {ds}")
    draw_legend(draw_detections(frame, dets.boxes, dets.names, distances,
                                scores=dets.scores),
                ["real inference: yolov8n INT8 onnx (CPU)",
                 "distance only for the 3 robot classes"])
    cv2.imwrite(str(out_dir / "08_real_model_int8.jpg"), frame)


def save_test_log(out_dir: Path):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                       capture_output=True, text=True, cwd=str(ROOT))
    (out_dir / "00_test_log.txt").write_text(
        "$ python -m pytest tests -q\n" + r.stdout + r.stderr)
    return r.returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== demo outputs ->", OUT)

    code = save_test_log(OUT)
    print(f"pytest exit: {code} -> 00_test_log.txt")

    frame, boxes, names, dists = save_basic(OUT)
    print("01_detection_distance.jpg:", names,
          [f"{d:.1f}m" for d in dists])

    save_bev(frame, boxes, names, dists, OUT)
    print("02_birdeye_inset.jpg saved")

    save_real_model(OUT)

    dist_seq = save_tracking(OUT)
    print(f"03_tracking_final.jpg + 04_tracking_clip.mp4 "
          f"(distance went {dist_seq[0]:.1f}m -> {dist_seq[-1]:.1f}m)")

    inliers, total = save_stereo(OUT)
    print(f"06_epipolar_lines.png: {inliers}/{total} epipolar inliers")

    print("done")


if __name__ == "__main__":
    main()
