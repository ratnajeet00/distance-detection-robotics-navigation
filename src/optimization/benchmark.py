"""FPS / latency benchmarking across backends and devices.

Produces a comparison table like:

    | Variant       | Device | Size MB | FPS    | ms/frame |
    |---------------|--------|---------|--------|----------|
    | FP32 torch    | CPU    | 19.6    | 34.1   | 29.3     |
    | INT8 onnx     | CPU    | 6.3     | 96.4   | 10.4     |

Write results to results/benchmark.md so "before vs. after" numbers are
recorded alongside the code.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    device: str
    size_mb: float
    fps: float
    ms_per_frame: float
    note: str = ""


def fps_of(fn: Callable[[], object], frames: int = 250, warmup: int = 25) -> tuple:
    """Time fn() over a loop. Returns (fps, ms_per_frame)."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(frames):
        fn()
    dt = time.perf_counter() - t0
    fps = frames / dt
    return fps, 1000.0 * dt / frames


def make_video_header(video_size=(1, 1)) -> None:
    """Compatibility no-op kept for older notebook calls."""
    return None


def bench_torch(weights: str, imgsz: int = 640, device: str = "cpu",
                frames: int = 150) -> BenchmarkResult:
    from ultralytics import YOLO

    # ultralytics takes "0"; torch itself needs "cuda"
    torch_dev = "cuda" if device in ("0", "cuda:0") else device
    model = YOLO(weights)
    model.model.to(torch_dev)
    if torch_dev.startswith("cuda"):
        model.model.half()
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, device=device, verbose=False)  # build graph

    self_size = Path(weights).stat().st_size / (1024 * 1024)
    fps, ms = fps_of(lambda: model.predict(dummy, device=device, verbose=False),
                     frames=frames)
    return BenchmarkResult(name=f"FP32 torch ({Path(weights).stem})",
                           device=("cuda" if torch_dev.startswith("cuda") else device),
                           size_mb=self_size, fps=fps, ms_per_frame=ms)


def bench_onnx(onnx_path: str, providers: List[str], imgsz: int = 640,
               frames: int = 250, label: str = None) -> BenchmarkResult:
    import onnxruntime as ort
    from src.detection.model import letterbox

    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual = "cpu"  # report what actually executed, not what we asked for
    for p in sess.get_providers():
        eps = p.replace("ExecutionProvider", "").strip()
        if eps and eps.lower() != "cpu":
            actual = eps.lower()
            break
    inp = sess.get_inputs()[0]
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    lb, _, _ = letterbox(dummy, imgsz)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]

    def run():
        sess.run(None, {inp.name: x})

    fps, ms = fps_of(run, frames=frames)
    return BenchmarkResult(name=label or Path(onnx_path).stem,
                           device=actual,
                           size_mb=Path(onnx_path).stat().st_size / (1024 * 1024),
                           fps=fps, ms_per_frame=ms)


def run_full_benchmark(
    torch_weights: str,
    onnx_paths: List[str],
    imgsz: int = 640,
    out_md: Optional[str] = "results/benchmark.md",
) -> List[BenchmarkResult]:
    """Benchmark torch on CPU (+GPU if available) and each given ONNX model
    on whichever ExecutionProviders onnxruntime offers (CPU, CUDA, TensorRT)."""
    import torch
    import onnxruntime as ort

    results: List[BenchmarkResult] = []
    results.append(bench_torch(torch_weights, imgsz, "cpu"))
    if torch.cuda.is_available():
        results.append(bench_torch(torch_weights, imgsz, "0"))
    ort_providers = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in ort_providers
    for p in onnx_paths:
        if not Path(p).exists():
            continue
        stem = Path(p).stem
        # CUDA when offered (auto-falls back internally if it can't init)
        if has_cuda:
            results.append(bench_onnx(p, ["CUDAExecutionProvider"], imgsz))
        # INT8/quantized variants are the candidate for CPU-only edge devices,
        # so always measure them on CPU too
        if has_cuda or "int8" in stem.lower():
            results.append(bench_onnx(p, ["CPUExecutionProvider"], imgsz))

    # write markdown table
    if out_md:
        out = Path(out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["## Before vs. after (640px, RTX-class machine here; re-measure on the edge device)",
                 "", "| Variant | Device | Size MB | FPS | ms/frame |",
                 "|---|---|---:|---:|---:|"]
        for r in results:
            lines.append(f"| {r.name} | {r.device} | {r.size_mb:6.2f} | "
                         f"{r.fps:6.1f} | {r.ms_per_frame:6.1f} |")
        out.write_text("\n".join(lines))
        print("\n".join(lines))
    return results
