#!/usr/bin/env python
"""Quantize the detector for edge devices and benchmark before/after.

Steps
-----
1. export torch weights -> ONNX FP32
2. static INT8 quantization using calibration images
3. (optional) FP16 conversion
4. benchmark FPS table (CPU / GPU) -> results/benchmark.md

Example
-------
python scripts/quantize.py --weights runs/train/bdd_finetune/weights/best.pt \
    --calib-dir data/dataset/train/images --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.benchmark import run_full_benchmark
from src.optimization.quantize import convert_fp16, export_onnx, model_size_mb, quantize_int8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, required=True,
                    help="trained torch .pt checkpoint")
    ap.add_argument("--calib-dir", type=str, required=True,
                    help="folder of images used for INT8 calibration")
    ap.add_argument("--out-dir", type=str, default="models")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--export", action="store_true", help="export ONNX FP32")
    ap.add_argument("--int8", action="store_true", help="static INT8 PTQ")
    ap.add_argument("--fp16", action="store_true", help="FP16 conversion")
    ap.add_argument("--benchmark", action="store_true", help="run FPS comparison")
    ap.add_argument("--all", action="store_true", help="do every step")
    args = ap.parse_args()

    do_export = args.all or args.export
    do_int8 = args.all or args.int8
    do_fp16 = args.all or args.fp16
    do_bench = args.all or args.benchmark

    fp32 = str(Path(args.out_dir) / (Path(args.weights).stem + ".onnx"))
    int8 = str(Path(args.out_dir) / (Path(args.weights).stem + "_int8.onnx"))
    fp16 = str(Path(args.out_dir) / (Path(args.weights).stem + "_fp16.onnx"))

    if do_export and not Path(fp32).exists():
        fp32 = export_onnx(args.weights, imgsz=args.imgsz, out_dir=args.out_dir)
    if do_int8:
        quantize_int8(fp32, args.calib_dir, int8, imgsz=args.imgsz)
        print(f"FP32 vs INT8 size: {model_size_mb(fp32):.1f} MB "
              f"-> {model_size_mb(int8):.1f} MB")
    if do_fp16:
        convert_fp16(fp32, fp16)
    if do_bench:
        paths = [fp32 if Path(fp32).exists() else None,
                 int8 if Path(int8).exists() else None,
                 fp16 if Path(fp16).exists() else None]
        run_full_benchmark(args.weights, [p for p in paths if p],
                           imgsz=args.imgsz, out_md="results/benchmark.md")


if __name__ == "__main__":
    main()
