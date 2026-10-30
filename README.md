# Object Detection + Distance Estimation for Robotics Navigation

Detect cones, barriers, and stop signs, estimate their distance from one
camera, and test edge-device optimizations.

## What is implemented

### Detection

- YOLOv8n transfer learning
- BDD100K real driving images
- Real complementary data:
  - TraCon: cones
  - DORIE/barrier data: barriers
  - GTSDB data: stop signs
- Unified classes:

```text
0 traffic_cone
1 barrier
2 stop_sign
```

Verified local data:

```text
BDD images:       2,000
Training images:    200
Validation images:  50
```

Build and train:

```powershell
python scripts\build_real_dataset.py
python scripts\train.py --data data\dataset\data.yaml `
  --base yolov8n.pt --epochs 50
```

The trained checkpoint is:

```text
runs/detect/runs/train/bdd_finetune/weights/best.pt
```

The current validation split has 173 cone instances and no barrier or
stop-sign instances. Therefore the following values are cone-only validation
results, not three-class results:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.9772 | 0.8960 | 0.9476 | 0.8301 |

Reports:

- `results/detection_metrics.json`
- `results/detection_metrics.md`

### Distance estimation

Distance uses the pinhole-camera equation:

```text
Z = fy * H / h
```

- `fy`: vertical focal length in pixels
- `H`: known object height
- `h`: detected bounding-box height in pixels

Configured reference heights:

| Class | Height |
|---|---:|
| traffic_cone | 0.50 m |
| barrier | 0.80 m |
| stop_sign | 0.60 m |

The camera values are in `configs/config.yaml`. They are documented
placeholders until a real camera is calibrated:

```powershell
python scripts\calibrate.py --images data\calib_imgs `
  --pattern 12,9 --square 0.03
```

Run detection and annotation on a real image:

```powershell
python scripts\run_inference.py `
  --source "data\dataset\val\images\TraCon_dataset_cone (10).jpg" `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --output results\annotated.jpg --no-show
```

Boxes include class, confidence, and distance, for example:

```text
Cone, 2.3m
Stop Sign, 3.2m
```

Real distance-error evaluation requires measured tape, laser, or known-position
data in `results/ground_truth.csv`. That file currently contains only its
header, so no real distance-error result is claimed.

### Optimization

Quantization:

```powershell
python scripts\quantize.py `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --calib-dir data\dataset\train\images --all
```

Pruning:

```powershell
python scripts\prune.py `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --amount 0.3
```

Benchmark:

```powershell
python scripts\quantize.py `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --calib-dir data\dataset\train\images --benchmark
```

Recorded benchmark hardware: Intel i7-14700F CPU and NVIDIA RTX 5070 GPU.

| Model | CPU FPS | GPU FPS | Size |
|---|---:|---:|---:|
| FP32 Torch | 31.7 | 177.6 | 6.3 MB |
| FP32 ONNX | 39.6 | 223.2 | 12.2 MB |
| INT8 ONNX | 26.8 | 123.8 | 4.9 MB |
| FP16 ONNX | 22.2 | 196.5 | 6.2 MB |

Pruning measured 29.9% sparsity with a 30% target. Full details are in
`results/benchmark.md` and `results/pruning.md`.

Compare accuracy on the same validation split:

```powershell
python scripts\compare_accuracy.py --data data\dataset\data.yaml `
  --model fp32=runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --model int8=models\yolov8n_int8.onnx
```

## Optional extra credit

### Homography / bird's-eye view

This is run on a real validation image:

```powershell
python scripts\real_demo.py `
  --source "data\dataset\val\images\TraCon_dataset_cone (10).jpg" `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --bev --output results\real_dataset_detection_bev.jpg
```

The included output is
`results/output_samples/real_dataset_detection_bev.jpg`.

### Epipolar geometry

The implementation requires two real stereo images and a measured baseline:

```powershell
python scripts\epipolar_demo.py `
  --left data\stereo\left.jpg --right data\stereo\right.jpg `
  --baseline 0.30 --out results\epipolar_real.png
```

No stereo pair is currently in the repository, so no stereo result is
claimed.

### Optical flow

The implementation is in `src/geometry/optical_flow.py` and requires real
consecutive video frames. No video is currently in the repository, so no
tracking result is claimed.

## Tests

```powershell
python -m pytest -q
```

Current result: 25 tests passed.
