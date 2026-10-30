# Object Detection + Distance Estimation for Robotics Navigation

## Project Overview

This project detects navigation-relevant objects, estimates their distance
from a single camera, and evaluates optimizations for edge devices.

Target objects:

- Traffic cones
- Road barriers
- Stop signs

The required workflow is implemented with real image data, YOLOv8 transfer
learning, pinhole-camera distance estimation, quantization, pruning, and
CPU/GPU benchmarking.

## Contact (Name, Phone, Email)

- **Name:** Ratnajeet Patil
- **Phone:** 9422221740
- **Email:** ratnajeetpatil62@gmail.com

## Features

- YOLOv8n object detection
- Three unified classes: `traffic_cone`, `barrier`, `stop_sign`
- Transfer learning from pretrained YOLOv8n weights
- Distance annotation on each detection
- Real camera calibration utility
- ONNX FP32, FP16, and INT8 export
- L1 pruning
- CPU/GPU FPS and model-size benchmarking
- Accuracy comparison between model variants
- Optional real-input bird's-eye view
- Optional epipolar-geometry utilities
- Optional optical-flow tracking utilities

## Dataset

BDD100K is used as the primary real driving dataset. Real complementary
datasets are used for classes that are not consistently available in the
BDD100K labels:

- TraCon: traffic cones
- DORIE/barrier data: barriers
- GTSDB: stop signs

Verified local data:

- 2,000 BDD images
- 200 training images
- 50 validation images

The unified YOLO class mapping is:

```text
0  traffic_cone
1  barrier
2  stop_sign
```

The current validation split contains 173 cone instances and no barrier or
stop-sign instances. Its metrics therefore do not represent three-class
accuracy.

## Model & Training

The project uses YOLOv8n transfer learning:

```text
pretrained YOLOv8n -> fine-tuning on the merged real dataset
```

The trained checkpoint is:

```text
runs/detect/runs/train/bdd_finetune/weights/best.pt
```

The recorded validation metrics are:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.9772 | 0.8960 | 0.9476 | 0.8301 |

These are cone-only validation results because of the current validation
split composition.

## Installation & Usage

Install the Python dependencies, prepare the real dataset, train the model,
and run inference on a real image, video, folder, or camera.

Detection output includes the class, confidence, and estimated distance, for
example:

```text
Cone, 2.3m
Stop Sign, 3.2m
```

## Distance Estimation

Distance is estimated from the pinhole-camera relationship:

```text
Z = fy * H / h
```

Where:

- `Z` is the estimated distance in meters
- `fy` is the vertical focal length in pixels
- `H` is the known object height
- `h` is the detected bounding-box height in pixels

Configured reference heights:

| Class | Height |
|---|---:|
| traffic_cone | 0.50 m |
| barrier | 0.80 m |
| stop_sign | 0.60 m |

Camera intrinsics are stored in `configs/config.yaml`. The default intrinsics
are documented placeholders and should be replaced with calibration results
for deployment.

## Results & Benchmarks

Benchmark hardware:

- CPU: Intel Core i7-14700F
- GPU: NVIDIA GeForce RTX 5070

| Model | CPU FPS | GPU FPS | Size |
|---|---:|---:|---:|
| FP32 Torch | 31.7 | 177.6 | 6.3 MB |
| FP32 ONNX | 39.6 | 223.2 | 12.2 MB |
| INT8 ONNX | 26.8 | 123.8 | 4.9 MB |
| FP16 ONNX | 22.2 | 196.5 | 6.2 MB |

Pruning measured 29.9% sparsity with a 30% target.

Detailed reports:

- `results/benchmark.md`
- `results/pruning.md`
- `results/detection_metrics.md`
- `results/detection_metrics.json`

The included real-data example is:

`results/output_samples/real_dataset_detection_bev.jpg`

## Project Structure

```text
configs/              camera, class, and runtime configuration
scripts/              dataset, training, inference, evaluation, optimization
src/detection/        detector, dataset conversion, and training
src/distance/         pinhole distance estimation
src/geometry/         optional BEV, epipolar, and optical-flow utilities
src/optimization/     quantization, pruning, and benchmarking
tests/                unit and pipeline tests
results/              reports and real-data output
```

## Limitations & References

- The current validation split does not evaluate barrier or stop-sign
  instances.
- `results/ground_truth.csv` contains only its header, so no measured
  distance-error result is claimed.
- Camera intrinsics are placeholders until a real camera is calibrated.
- No stereo pair is currently included, so no epipolar result is claimed.
- No consecutive video is currently included, so no optical-flow result is
  claimed.
- Optional features must be run with real stereo images or real video frames.
- BDD100K: https://github.com/bdd100k/bdd100k
- TraCon: https://github.com/ikatsamenis/Cone-Detection
- DORIE: https://zenodo.org/records/17277466
- GTSDB: https://huggingface.co/datasets/GlowBond/GTSDB

## Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Build the merged real dataset
python scripts\build_real_dataset.py

# Train and validate
python scripts\train.py --data data\dataset\data.yaml `
  --base yolov8n.pt --epochs 50

# Validate an existing checkpoint
python scripts\train.py --validate-only --name bdd_finetune

# Run inference on a real image
python scripts\run_inference.py `
  --source "data\dataset\val\images\TraCon_dataset_cone (10).jpg" `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --output results\annotated.jpg --no-show

# Calibrate camera intrinsics
python scripts\calibrate.py --images data\calib_imgs `
  --pattern 12,9 --square 0.03

# Evaluate measured distance error
python scripts\evaluate_distance.py --gt results\ground_truth.csv

# Quantize and benchmark
python scripts\quantize.py `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --calib-dir data\dataset\train\images --all

# Prune
python scripts\prune.py `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --amount 0.3

# Compare model accuracy
python scripts\compare_accuracy.py --data data\dataset\data.yaml `
  --model fp32=runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --model int8=models\yolov8n_int8.onnx

# Optional real-image BEV
python scripts\real_demo.py `
  --source "data\dataset\val\images\TraCon_dataset_cone (10).jpg" `
  --weights runs\detect\runs\train\bdd_finetune\weights\best.pt `
  --bev --output results\real_dataset_detection_bev.jpg

# Optional epipolar geometry with a real stereo pair
python scripts\epipolar_demo.py `
  --left data\stereo\left.jpg --right data\stereo\right.jpg `
  --baseline 0.30 --out results\epipolar_real.png

# Run tests
python -m pytest -q
```
