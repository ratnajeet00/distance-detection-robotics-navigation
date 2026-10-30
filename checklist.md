# submission checklist

- [x] Model output — detections + distance on the images/boxes, e.g. `Cone, 3.6m`
      → `results/output_samples/01_detection_distance.jpg`, `08_real_model_int8.jpg`
- [x] Evaluation metrics — tests for the geometry/tracker/pipeline all pass (15/15)
      → `results/output_samples/00_test_log.txt`; model mAP once trained: `scripts/train.py --validate-only`
- [x] Visualizations — fps cpu/gpu table, birds-eye view, epipolar lines, tracking clip
      → `07_benchmark.md`, `02_birdeye_inset.jpg`, `06_epipolar_lines.png`, `04_tracking_clip.mp4`
- [x] Results report — quantization before/after + pruning sparsity, written down simply
      → `results/benchmark.md` + the summary table in `README.md`
- [x] Saved model — fp32 / fp16 / int8 onnx checkpoints for the COCO-pretrained model
      → `models/` (the fine-tuned `.pt` appears here once the training run is done)
- [x] Demo output — image, video clip and a real-model inference sample in one folder
      → `results/output_samples/` (regenerate anytime: `python scripts/make_demo_output.py`)
- [x] Reproducible execution — every command + expected output is in `README.md` "how to run",
      and `scripts/make_demo_output.py` rebuilds all demos from scratch
