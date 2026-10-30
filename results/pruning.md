# results - pruning (before vs after)

30% L1 magnitude pruning on yolov8n.pt, measured:

| model                     | size   | sparsity |
|---------------------------|--------|----------|
| yolov8n.pt (baseline)     | 6.6mb  | 0%       |
| yolov8n_pruned30.pt       | 12.9mb | **29.9%** |

one line: 30% of the smallest weights are now zero.

honest note: sparse weights only get FASTER if the runtime skips zero weights,
and the checkpoint stays dense on disk, so the file size went up - the win is
the sparsity number (and whatever fine-tuning is done after). for raw speed/size
on edge devices, the int8 quantization in `results/benchmark.md` is the bigger
and simpler win, so that's the recommended route.

pruning command: `python scripts/prune.py --weights yolov8n.pt --amount 0.3`
(re-run output: `yolov8n_pruned30.pt`, deleted from the repo after measuring)
