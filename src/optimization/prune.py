"""Magnitude-based weight pruning.

Global L1 unstructured pruning on Conv/Linear layers: we remove the weights
with the smallest absolute magnitude, targeting a global sparsity fraction
(``amount``). After pruning you should fine-tune for a few epochs to recover
accuracy; the benchmark script then shows the before/after FPS & mAP table.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)


def apply_global_l1_pruning(pt_model, amount: float = 0.3) -> Dict[str, float]:
    """Prune a torch nn.Module globally by L1 magnitude. Returns stats."""
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    targets = []
    for m in pt_model.modules():
        is_conv_or_linear = isinstance(m, (nn.Conv2d, nn.Linear))
        # YOLO re-parametrization layers carry .weight too
        if is_conv_or_linear and hasattr(m, "weight") and m.weight is not None:
            if not getattr(m, "training", True):
                m.train()
            targets.append(m)
    for m in targets:
        prune.l1_unstructured(m, name="weight", amount=amount)

    torch = __import__("torch")
    zeros = total = 0
    for m in targets:
        t = m.weight.detach().cpu().numpy()
        zeros += int((np.abs(t) < 1e-12).sum())
        total += t.size
    stats = {
        "sparsity": (zeros / total) if total else 0.0,
        "pruned_modules": len(targets),
    }
    # make pruning permanent + strip autograd hooks
    for m in targets:
        prune.remove(m, "weight")
    log.info("pruning: sparsity=%.2f%% over %d modules",
             100 * stats["sparsity"], stats["pruned_modules"])
    return stats


def prune_yolo(weights: str, amount: float = 0.3, size_variant: str = None) -> str:
    """Prune a saved YOLO .pt checkpoint and save as *_pruned.pt next to it.

    Note: after pruning, re-validate weights; if mAP drops too much, run a few
    fine-tuning epochs (scripts/prune.ps1 --fine-tune) to recover accuracy.
    """
    import torch
    from ultralytics import YOLO

    yolo = YOLO(weights)
    model = yolo.model
    stats = apply_global_l1_pruning(model, amount=amount)
    out = Path(weights).with_name(Path(weights).stem + f"_pruned{int(amount*100)}.pt")
    torch.save({"model": model, "names": yolo.names, "sparsity": stats["sparsity"]},
               str(out))
    log.info("saved pruned checkpoint: %s", out)
    return str(out)
