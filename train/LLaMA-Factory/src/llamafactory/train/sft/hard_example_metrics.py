"""Detached, example-counted weight statistics for each training log window."""

import math

import torch
import torch.distributed as dist


class HardExampleWeightStats:
    def __init__(self):
        self._totals = None

    @torch.no_grad()
    def update(self, weights: torch.Tensor, valid_token_counts: torch.Tensor):
        weights = weights.detach()[valid_token_counts > 0]
        if self._totals is None:
            self._totals = torch.zeros(5, dtype=torch.float64, device=weights.device)
        if not weights.numel():
            return
        values = torch.stack([
            weights.sum(dtype=torch.float64),
            weights.new_tensor(weights.numel(), dtype=torch.float64),
            (weights > 1.1).sum(dtype=torch.float64),
            (weights > 1.3).sum(dtype=torch.float64),
            weights.max().to(torch.float64),
        ])
        self._totals[:4].add_(values[:4])
        self._totals[4] = torch.maximum(self._totals[4], values[4])

    @torch.no_grad()
    def take(self, device):
        totals = (self._totals.clone() if self._totals is not None
                  else torch.zeros(5, dtype=torch.float64, device=device))
        # Every rank participates, even if it had no valid answers in this window.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals[:4], op=dist.ReduceOp.SUM)
            dist.all_reduce(totals[4:], op=dist.ReduceOp.MAX)
        self._totals = None
        weight_sum, count, gt_11, gt_13, maximum = totals.cpu().tolist()
        if not count:
            return {}
        finite = math.isfinite(weight_sum)
        return {
            "mean_weight": weight_sum / count,
            "pct_weight_gt_1.1": 100.0 * gt_11 / count if finite else float("nan"),
            "pct_weight_gt_1.3": 100.0 * gt_13 / count if finite else float("nan"),
            "max_weight": maximum if finite else float("nan"),
            "weight_samples": count,
        }
