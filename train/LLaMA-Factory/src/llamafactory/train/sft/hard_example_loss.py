# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import math
from typing import Union

import torch
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX


DEFAULT_HARD_EXAMPLE_LOSS_ALPHA: float = 0.5
DEFAULT_HARD_EXAMPLE_LOSS_MAX_WEIGHT: float = 1.5


@dataclass(frozen=True)
class HardExampleLossDetails:
    r"""Diagnostic tensors produced by hard-example-weighted causal LM loss."""

    loss: torch.Tensor
    per_example_token_ce_sum: torch.Tensor
    valid_token_counts: torch.Tensor
    difficulties: torch.Tensor
    mean_difficulty: torch.Tensor
    weights: torch.Tensor


def validate_hard_example_loss_config(
    alpha: float = DEFAULT_HARD_EXAMPLE_LOSS_ALPHA,
    max_weight: float = DEFAULT_HARD_EXAMPLE_LOSS_MAX_WEIGHT,
) -> tuple[float, float]:
    r"""Validate `alpha` (>= 0.0) and `max_weight` (>= 1.0) for hard-example loss."""
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha < 0.0
    ):
        raise ValueError("`hard_example_loss_alpha` must be a finite number >= 0.0.")

    if (
        isinstance(max_weight, bool)
        or not isinstance(max_weight, (int, float))
        or not math.isfinite(max_weight)
        or max_weight < 1.0
    ):
        raise ValueError("`hard_example_loss_max_weight` must be a finite number >= 1.0.")

    return float(alpha), float(max_weight)


def compute_hard_example_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = DEFAULT_HARD_EXAMPLE_LOSS_ALPHA,
    max_weight: float = DEFAULT_HARD_EXAMPLE_LOSS_MAX_WEIGHT,
    ignore_index: int = IGNORE_INDEX,
    return_details: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, HardExampleLossDetails]]:
    r"""Compute bounded per-example difficulty-weighted shifted token cross-entropy loss.

    For each example `i` in the microbatch:
    - Shift causal targets (`logits[i, :-1]` predicts `labels[i, 1:]`).
    - Mask out prompt/padding positions where `shift_labels == ignore_index`.
    - Compute FP32 cross-entropy only over valid answer tokens of example `i` to avoid
      materializing a full-batch/full-sequence FP32 vocabulary tensor.
    - Detach per-example mean valid-token CE (`difficulty_i`) and compute relative
      difficulty against the microbatch mean over valid examples (`valid_count_i > 0`):
        `weight_i = clamp(1 + alpha * relu(difficulty_i / mean_difficulty - 1), 1, max_weight)`.
    - Return weighted token-mean loss:
        `sum_i(weight_i * sum_valid_token_ce_i) / sum_i(weight_i * count_valid_tokens_i)`.
    """
    alpha_f, max_weight_f = validate_hard_example_loss_config(alpha=alpha, max_weight=max_weight)

    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("`logits` and `labels` must be `torch.Tensor` instances.")

    if logits.ndim != 3:
        raise ValueError(f"`logits` must have shape (batch_size, seq_len, vocab_size), got {tuple(logits.shape)}.")

    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError(
            f"`labels` must have shape (batch_size, seq_len) matching `logits.shape[:2]` "
            f"{tuple(logits.shape[:2])}, got {tuple(labels.shape)}."
        )

    batch_size, seq_len, _ = logits.shape
    device = logits.device

    if batch_size == 0 or seq_len <= 1:
        zero_loss = logits.reshape(-1)[:0].float().sum()
        if return_details:
            details = HardExampleLossDetails(
                loss=zero_loss,
                per_example_token_ce_sum=torch.zeros(batch_size, device=device, dtype=torch.float32),
                valid_token_counts=torch.zeros(batch_size, device=device, dtype=torch.long),
                difficulties=torch.zeros(batch_size, device=device, dtype=torch.float32),
                mean_difficulty=torch.zeros((), device=device, dtype=torch.float32),
                weights=torch.ones(batch_size, device=device, dtype=torch.float32),
            )
            return zero_loss, details
        return zero_loss

    token_ce_sum_list: list[torch.Tensor] = []
    difficulty_list: list[torch.Tensor] = []
    valid_count_list: list[int] = []

    for i in range(batch_size):
        shift_labels_i = labels[i, 1:].to(device=device)
        valid_mask_i = shift_labels_i.ne(ignore_index)
        valid_count_i = int(valid_mask_i.sum().item())
        valid_count_list.append(valid_count_i)

        if valid_count_i > 0:
            valid_logits_i = logits[i, :-1, :][valid_mask_i].float()
            valid_targets_i = shift_labels_i[valid_mask_i].long()
            token_ce_sum_i = F.cross_entropy(valid_logits_i, valid_targets_i, reduction="sum")
            difficulty_i = token_ce_sum_i.detach() / float(valid_count_i)
        else:
            token_ce_sum_i = logits[i, :0, :0].float().sum()
            difficulty_i = torch.zeros((), device=device, dtype=torch.float32)

        token_ce_sum_list.append(token_ce_sum_i)
        difficulty_list.append(difficulty_i)

    if not any(count > 0 for count in valid_count_list):
        zero_loss = logits.reshape(-1)[:0].float().sum()
        if return_details:
            details = HardExampleLossDetails(
                loss=zero_loss,
                per_example_token_ce_sum=torch.zeros(batch_size, device=device, dtype=torch.float32),
                valid_token_counts=torch.zeros(batch_size, device=device, dtype=torch.long),
                difficulties=torch.zeros(batch_size, device=device, dtype=torch.float32),
                mean_difficulty=torch.zeros((), device=device, dtype=torch.float32),
                weights=torch.ones(batch_size, device=device, dtype=torch.float32),
            )
            return zero_loss, details
        return zero_loss

    with torch.no_grad():
        valid_token_counts = torch.tensor(valid_count_list, device=device, dtype=torch.long)
        valid_counts_f = valid_token_counts.to(dtype=torch.float32)
        difficulties = torch.stack(difficulty_list).detach()
        valid_example_mask = valid_token_counts > 0
        mean_difficulty = difficulties[valid_example_mask].mean().detach()

        if torch.isfinite(mean_difficulty) and mean_difficulty <= 0.0:
            weights = torch.ones_like(difficulties)
        else:
            relative_excess = F.relu(difficulties / mean_difficulty - 1.0)
            unclamped_weights = 1.0 + alpha_f * relative_excess
            weights = torch.clamp(unclamped_weights, min=1.0, max=max_weight_f)
            weights = torch.where(valid_example_mask, weights, torch.ones_like(weights))

        weights = weights.detach()

    per_example_token_ce_sum = torch.stack(token_ce_sum_list)
    weighted_token_ce_sum = (weights * per_example_token_ce_sum).sum()
    weighted_token_count = (weights * valid_counts_f).sum()
    loss = weighted_token_ce_sum / weighted_token_count

    if return_details:
        details = HardExampleLossDetails(
            loss=loss,
            per_example_token_ce_sum=per_example_token_ce_sum.detach(),
            valid_token_counts=valid_token_counts,
            difficulties=difficulties,
            mean_difficulty=mean_difficulty,
            weights=weights,
        )
        return loss, details

    return loss
