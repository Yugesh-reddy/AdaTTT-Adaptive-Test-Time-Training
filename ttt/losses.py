"""
Training losses for the base VQA model.

The soft-BCE term is summed over answers and averaged over the batch, the
standard VQA scaling (bottom-up-attention-vqa multiplies the mean by the number
of answers). PyTorch's default mean over all 3129 answers shrinks it ~3129x.
At that scale the 0.1-weighted gate loss outweighed it ~19x on the shared
fusion trunk, and the fusion learned VQA at ~3% of its nominal learning rate
(measured on the Phase 1 vit_bert control after epoch 3).
"""

from typing import Optional

import torch
import torch.nn.functional as F


def vqa_loss(
    logits: torch.Tensor,
    answers: torch.Tensor,
    answer_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """VQA term: soft-label BCE when soft scores are given, else cross-entropy.

    Args:
        logits: (B, num_answers)
        answers: (B,) mode-answer indices, used by cross-entropy.
        answer_scores: (B, num_answers) soft targets min(count / 3, 1), or None.

    Returns:
        Scalar loss. Both branches put O(1/B) gradient on the correct answer's
        logit, so the gate loss weight means the same thing under either.
    """
    if answer_scores is None:
        return F.cross_entropy(logits, answers)
    return F.binary_cross_entropy_with_logits(
        logits, answer_scores, reduction="sum"
    ) / logits.size(0)
