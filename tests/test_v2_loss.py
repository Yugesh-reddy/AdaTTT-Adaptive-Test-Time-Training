"""
The VQA term must dominate the gate term on the shared fusion trunk.

Phase 1 first trained with PyTorch's default mean-reduced soft BCE over 3129
answers. That shrinks the VQA term ~3129x, and the 0.1-weighted gate loss then
set the fusion's Adam normalizer: after the control's epoch 3 the fusion learned
VQA at ~3% of its nominal learning rate while validation stayed flat.
"""

import os

import torch
import torch.nn.functional as F

from ttt.losses import vqa_loss

TRAIN_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu", "train_base.py"
)
NUM_ANSWERS = 3129


def _one_hot_scores(answers, num_answers=NUM_ANSWERS):
    scores = torch.zeros(len(answers), num_answers)
    scores[torch.arange(len(answers)), answers] = 1.0
    return scores


def test_soft_bce_sums_over_answers_and_averages_over_batch():
    torch.manual_seed(0)
    logits = torch.randn(8, NUM_ANSWERS)
    scores = _one_hot_scores(torch.randint(0, NUM_ANSWERS, (8,)))
    per_sample = F.binary_cross_entropy_with_logits(logits, scores, reduction="none").sum(1)
    assert torch.allclose(vqa_loss(logits, None, scores), per_sample.mean())


def test_correct_answer_gradient_is_not_diluted_by_vocabulary_size():
    """At a trained operating point the positive logit gets ~1/B, as under v1's CE."""
    batch = 4
    answers = torch.zeros(batch, dtype=torch.long)
    logits = torch.full((batch, NUM_ANSWERS), -8.0, requires_grad=True)

    vqa_loss(logits, answers, _one_hot_scores(answers)).backward()
    soft_grad = logits.grad[:, 0].abs().mean().item()

    logits.grad = None
    F.cross_entropy(logits, answers).backward()
    ce_grad = logits.grad[:, 0].abs().mean().item()

    assert soft_grad > 0.5 * ce_grad
    assert soft_grad > 100 / (batch * NUM_ANSWERS)  # the mean-reduced bug gave 1/(B*C)


def test_cross_entropy_without_soft_scores():
    torch.manual_seed(0)
    logits = torch.randn(8, NUM_ANSWERS)
    answers = torch.randint(0, NUM_ANSWERS, (8,))
    assert torch.allclose(vqa_loss(logits, answers), F.cross_entropy(logits, answers))


def test_train_base_uses_the_scaled_loss():
    src = open(TRAIN_BASE).read()
    assert "vqa_loss(logits, answers, answer_scores)" in src
    assert "binary_cross_entropy_with_logits(logits, answer_scores)" not in src
