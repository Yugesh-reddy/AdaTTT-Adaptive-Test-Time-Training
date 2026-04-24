"""
Phase 2 gate: weighted score (not AND), τ on corruptions only, MaxProb AUROC/AURC.
"""

import numpy as np
import pytest
import torch

from ttt.score_gate import (
    ScoreWeights,
    aurc,
    binary_auroc,
    maxprob_metrics,
    route_adapt,
    signals_from_logits,
    tune_tau,
    weighted_score,
)


def test_weighted_score_is_not_an_and():
    """AND would drop a sample that fails one criterion; a sum still ranks it."""
    logits = torch.zeros(2, 4)
    # Sample 0: peaked (high maxprob, low entropy) — AND(entropy, 1-maxprob) is false.
    logits[0, 0] = 8.0
    # Sample 1: nearly uniform.
    logits[1] = 0.1

    sig = signals_from_logits(logits)
    weights = ScoreWeights(entropy=0.5, maxprob=0.5, margin=0.0)
    scores = weighted_score(sig, weights)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    # A low-entropy sample still has a strictly positive combined score.
    assert scores[0] > 0
    # Routing is a single τ on the sum, not a conjunction of thresholds.
    adapt = route_adapt(scores, tau=scores.mean().item())
    assert adapt.dtype == torch.bool
    assert adapt.tolist() == [False, True]


def test_maxprob_baseline_is_one_minus_maxprob():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sig = signals_from_logits(logits)
    scores = weighted_score(sig, ScoreWeights(entropy=0.0, maxprob=1.0, margin=0.0))
    assert scores[0].item() == pytest.approx(1.0 - sig["maxprob"][0].item(), abs=1e-5)
    assert scores[1] > scores[0]


def test_tune_tau_rejects_vqacp():
    scores = np.array([0.1, 0.9])
    base = np.array([0.5, 0.5])
    adapted = np.array([0.5, 0.8])
    with pytest.raises(ValueError, match="VQA-CP"):
        tune_tau(scores, base, adapted, source="vqa_cp")
    with pytest.raises(ValueError, match="VQA-CP"):
        tune_tau(scores, base, adapted, source="vqacp_test")


def test_tune_tau_on_corruption_picks_a_threshold():
    # High score = adapt. Adapting the second sample helps, the first hurts.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    base = np.array([1.0, 1.0, 0.0, 0.0])
    adapted = np.array([0.0, 0.0, 1.0, 1.0])
    chosen = tune_tau(scores, base, adapted, source="gaussian_blur_s3")
    assert "tau" in chosen
    assert chosen["source"] == "gaussian_blur_s3"
    # With a τ between 0.2 and 0.8 we adapt only the last two: accuracy 1.0.
    assert chosen["gated_metric"] == pytest.approx(1.0)
    assert chosen["adapt_rate"] == pytest.approx(0.5)


def test_auroc_ranks_a_perfect_score():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert binary_auroc(labels, scores) == pytest.approx(1.0)
    assert binary_auroc(labels, -scores) == pytest.approx(0.0)


def test_aurc_is_lower_for_a_better_ranking():
    correct = np.array([1, 1, 1, 0], dtype=float)
    good = aurc(confidence=np.array([0.9, 0.8, 0.7, 0.1]), correct=correct)
    bad = aurc(confidence=np.array([0.1, 0.2, 0.3, 0.9]), correct=correct)
    assert good < bad


def test_maxprob_metrics_on_known_logits():
    logits = torch.tensor(
        [
            [10.0, 0.0],  # confident, correct
            [10.0, 0.0],  # confident, wrong
            [0.0, 0.0],  # unconfident, correct
            [0.0, 0.0],  # unconfident, wrong
        ]
    )
    correct = np.array([True, False, True, False])
    metrics = maxprob_metrics(logits, correct)
    assert "auroc" in metrics and "aurc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["aurc"] >= 0.0
