"""Selective-prediction helpers (scripts/abstention.py)."""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("abstention", os.path.join(ROOT, "scripts", "abstention.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _data(n=4000, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    p_correct = 1 - scores if signal else np.full(n, 0.5)
    correct = rng.random(n) < p_correct
    return scores, correct.astype(float), correct


def test_threshold_reaches_the_target_coverage_on_its_own_split():
    scores, _, _ = _data()
    for target in (0.9, 0.8):
        t = ab.threshold_for_coverage(scores, target)
        assert (scores <= t).mean() >= target


def test_abstaining_on_uncertain_questions_raises_accuracy():
    scores, soft, correct = _data()
    r = ab.selective(scores, soft, correct, ab.threshold_for_coverage(scores, 0.8))
    assert r["soft_gain_pp"] > 5 and r["soft_gain_ci95_pp"][0] > 0


def test_no_signal_gives_no_reliable_gain():
    scores, soft, correct = _data(signal=False)
    r = ab.selective(scores, soft, correct, ab.threshold_for_coverage(scores, 0.8))
    lo, hi = r["soft_gain_ci95_pp"]
    assert lo < 0 < hi


def test_risk_coverage_ends_at_full_accuracy():
    scores, soft, _ = _data()
    rc = ab.risk_coverage(scores, soft)
    assert rc["coverage"][-1] == 1.0
    assert rc["soft_answered"][-1] == pytest.approx(100 * soft.mean(), abs=1e-3)
    assert rc["soft_answered"][0] > rc["soft_answered"][-1]
