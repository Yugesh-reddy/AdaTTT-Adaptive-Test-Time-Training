"""Helpers behind results/writeup/numbers.json."""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "writeup_numbers", os.path.join(ROOT, "scripts", "writeup_numbers.py"))
wn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wn)


def test_mcnemar_exact_p_matches_the_blur_s3_count():
    assert wn.mcnemar_exact_p(17, 15) == pytest.approx(0.8601, abs=1e-4)
    assert wn.mcnemar_exact_p(0, 0) == 1.0
    assert wn.mcnemar_exact_p(0, 4) == pytest.approx(0.125)


def test_paired_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    delta = rng.normal(0.01, 0.1, size=4000)
    lo, hi = wn.paired_bootstrap_ci_pp(delta)
    assert (lo, hi) == tuple(wn.paired_bootstrap_ci_pp(delta))
    assert lo < 100 * delta.mean() < hi


def test_zero_delta_has_a_zero_width_interval():
    assert wn.paired_bootstrap_ci_pp(np.zeros(100)) == [0.0, 0.0]
