"""
Phase 2 reporting: sample_flops ladder (flag 58/93/128), compact npz, Pareto, δ/p.
"""

import json
import os

import numpy as np
import pytest
import torch

from ttt.gate import AdaptiveRouter
from ttt.phase2_report import (
    LEGACY_LADDER_GFLOPS,
    compact_outcomes,
    delta_over_p,
    oracle_recovery,
    save_outcomes_npz,
    sample_flops_ladder,
    write_phase2_report,
)


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    AdaptiveRouter.configure_for_backend("vit_bert")


def test_ladder_uses_sample_flops_and_flags_legacy_spec():
    AdaptiveRouter.configure_for_backend("clip")
    ladder = sample_flops_ladder("clip")
    base = AdaptiveRouter.sample_flops(adapted=False) / 1e9
    tent = AdaptiveRouter.sample_flops(adapted=True, n_aug=0, k_steps=1) / 1e9
    memo2 = AdaptiveRouter.sample_flops(adapted=True, n_aug=2, k_steps=1) / 1e9
    memo4 = AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1) / 1e9
    assert ladder["sample_flops_gflops"]["base"] == pytest.approx(base, rel=1e-6)
    assert ladder["sample_flops_gflops"]["tent_k1"] == pytest.approx(tent, rel=1e-6)
    assert ladder["sample_flops_gflops"]["memo2_k1"] == pytest.approx(memo2, rel=1e-6)
    assert ladder["sample_flops_gflops"]["memo4_k1"] == pytest.approx(memo4, rel=1e-6)
    assert ladder["legacy_ladder_gflops"] == LEGACY_LADDER_GFLOPS
    assert ladder["legacy_ladder_gflops"]["tent_k1"] == 58
    assert ladder["legacy_ladder_gflops"]["memo2_k1"] == 93
    assert ladder["legacy_ladder_gflops"]["memo4_k1"] == 128
    assert ladder["used_for_reporting"] == "sample_flops"
    assert "does not match" in ladder["legacy_ladder_note"].lower() or "gap" in ladder[
        "legacy_ladder_note"
    ].lower()
    # The CLIP numbers the handoff cites.
    assert tent == pytest.approx(43.4, abs=0.1)
    assert memo4 == pytest.approx(138.6, abs=0.2)


def test_oracle_recovery_picks_the_better_per_sample():
    base = np.array([1.0, 0.0, 0.3])
    adapted = np.array([0.2, 1.0, 0.4])
    out = oracle_recovery(base, adapted)
    assert out["oracle_soft"] == pytest.approx((1.0 + 1.0 + 0.4) / 3)
    assert out["base_soft"] == pytest.approx(base.mean())
    assert out["adapted_soft"] == pytest.approx(adapted.mean())


def test_delta_over_p():
    out = delta_over_p(base_acc=0.50, gated_acc=0.53, adapt_rate=0.10)
    assert out["delta"] == pytest.approx(0.03)
    assert out["p"] == pytest.approx(0.10)
    assert out["delta_over_p"] == pytest.approx(0.30)
    empty = delta_over_p(0.5, 0.5, 0.0)
    assert empty["delta_over_p"] is None


def test_compact_npz_is_small(tmp_path):
    n = 8000
    arrays = compact_outcomes(
        prediction=np.zeros(n, dtype=np.int32),
        ground_truth=np.ones(n, dtype=np.int32),
        soft_score=np.full(n, 0.3, dtype=np.float32),
        adapted=np.zeros(n, dtype=np.uint8),
        gate_score=np.linspace(0, 1, n, dtype=np.float32),
        flops_g=np.full(n, 24.8, dtype=np.float32),
    )
    path = tmp_path / "outcomes.npz"
    save_outcomes_npz(str(path), **arrays)
    size = os.path.getsize(path)
    # Fat Phase-1 JSON was ~135 B/sample; compact target is ~13 B/sample (~100 KB @ 8k).
    assert size < 200_000
    loaded = np.load(path)
    assert loaded["prediction"].shape == (n,)


def test_write_phase2_report_pareto_and_percentiles(tmp_path):
    AdaptiveRouter.configure_for_backend("clip")
    runs = [
        {
            "config": "no_adapt",
            "accuracy": 0.50,
            "soft": 0.54,
            "exact": 0.50,
            "avg_flops": 24.8,
            "flops": np.full(8, 24.8),
            "adapt_rate": 0.0,
            "method": "no_adapt",
        },
        {
            "config": "dense_memo",
            "accuracy": 0.56,
            "soft": 0.60,
            "exact": 0.55,
            "avg_flops": 138.6,
            "flops": np.full(8, 138.6),
            "adapt_rate": 1.0,
            "method": "memo",
        },
        {
            "config": "gated",
            "accuracy": 0.55,
            "soft": 0.59,
            "exact": 0.54,
            "avg_flops": 40.0,
            "flops": np.array([24.8, 24.8, 138.6, 24.8, 24.8, 138.6, 24.8, 24.8]),
            "adapt_rate": 0.25,
            "method": "gated_memo_sar",
        },
        {
            "config": "dominated",
            "accuracy": 0.51,
            "soft": 0.55,
            "exact": 0.50,
            "avg_flops": 90.0,
            "flops": np.full(8, 90.0),
            "adapt_rate": 0.5,
            "method": "tent",
        },
    ]
    out = write_phase2_report(runs, str(tmp_path), backend="clip")
    assert os.path.isfile(os.path.join(str(tmp_path), "summary.json"))
    assert os.path.isfile(os.path.join(str(tmp_path), "pareto.json"))
    assert os.path.isfile(os.path.join(str(tmp_path), "flops_ladder.json"))
    summary = json.load(open(os.path.join(str(tmp_path), "summary.json")))
    pareto = json.load(open(os.path.join(str(tmp_path), "pareto.json")))
    names = [p["config"] for p in pareto]
    assert "dominated" not in names
    assert "no_adapt" in names
    gated = next(r for r in summary["runs"] if r["config"] == "gated")
    assert "flops_p50" in gated and "flops_p95" in gated
    assert summary["flops_ladder"]["used_for_reporting"] == "sample_flops"
    assert out["n_runs"] == 4


def test_write_phase2_report_persists_oracle(tmp_path):
    AdaptiveRouter.configure_for_backend("clip")
    runs = [
        {
            "config": "no_adapt",
            "accuracy": 0.50,
            "soft": 0.50,
            "exact": 0.50,
            "avg_flops": 24.8,
            "flops": np.full(3, 24.8),
            "adapt_rate": 0.0,
            "method": "no_adapt",
        },
        {
            "config": "memo",
            "accuracy": 0.40,
            "soft": 0.40,
            "exact": 0.40,
            "avg_flops": 138.6,
            "flops": np.full(3, 138.6),
            "adapt_rate": 1.0,
            "method": "memo",
        },
    ]
    oracle = oracle_recovery(np.array([1.0, 0.0, 0.3]), np.array([0.2, 1.0, 0.4]))
    out = write_phase2_report(runs, str(tmp_path), backend="clip", oracle=oracle)
    assert out["oracle"]["oracle_soft"] == pytest.approx((1.0 + 1.0 + 0.4) / 3)
    landed = json.load(open(os.path.join(str(tmp_path), "oracle.json")))
    summary = json.load(open(os.path.join(str(tmp_path), "summary.json")))
    assert landed["oracle_soft"] == pytest.approx(out["oracle"]["oracle_soft"])
    assert summary["oracle"]["base_soft"] == pytest.approx(1.3 / 3)
