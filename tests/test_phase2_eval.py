"""
Phase 2 eval loop on a multi-view cache: methods, gating, sample_flops, no UNK credit.
"""

import numpy as np
import pytest
import torch

from ttt.gate import AdaptiveRouter
from ttt.models import FullVQAModel
from ttt.phase2_eval import evaluate_condition
from ttt.score_gate import ScoreWeights
from ttt.tta import TTAAdapter


def _config():
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 1,
        "fusion_dropout": 0.0,
        "prediction_hidden": 32,
        "num_answers": 8,
        "gate_hidden": 16,
        "num_query_tokens": 1,
        "text_dim": 512,
        "ttt_lr": 1e-2,
        "ttt_grad_clip": 1.0,
        "ttt_adapt_modules": ["fusion"],
        "encoder_backend": "clip",
    }


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    AdaptiveRouter.configure_for_backend("vit_bert")


def _samples(n=4, n_answers=8):
    torch.manual_seed(1)
    scores = torch.zeros(n, n_answers)
    # Official soft: no mass on index 0 (<UNK>).
    for i in range(n):
        scores[i, (i % (n_answers - 1)) + 1] = 1.0
    return {
        "visual_tokens": torch.randn(n, 5, 197, 768),
        "text_tokens": torch.randn(n, 20, 512),
        "attention_mask": torch.ones(n, 20, dtype=torch.long),
        "answer_idx": torch.tensor([(i % (n_answers - 1)) + 1 for i in range(n)]),
        "answer_scores": scores,
        "sample_ids": [str(i) for i in range(n)],
        "question_types": ["other"] * n,
    }


def test_no_adapt_never_sets_adapted_flag():
    model = FullVQAModel(_config())
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    out = evaluate_condition(model, _samples(), method="no_adapt")
    assert out["adapted"].sum() == 0
    assert np.allclose(out["flops_g"], AdaptiveRouter.sample_flops(False) / 1e9)
    assert (out["soft_score"] >= 0).all()
    assert "maxprob_auroc" in out
    assert "maxprob_aurc" in out


def test_dense_memo_adapts_every_sample_and_restores():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    named = model.get_ttt_params_named(adapt_modules=["fusion"], layernorm_only=True)
    before = {n: p.detach().clone() for n, p in named}
    adapter = TTAAdapter(model, cfg, method="memo", k_steps=1, n_aug=4)
    out = evaluate_condition(model, _samples(), method="memo", adapter=adapter)
    assert out["adapted"].all()
    expected = AdaptiveRouter.sample_flops(True, n_aug=4, k_steps=1) / 1e9
    assert np.allclose(out["flops_g"], expected)
    for n, p in named:
        assert torch.equal(p.detach(), before[n])


def test_gated_mixes_skip_and_adapt_flops():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    adapter = TTAAdapter(model, cfg, method="memo_sar", k_steps=1, n_aug=4, sar_e0_ratio=1.0)
    out = evaluate_condition(
        model,
        _samples(),
        method="gated_memo_sar",
        adapter=adapter,
        tau=0.0,
        weights=ScoreWeights(entropy=0.5, maxprob=0.5, margin=0.0),
    )
    # tau=0 adapts everyone whose score >= 0, which is all of them.
    skip = AdaptiveRouter.sample_flops(False) / 1e9
    adapt = AdaptiveRouter.sample_flops(True, n_aug=4, k_steps=1) / 1e9
    got = {round(float(x), 5) for x in out["flops_g"]}
    assert got.issubset({round(skip, 5), round(adapt, 5)})


def test_unk_index_gets_zero_soft_credit():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    samples = _samples()
    samples["answer_scores"][:, 0] = 0.0
    out = evaluate_condition(model, samples, method="no_adapt")
    unk = out["prediction"] == 0
    if unk.any():
        assert np.allclose(out["soft_score"][unk], 0.0)
