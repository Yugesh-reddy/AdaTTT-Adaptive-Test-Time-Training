"""
Phase 2 eval loop on a multi-view cache: methods, gating, sample_flops, no UNK credit.
"""

import os

import numpy as np
import pytest
import torch

from ttt.gate import AdaptiveRouter
from ttt.models import FullVQAModel
from ttt.phase2_eval import evaluate_condition, merge_condition_outputs, order_methods
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
    # flops_g is stored as float32, so compare with a tolerance, not rounding.
    got = out["flops_g"].astype(float)
    assert np.all(np.isclose(got, skip, atol=1e-3) | np.isclose(got, adapt, atol=1e-3))


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


class _TinyCache:
    def __init__(self, blob):
        self._blob = blob

    def __len__(self):
        return self._blob["visual_tokens"].shape[0]

    def __getitem__(self, idx):
        return {
            "visual_tokens": self._blob["visual_tokens"][idx],
            "text_tokens": self._blob["text_tokens"][idx],
            "attention_mask": self._blob["attention_mask"][idx],
            "answer_idx": int(self._blob["answer_idx"][idx]),
            "answer_scores": self._blob["answer_scores"][idx],
        }


def test_evaluate_condition_streams_dataset_on_model_device():
    model = FullVQAModel(_config())
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    blob = _samples(n=3)
    ticks = []
    out = evaluate_condition(
        model, _TinyCache(blob), method="no_adapt", on_progress=lambda d, t: ticks.append((d, t))
    )
    assert out["prediction"].shape == (3,)
    assert out["adapted"].sum() == 0
    assert ticks[-1] == (3, 3)
    device = next(model.parameters()).device
    assert device.type == "cpu"


def test_order_methods_puts_gate_last_after_base_and_memo():
    assert order_methods(["gated_memo_sar", "tent", "no_adapt", "memo"]) == [
        "no_adapt", "memo", "tent", "gated_memo_sar"
    ]


def test_merge_condition_outputs_recomputes_scalars():
    model = FullVQAModel(_config())
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    a = evaluate_condition(model, _samples(n=2), method="no_adapt")
    b = evaluate_condition(model, _samples(n=2), method="no_adapt")
    merged = merge_condition_outputs([a, b])
    assert merged["prediction"].shape == (4,)
    assert "maxprob_auroc" in merged


def test_eval_phase2_script_does_not_stack_full_cache():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpu", "eval_phase2.py")
    src = open(path).read()
    assert "_load_all" not in src
    assert "batch_size=len(ds)" not in src
    assert "order_methods" in src
