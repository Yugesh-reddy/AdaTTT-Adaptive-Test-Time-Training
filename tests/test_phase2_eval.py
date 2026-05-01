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


# --- τ provenance: the eval 8k is report-only --------------------------------

import argparse
import json
import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu"))
import eval_phase2  # noqa: E402


def _argv(*extra):
    return ["--features", "/content/x.pt", "--checkpoint", "ck.pt", *extra]


def test_gated_without_a_tau_source_is_refused():
    with pytest.raises(SystemExit) as err:
        eval_phase2.main(_argv("--methods", "no_adapt", "memo_sar", "gated_memo_sar"))
    assert err.value.code == 2


def test_fit_tau_needs_the_adapter_the_gate_runs():
    with pytest.raises(SystemExit) as err:
        eval_phase2.main(_argv("--methods", "no_adapt", "memo", "--fit-tau"))
    assert err.value.code == 2


def test_tau_file_fit_on_the_reported_source_is_refused(tmp_path):
    tau = tmp_path / "tau.json"
    tau.write_text(json.dumps({"tau": 0.4, "source": "corruption_gaussian_noise_s5"}))
    with pytest.raises(SystemExit) as err:
        eval_phase2.main(_argv("--methods", "no_adapt", "gated_memo_sar",
                               "--source", "corruption_gaussian_noise_s5",
                               "--tau-file", str(tau)))
    assert err.value.code == 2


def test_tau_file_from_gate_train_is_held_out(tmp_path):
    tau = tmp_path / "tau.json"
    tau.write_text(json.dumps({"tau": 0.4, "source": "gate_train_corruption_gaussian_noise_s5",
                               "target_method": "memo_sar"}))
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(methods=["no_adapt", "gated_memo_sar"], tau=None,
                              tau_file=str(tau), fit_tau=False,
                              source="corruption_gaussian_noise_s5")
    info = eval_phase2.resolve_tau_protocol(args, parser)
    assert info["protocol"] == "held_out"
    assert info["tuned_on"] == "gate_train_corruption_gaussian_noise_s5"
    assert info["tau"] == pytest.approx(0.4)


def test_in_sample_fit_is_labelled():
    rec = eval_phase2.fit_tau_record(np.array([0.1, 0.9]), np.array([1.0, 0.0]),
                                     np.array([1.0, 1.0]), source="corruption_x",
                                     reported_here=True)
    assert rec["protocol"] == "in_sample" and rec["target_method"] == "memo_sar"
    fit = eval_phase2.fit_tau_record(np.array([0.1, 0.9]), np.array([1.0, 0.0]),
                                     np.array([1.0, 1.0]), source="gate_train_x",
                                     reported_here=False)
    assert fit["protocol"] == "fit"


def test_tau_fit_at_another_step_size_is_refused(tmp_path):
    tau = tmp_path / "tau.json"
    tau.write_text(json.dumps({"tau": 0.4, "source": "gate_train_corruption_gaussian_noise_s5",
                               "lr": 0.01}))
    with pytest.raises(SystemExit) as err:
        eval_phase2.main(_argv("--methods", "no_adapt", "gated_memo_sar",
                               "--source", "corruption_gaussian_noise_s5",
                               "--tau-file", str(tau), "--lr", "0.001"))
    assert err.value.code == 2


# --- per-sample gate signals ----------------------------------------------------

from ttt.phase2_eval import SIGNAL_TIERS  # noqa: E402


def _signal_names():
    return [name for tier in SIGNAL_TIERS.values() for name in tier]


def test_memo_run_records_every_signal_tier():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    adapter = TTAAdapter(model, cfg, method="memo", k_steps=1, n_aug=4)
    out = evaluate_condition(model, _samples(), method="memo", adapter=adapter, signals=True)
    sig = out["signals"]
    assert sorted(sig) == sorted(_signal_names())
    assert all(len(v) == 4 for v in sig.values())
    assert np.all(np.isfinite(sig["post_entropy_drop"]))  # every sample adapted
    assert set(np.unique(sig["probe_agree"])) <= {0.0, 1.0}
    assert np.all((sig["views_agree_frac"] >= 0) & (sig["views_agree_frac"] <= 1))
    assert np.allclose(sig["maxprob"], out["maxprob"])


def test_skip_run_leaves_post_signals_empty():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    out = evaluate_condition(model, _samples(), method="no_adapt", signals=True)
    for name in SIGNAL_TIERS["post"]:
        assert np.all(np.isnan(out["signals"][name]))
    assert np.all(np.isfinite(out["signals"]["probe_kl"]))


def test_signals_are_off_by_default():
    cfg = _config()
    model = FullVQAModel(cfg)
    model.eval()
    AdaptiveRouter.configure_for_backend("clip")
    assert "signals" not in evaluate_condition(model, _samples(), method="no_adapt")
