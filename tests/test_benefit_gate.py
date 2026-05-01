"""Session E benefit gate: selection rule, stop rule, costs, and the eval seal."""

import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

from ttt import benefit_gate as bg
from ttt.gate import AdaptiveRouter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    AdaptiveRouter.configure_for_backend("vit_bert")


def _synthetic(n=3000, signal=True, seed=0):
    """20% of samples change; if `signal`, low maxprob predicts that MEMO helps."""
    rng = np.random.default_rng(seed)
    data = {name: rng.random(n) for name in bg.POST}
    data["probe_agree"] = (rng.random(n) > 0.5).astype(float)
    data["post_answer_changed"] = (rng.random(n) > 0.5).astype(float)
    data["gate_score"] = rng.random(n)
    changed = rng.random(n) < 0.2
    helped = data["maxprob"] < 0.5 if signal else rng.random(n) < 0.5
    gain = np.where(changed, np.where(helped, 1 / 3, -1 / 3), 0.0)
    data["skip_soft"] = np.full(n, 0.5)
    data["memo_soft"] = 0.5 + gain
    data["skip_correct"] = np.zeros(n, dtype=bool)
    data["memo_correct"] = gain > 0
    return data


def test_threshold_never_adapts_when_nothing_helps():
    assert bg.choose_threshold(np.array([0.1, 0.9]), np.array([-1.0, -0.5])) == np.inf


def test_threshold_keeps_only_the_helpful_top():
    t = bg.choose_threshold(np.array([0.9, 0.8, 0.1]), np.array([1.0, 1.0, -1.0]))
    assert t == pytest.approx(0.8)


def test_cheapest_near_best_gate_wins_when_a_free_signal_exists():
    cv = {name: bg.cross_validate(name, _synthetic()) for name in bg.GATES}
    assert cv["free"]["gain_pp"] > 1.0 and cv["free"]["auroc_helped_vs_hurt"] > 0.9
    assert cv["score"]["gain_pp"] < cv["free"]["gain_pp"]
    chosen, record = bg.select_gate(cv)
    assert chosen == "free" and record["proceed"]


def test_stop_rule_fires_on_pure_noise():
    cv = {name: bg.cross_validate(name, _synthetic(signal=False, n=6000)) for name in bg.GATES}
    chosen, record = bg.select_gate(cv)
    assert chosen is None and record["proceed"] is False


def test_frozen_gate_reproduces_its_training_decisions():
    data = _synthetic()
    gate = bg.fit_final("free", data)
    adapt = bg.apply_gate(json.loads(json.dumps(gate)), data)  # survives a JSON round trip
    assert adapt.any() and (data["memo_soft"][adapt] >= data["skip_soft"][adapt]).mean() > 0.7


def test_gate_costs_follow_the_signal_tier():
    none = np.zeros(4, dtype=bool)
    assert bg.per_sample_gflops("free", none) == pytest.approx(np.full(4, 24.808))
    assert bg.per_sample_gflops("one_view", none) == pytest.approx(np.full(4, 24.808 + 23.808))
    assert bg.per_sample_gflops("post", none) == pytest.approx(np.full(4, 175.816))
    assert bg.per_sample_gflops("four_views", np.ones(4, dtype=bool)) == pytest.approx(np.full(4, 175.816))


def _write_part(directory, data):
    os.makedirs(directory, exist_ok=True)
    n = len(data["skip_soft"])
    gt = np.zeros(n, dtype=np.uint16)
    np.savez(os.path.join(directory, "no_adapt.npz"), soft_score=data["skip_soft"],
             prediction=np.where(data["skip_correct"], 0, 1).astype(np.uint16),
             ground_truth=gt, gate_score=data["gate_score"])
    np.savez(os.path.join(directory, "memo.npz"), soft_score=data["memo_soft"],
             prediction=np.where(data["memo_correct"], 0, 1).astype(np.uint16), ground_truth=gt)
    np.savez(os.path.join(directory, "memo_signals.npz"), **{k: data[k] for k in bg.POST})


def test_fit_script_refuses_the_sealed_split(tmp_path):
    fit = _script("phase2_gate_fit")
    with pytest.raises(SystemExit) as err:
        fit.main(["--part", str(tmp_path / "eval_sealed"), "--spec", str(tmp_path / "s.json")])
    assert err.value.code == 2


def test_fit_then_score_requires_a_committed_spec(tmp_path):
    fit, score = _script("phase2_gate_fit"), _script("phase2_gate_score")
    _write_part(tmp_path / "gate_train", _synthetic(seed=1))
    _write_part(tmp_path / "eval_sealed", _synthetic(seed=2))
    spec = tmp_path / "spec.json"
    assert fit.main(["--part", str(tmp_path / "gate_train"), "--spec", str(spec)]) == 0
    frozen = json.load(open(spec))
    assert frozen["proceed"] and frozen["gate"]["name"] == "free"

    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert not score.spec_committed(str(spec), str(tmp_path))
    subprocess.run(git + ["add", "spec.json"], check=True)
    subprocess.run(git + ["commit", "-qm", "freeze gate"], check=True)
    assert score.spec_committed(str(spec), str(tmp_path))
    spec.write_text(spec.read_text() + " ")
    assert not score.spec_committed(str(spec), str(tmp_path))
