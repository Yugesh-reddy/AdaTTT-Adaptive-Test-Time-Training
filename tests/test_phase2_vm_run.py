"""Session C VM runner helpers. No GPU, no Colab."""

import json
import os
import sys

GPU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu")
sys.path.insert(0, GPU)
import vm_run_phase2 as vm  # noqa: E402
from ttt.phase2_session import artifacts_for, session_c_conditions  # noqa: E402


def test_skip_when_summary_already_landed(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "RESULT_ROOT", str(tmp_path))
    cond = {"id": "identity", "result_name": "identity"}
    out = tmp_path / "identity"
    out.mkdir()
    assert not vm.condition_already_landed(cond)
    (out / "summary.json").write_text(json.dumps({"ok": True}))
    assert vm.condition_already_landed(cond)


def _fakes(tmp_path, monkeypatch, cond):
    monkeypatch.setattr(vm, "RESULT_ROOT", str(tmp_path / "results"))
    monkeypatch.setattr(vm, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(vm, "PROGRESS", str(tmp_path / "progress.json"))
    calls = []

    def precompute(argv):
        calls.append(("precompute", argv))
        out = argv[argv.index("--output") + 1]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(b"cache")
        return 0

    def evaluate(argv):
        calls.append(("eval", argv))
        out = argv[argv.index("--output") + 1]
        os.makedirs(out, exist_ok=True)
        if "--fit-tau" in argv:
            names = ["tau.json", "summary.json"]
        else:
            names = [n for n in artifacts_for(cond) if "/" not in n]
        for name in names:
            open(os.path.join(out, name), "w").write(json.dumps({"tau": 0.4}))
        return 0

    return calls, precompute, evaluate


def test_gated_condition_fits_tau_on_gate_train_before_eval(tmp_path, monkeypatch):
    cond = session_c_conditions()[1]
    calls, precompute, evaluate = _fakes(tmp_path, monkeypatch, cond)
    vm.run_condition(cond, precompute, evaluate)
    kinds = [(k, a[a.index("--subset") + 1] if k == "precompute" else None) for k, a in calls]
    assert kinds == [("precompute", vm.GATE_SUBSET), ("eval", None),
                     ("precompute", vm.SUBSET), ("eval", None)]
    fit, report = calls[1][1], calls[3][1]
    assert "--fit-tau" in fit
    assert fit[fit.index("--source") + 1] == f"gate_train_{cond['source']}"
    assert fit[fit.index("--methods") + 1: fit.index("--fit-tau")] == ["no_adapt", "memo_sar"]
    assert report[report.index("--tau-file") + 1].endswith(os.path.join("tau_fit", "tau.json"))
    assert "--fit-tau" not in report
    assert not os.listdir(tmp_path / "cache") if os.path.isdir(tmp_path / "cache") else True


def test_ungated_condition_needs_no_tau(tmp_path, monkeypatch):
    cond = session_c_conditions()[0]
    calls, precompute, evaluate = _fakes(tmp_path, monkeypatch, cond)
    vm.run_condition(cond, precompute, evaluate)
    assert [k for k, _ in calls] == ["precompute", "eval"]
    assert "--tau-file" not in calls[1][1] and "--fit-tau" not in calls[1][1]


def test_existing_tau_fit_is_reused_after_preemption(tmp_path, monkeypatch):
    cond = session_c_conditions()[1]
    calls, precompute, evaluate = _fakes(tmp_path, monkeypatch, cond)
    fit_dir = os.path.join(str(tmp_path / "results"), cond["result_name"], "tau_fit")
    os.makedirs(fit_dir)
    for name in ("tau.json", "summary.json"):
        open(os.path.join(fit_dir, name), "w").write(json.dumps({"tau": 0.4}))
    vm.run_condition(cond, precompute, evaluate)
    assert [k for k, _ in calls] == ["precompute", "eval"]
    assert "--tau-file" in calls[1][1]


# --- Session D step sweep -------------------------------------------------------

import numpy as np  # noqa: E402

from ttt.phase2_session import eval_artifacts_for, session_d_conditions  # noqa: E402


def _sweep_fakes(tmp_path, monkeypatch, gains_pp):
    cond = session_d_conditions()[0]
    monkeypatch.setattr(vm, "RESULT_ROOT", str(tmp_path / "results"))
    monkeypatch.setattr(vm, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(vm, "PROGRESS", str(tmp_path / "progress.json"))
    calls = []

    def precompute(argv):
        calls.append(("precompute", argv[argv.index("--subset") + 1]))
        out = argv[argv.index("--output") + 1]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(b"cache")
        return 0

    def evaluate(argv):
        out = argv[argv.index("--output") + 1]
        lr = float(argv[argv.index("--lr") + 1]) if "--lr" in argv else None
        os.makedirs(out, exist_ok=True)
        if "--fit-tau" in argv:
            calls.append(("fit_tau", lr))
            for name in ("tau.json", "summary.json"):
                open(os.path.join(out, name), "w").write(
                    json.dumps({"tau": 0.4, "source": "gate_train_x", "lr": lr}))
        elif os.sep + "sweep" + os.sep in out:
            calls.append(("sweep", lr))
            base = 0.60
            memo = base + gains_pp[lr] / 100.0
            open(os.path.join(out, "summary.json"), "w").write(json.dumps({
                "runs": [{"config": "no_adapt", "soft": base},
                         {"config": "memo", "soft": memo, "pred_flips_vs_no_adapt": 100}],
                "oracle": {"oracle_soft": memo + 0.001, "base_soft": base},
            }))
            np.savez(os.path.join(out, "memo.npz"), x=np.zeros(1))
        else:
            calls.append(("eval", lr, argv[argv.index("--tau-file") + 1]))
            for name in eval_artifacts_for(cond):
                if "/" not in name:
                    open(os.path.join(out, name), "w").write("{}")
        return 0

    return cond, calls, precompute, evaluate


def test_step_sweep_picks_on_gate_train_then_scores_eval_once(tmp_path, monkeypatch):
    gains = {1e-3: 0.05, 3e-3: 0.2, 1e-2: 0.3, 3e-2: 0.3}
    cond, calls, precompute, evaluate = _sweep_fakes(tmp_path, monkeypatch, gains)
    vm.run_step_sweep(cond, precompute, evaluate)
    assert calls[0] == ("precompute", vm.GATE_SUBSET)
    assert [c for c in calls if c[0] == "sweep"] == [("sweep", lr) for lr in cond["lrs"]]
    assert ("fit_tau", 1e-2) in calls  # tie between 1e-2 and 3e-2 goes to the smaller
    assert calls[-2] == ("precompute", vm.SUBSET)
    assert calls[-1][:2] == ("eval", 1e-2)
    assert calls[-1][2].endswith(os.path.join("tau_fit", "tau.json"))
    assert sum(1 for c in calls if c == ("precompute", vm.GATE_SUBSET)) == 1
    decision = json.load(open(os.path.join(vm.condition_output(cond), "decision.json")))
    assert decision["proceed"] and decision["chosen_lr"] == 1e-2
    assert decision["eval_subset_touched"] is True


def test_step_sweep_stops_before_the_eval_subset(tmp_path, monkeypatch):
    gains = {1e-3: 0.01, 3e-3: 0.05, 1e-2: 0.08, 3e-2: -0.4}
    cond, calls, precompute, evaluate = _sweep_fakes(tmp_path, monkeypatch, gains)
    vm.run_step_sweep(cond, precompute, evaluate)
    assert ("precompute", vm.SUBSET) not in calls
    assert not any(c[0] in ("fit_tau", "eval") for c in calls)
    decision = json.load(open(os.path.join(vm.condition_output(cond), "decision.json")))
    assert decision["proceed"] is False and decision["eval_subset_touched"] is False
    assert not os.path.exists(vm.tau_fit_cache(cond))


def test_step_sweep_resumes_from_its_decision(tmp_path, monkeypatch):
    gains = {lr: 0.3 for lr in (1e-3, 3e-3, 1e-2, 3e-2)}
    cond, calls, precompute, evaluate = _sweep_fakes(tmp_path, monkeypatch, gains)
    out = vm.condition_output(cond)
    os.makedirs(os.path.join(out, "tau_fit"))
    open(os.path.join(out, "decision.json"), "w").write(
        json.dumps({"proceed": True, "chosen_lr": 1e-3}))
    for name in ("tau.json", "summary.json"):  # what a finished --fit-tau run leaves
        open(os.path.join(out, "tau_fit", name), "w").write(json.dumps({"tau": 0.4}))
    vm.run_step_sweep(cond, precompute, evaluate)
    assert calls == [("precompute", vm.SUBSET), calls[-1]]
    assert calls[-1][:2] == ("eval", 1e-3)
