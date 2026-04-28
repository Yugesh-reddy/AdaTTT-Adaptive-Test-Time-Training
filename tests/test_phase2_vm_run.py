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
