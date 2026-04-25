"""Session C VM runner helpers. No GPU, no Colab."""

import json
import os
import sys

GPU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu")
sys.path.insert(0, GPU)
import vm_run_phase2 as vm  # noqa: E402


def test_skip_when_summary_already_landed(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "RESULT_ROOT", str(tmp_path))
    cond = {"id": "identity", "result_name": "identity"}
    out = tmp_path / "identity"
    out.mkdir()
    assert not vm.condition_already_landed(cond)
    (out / "summary.json").write_text(json.dumps({"ok": True}))
    assert vm.condition_already_landed(cond)
