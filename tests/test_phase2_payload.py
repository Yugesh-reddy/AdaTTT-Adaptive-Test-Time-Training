"""The Phase 2 VM payload must ship every file the VM runner reads.

Session D and the held-out τ fit read gate_train_subset_8k.json on the VM; a
payload without it fails ~20 minutes into a paid run, at precompute.
"""

import importlib.util
import os

from ttt.phase2_session import GATE_SUBSET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _payload_module():
    path = os.path.join(ROOT, "tools", "colab_orchestrator", "build_payload_phase2.py")
    spec = importlib.util.spec_from_file_location("build_payload_phase2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_payload_ships_both_frozen_subsets():
    names = set(_payload_module().EXPLICIT)
    assert "data/eval_subset_8k.json" in names
    assert GATE_SUBSET in names


def test_payload_ships_the_session_code():
    files = _payload_module().files()
    for rel in ("ttt/phase2_session.py", "gpu/vm_run_phase2.py", "gpu/eval_phase2.py"):
        assert rel in files
