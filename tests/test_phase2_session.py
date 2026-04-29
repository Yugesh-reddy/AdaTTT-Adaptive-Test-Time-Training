"""Session C: ID control, then one hard visual shift. No GPU."""

from ttt.phase2_session import artifacts_for, session_c_conditions


def test_session_c_is_identity_then_noise_s5():
    conds = session_c_conditions()
    assert [c["id"] for c in conds] == ["identity", "gaussian_noise_s5"]
    ident, noise = conds
    assert ident["corruption"] == "identity"
    assert ident["methods"] == ["no_adapt", "memo"]
    assert ident["result_name"] == "identity"
    assert noise["corruption"] == "gaussian_noise"
    assert noise["severity"] == 5
    assert noise["methods"][0] == "no_adapt"
    assert "gated_memo_sar" in noise["methods"]
    assert noise["result_name"] == "noise_s5"


def test_identity_artifacts_omit_unused_methods():
    ident = session_c_conditions()[0]
    names = artifacts_for(ident)
    assert "summary.json" in names
    assert "oracle.json" in names
    assert "no_adapt.npz" in names
    assert "memo.npz" in names
    assert "tent.npz" not in names
    assert "tau.json" not in names


def test_noise_s5_artifacts_include_gate_and_oracle():
    noise = session_c_conditions()[1]
    names = artifacts_for(noise)
    assert "tau.json" in names
    assert "oracle.json" in names
    assert "gated_memo_sar.npz" in names
    assert "tent.npz" in names


# --- Session D: bigger update step, chosen on gate-train only -----------------

from ttt import phase2_session as ps  # noqa: E402


def test_active_session_is_the_step_sweep():
    (cond,) = ps.active_conditions()
    assert cond["kind"] == "step_sweep" and cond["corruption"] == "gaussian_noise"
    assert cond["severity"] == 5 and 1e-4 not in cond["lrs"]
    assert ps.essential_artifacts(cond) == ["decision.json"]


def test_step_sweep_artifacts_cover_sweep_decision_and_eval():
    (cond,) = ps.session_d_conditions()
    names = ps.artifacts_for(cond)
    assert names[0] == "decision.json"
    assert "sweep/lr_0.01/summary.json" in names and "sweep/lr_0.03/memo.npz" in names
    assert "tau_fit/tau.json" in names and "gated_memo_sar.npz" in names


def test_select_step_ties_go_to_the_smaller_lr():
    chosen, record = ps.select_step({1e-3: 0.05, 1e-2: 0.30, 3e-2: 0.30}, 0.1)
    assert chosen == 1e-2 and record["proceed"] is True
    assert record["gain_pp_by_lr"]["lr_0.01"] == 0.30


def test_select_step_stops_below_the_threshold():
    chosen, record = ps.select_step({1e-3: 0.02, 1e-2: 0.09}, 0.1)
    assert chosen is None and record["proceed"] is False and record["best_lr"] == 1e-2
