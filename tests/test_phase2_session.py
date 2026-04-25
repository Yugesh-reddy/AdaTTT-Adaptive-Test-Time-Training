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
