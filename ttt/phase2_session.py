"""Phase 2 Session C: ID control on eval 8k, then one hard visual shift.

Torch-free so the Mac orchestrator can load this file without importing ttt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

ALL_METHODS: Sequence[str] = (
    "no_adapt",
    "tent",
    "eata",
    "memo",
    "memo_sar",
    "gated_memo_sar",
)

ID_METHODS: Sequence[str] = ("no_adapt", "memo")

# τ is fit here, under the same corruption, never on the report-only eval 8k.
GATE_SUBSET = "data/gate_train_subset_8k.json"


def needs_tau_fit(condition: Dict[str, Any]) -> bool:
    """Gated conditions fit τ on GATE_SUBSET first (held-out protocol)."""
    return "gated_memo_sar" in condition["methods"]


def tau_fit_source(condition: Dict[str, Any]) -> str:
    """Source tag for the gate-train fit; differs from the eval source by design."""
    return f"gate_train_{condition['source']}"


def session_c_conditions() -> List[Dict[str, Any]]:
    """Identity skip+MEMO, then gaussian noise s5 with the full method set."""
    return [
        {
            "id": "identity",
            "corruption": "identity",
            "severity": 1,
            "methods": list(ID_METHODS),
            "cache_name": "identity.pt",
            "result_name": "identity",
            "source": "identity_eval8k",
        },
        {
            "id": "gaussian_noise_s5",
            "corruption": "gaussian_noise",
            "severity": 5,
            "methods": list(ALL_METHODS),
            "cache_name": "noise_s5.pt",
            "result_name": "noise_s5",
            "source": "corruption_gaussian_noise_s5",
        },
    ]


# Session D, fixed before any run. For each lr, skip and dense MEMO (K=1, fusion
# LayerNorms, reset per sample) on GATE_SUBSET under gaussian noise s5. The lr
# with the highest gate-train MEMO gain wins, ties to the smaller lr. If that
# gain is under STEP_SWEEP_MIN_GAIN_PP the session stops and the eval 8k is
# never touched; otherwise τ is fit on GATE_SUBSET with memo_sar at that lr and
# the eval 8k is scored once. 1e-4, the Session C step, moved ~0.5% of answers.
ACTIVE_SESSION = "d"
STEP_SWEEP_LRS: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2)
STEP_SWEEP_MIN_GAIN_PP = 0.1


def session_d_conditions() -> List[Dict[str, Any]]:
    """Does a bigger update step recover the noise-s5 drop? Chosen on gate-train only."""
    return [
        {
            "id": "gaussian_noise_s5_step_sweep",
            "kind": "step_sweep",
            "corruption": "gaussian_noise",
            "severity": 5,
            "lrs": list(STEP_SWEEP_LRS),
            "min_gain_pp": STEP_SWEEP_MIN_GAIN_PP,
            "methods": ["no_adapt", "memo", "memo_sar", "gated_memo_sar"],
            "cache_name": "noise_s5.pt",
            "result_name": "noise_s5_step_sweep",
            "source": "corruption_gaussian_noise_s5",
        },
    ]


def active_conditions() -> List[Dict[str, Any]]:
    """The conditions the next VM session runs."""
    return {"c": session_c_conditions, "d": session_d_conditions}[ACTIVE_SESSION]()


def lr_tag(lr: float) -> str:
    return f"lr_{lr:g}"


def select_step(gain_pp_by_lr: Dict[float, float], min_gain_pp: float):
    """The pre-registered choice. Returns (lr or None, record); None means stop.

    Picks the lr with the largest gate-train gain, ties to the smaller lr, and
    proceeds only if that gain reaches min_gain_pp.
    """
    best = max(gain_pp_by_lr.values())
    chosen = min(lr for lr, g in gain_pp_by_lr.items() if g >= best - 1e-9)
    proceed = best >= min_gain_pp
    record = {
        "rule": ("argmax gate-train dense-MEMO gain, ties to the smaller lr; "
                 f"proceed only if the gain is >= {min_gain_pp} pp"),
        "gain_pp_by_lr": {lr_tag(lr): g for lr, g in sorted(gain_pp_by_lr.items())},
        "best_lr": chosen,
        "best_gain_pp": best,
        "min_gain_pp": min_gain_pp,
        "proceed": proceed,
    }
    return (chosen if proceed else None), record


def _eval_artifacts(methods: Sequence[str]) -> List[str]:
    names = ["summary.json", "pareto.json", "flops_ladder.json"]
    names.extend(f"{m}.npz" for m in methods)
    if "no_adapt" in methods and "memo" in methods:
        names.append("oracle.json")
    if "gated_memo_sar" in methods:
        names.extend(["tau.json", "tau_fit/tau.json", "tau_fit/summary.json"])
    return names


def eval_artifacts_for(condition: Dict[str, Any]) -> List[str]:
    """Files the eval-8k stage lands (absent for a step sweep that stopped)."""
    return _eval_artifacts(list(condition["methods"]))


def essential_artifacts(condition: Dict[str, Any]) -> List[str]:
    """Files that must land before the VM is stopped."""
    return ["decision.json"] if condition.get("kind") == "step_sweep" else ["summary.json"]


def artifacts_for(condition: Dict[str, Any]) -> List[str]:
    """Compact files to land for one condition. No feature cache."""
    names = eval_artifacts_for(condition)
    if condition.get("kind") == "step_sweep":
        sweep = [f"sweep/{lr_tag(lr)}/{f}" for lr in condition["lrs"]
                 for f in ("summary.json", "memo.npz")]
        names = ["decision.json", *sweep, *names]
    return names
