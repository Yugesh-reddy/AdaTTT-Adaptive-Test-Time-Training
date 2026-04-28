#!/usr/bin/env python3
"""
Derive results/writeup/numbers.json from the landed Phase 1 and Phase 2 artifacts.

Every number in the writeup comes from here, never typed by hand. Inputs are the
gitignored results under results/phase1/ and results/phase2/ (see ARTIFACTS.md).

Per-sample FLOPs are recomputed with AdaptiveRouter.sample_flops from each
method's `adapted` flags. The npz files carry the accounting in force when they
were written, which charged one backward per step and billed MEMO4 138.6G of
the ~175.8G it runs; the landed averages are kept alongside for comparison.

    python scripts/writeup_numbers.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttt.gate import AdaptiveRouter  # noqa: E402
from ttt.models import FullVQAModel  # noqa: E402
from ttt.utils import load_config  # noqa: E402

PHASE1 = os.path.join(ROOT, "results", "phase1")
PHASE2 = os.path.join(ROOT, "results", "phase2")
OUT = os.path.join(ROOT, "results", "writeup", "numbers.json")
CONDITIONS = ("identity", "blur_s3", "noise_s5")
METHODS = ("no_adapt", "tent", "eata", "memo", "memo_sar", "gated_memo_sar")
N_AUG = {"tent": 0, "eata": 0, "memo": 4, "memo_sar": 4, "gated_memo_sar": 4}
LABELS = {
    "no_adapt": "skip",
    "tent": "TENT-style (episodic, 1 step)",
    "eata": "EATA-style filter (episodic, no Fisher term)",
    "memo": "MEMO, 4 views",
    "memo_sar": "MEMO + SAR entropy filter",
    "gated_memo_sar": "gated MEMO + SAR filter",
}


def _ceiling_module():
    path = os.path.join(ROOT, "scripts", "06_ceiling_check.py")
    spec = importlib.util.spec_from_file_location("ceiling_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _pct(x: float) -> float:
    return round(100.0 * float(x), 3)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def corrected_flops(method: str, adapted: np.ndarray) -> np.ndarray:
    skip = AdaptiveRouter.sample_flops(adapted=False) / 1e9
    if method == "no_adapt":
        return np.full(len(adapted), skip)
    full = AdaptiveRouter.sample_flops(adapted=True, n_aug=N_AUG[method], k_steps=1) / 1e9
    return np.where(adapted.astype(bool), full, skip)


def phase1_numbers(ceiling) -> dict:
    c = _load(os.path.join(PHASE1, "ceiling_check.json"))
    soft, exact = c["runs"], c["exact"]["runs"]
    clip, ctrl = soft["clip"], soft["control (vit_bert)"]
    return {
        "subset": "vqa_v2 val (full)",
        "hours": _load(os.path.join(PHASE1, "run_summary.json"))["vm_hours"],
        "soft": {
            "v1": ceiling.V1_REFERENCE_SOFT,
            "control_vitbert": ctrl["final"],
            "clip": clip["final"],
            "encoder_delta_pp": round(clip["final"] - ctrl["final"], 2),
            "clip_ep5": clip["acc_ep5"],
            "clip_slope_5_to_8_pp": round(clip["slope_5_to_8"], 2),
            "control_ep5": ctrl["acc_ep5"],
            "control_slope_5_to_8_pp": round(ctrl["slope_5_to_8"], 2),
        },
        "exact": {
            "v1": ceiling.V1_REFERENCE_EXACT,
            "control_vitbert": exact["control (vit_bert)"]["final"],
            "clip": exact["clip"]["final"],
            "encoder_delta_pp": round(exact["clip"]["final"] - exact["control (vit_bert)"]["final"], 2),
        },
        "verdict": {"clip": clip["verdict"], "control_vitbert": ctrl["verdict"]},
    }


def condition_numbers(cond: str, id_skip_soft: float | None) -> dict:
    d = os.path.join(PHASE2, cond)
    runs = {m: np.load(os.path.join(d, f"{m}.npz")) for m in METHODS
            if os.path.exists(os.path.join(d, f"{m}.npz"))}
    summary = _load(os.path.join(d, "summary.json"))
    base = runs["no_adapt"]
    b_soft = base["soft_score"].astype(float)
    b_ok = base["prediction"] == base["ground_truth"]
    n = len(b_soft)

    methods = {}
    for m, r in runs.items():
        soft = r["soft_score"].astype(float)
        ok = r["prediction"] == r["ground_truth"]
        adapted = r["adapted"].astype(bool)
        flops = corrected_flops(m, adapted)
        p = float(adapted.mean())
        delta = soft.mean() - b_soft.mean()
        b1a0, b0a1 = int((b_ok & ~ok).sum()), int((~b_ok & ok).sum())
        methods[m] = {
            "label": LABELS[m],
            "soft": _pct(soft.mean()),
            "exact": _pct(ok.mean()),
            "delta_soft_pp": _pct(delta),
            "adapt_rate": round(p, 4),
            "delta_over_p_pp": _pct(delta / p) if p > 0 else None,
            "avg_gflops": round(float(flops.mean()), 2),
            "p50_gflops": round(float(np.percentile(flops, 50)), 2),
            "p95_gflops": round(float(np.percentile(flops, 95)), 2),
            "landed_avg_gflops_old_accounting": round(float(r["flops_g"].astype(float).mean()), 2),
            "pred_flips_vs_skip": int((r["prediction"] != base["prediction"]).sum()),
            "soft_better": int((soft > b_soft + 1e-6).sum()),
            "soft_worse": int((soft < b_soft - 1e-6).sum()),
            "mcnemar_skip_right_method_wrong": b1a0,
            "mcnemar_skip_wrong_method_right": b0a1,
            "mcnemar_exact_p": round(mcnemar_exact_p(b1a0, b0a1), 4),
        }

    out = {
        "n": n,
        "skip_soft": _pct(b_soft.mean()),
        "skip_exact": _pct(b_ok.mean()),
        "drop_vs_id_pp": None if id_skip_soft is None else round(_pct(b_soft.mean()) - id_skip_soft, 2),
        "methods": methods,
    }
    if "memo" in runs:
        oracle = np.maximum(b_soft, runs["memo"]["soft_score"].astype(float))
        out["oracle_soft"] = _pct(oracle.mean())
        out["oracle_delta_pp"] = _pct(oracle.mean() - b_soft.mean())
    base_run = next(r for r in summary["runs"] if r["config"] == "no_adapt")
    out["maxprob_auroc"] = round(base_run["maxprob_auroc"], 3)
    out["maxprob_aurc"] = round(base_run["maxprob_aurc"], 3)

    tau_path = os.path.join(d, "tau.json")
    if os.path.exists(tau_path):
        t = _load(tau_path)
        tau = float(t["tau"])
        out["gate"] = {
            "tau": round(tau, 4),
            # Runs landed before 2026-09-18 carry no protocol field: they tuned
            # τ on this condition's own eval-8k no_adapt vs memo outcomes.
            "protocol": t.get("protocol", "in_sample"),
            "tuned_on": t.get("tuned_on", t.get("source")),
            "tuned_against": t.get("target_method", "memo"),
            "gate_pass_rate": round(float((base["gate_score"].astype(float) >= tau).mean()), 4),
            "adapt_rate": methods["gated_memo_sar"]["adapt_rate"],
        }
    return out


def main() -> int:
    AdaptiveRouter.configure_for_backend("clip")
    config = load_config(os.path.join(ROOT, "config", "config.yaml"))
    model = FullVQAModel(config)
    adapted_values = sum(p.numel() for _, p in model.get_ttt_params_named(
        adapt_modules=["fusion"], include_auxiliary=False, layernorm_only=True))

    def g(adapted, n_aug=0):
        return round(AdaptiveRouter.sample_flops(adapted=adapted, n_aug=n_aug, k_steps=1) / 1e9, 1)

    eval_8k = {}
    id_skip = None
    for cond in CONDITIONS:
        eval_8k[cond] = condition_numbers(cond, id_skip)
        if cond == "identity":
            id_skip = eval_8k[cond]["skip_soft"]

    ceiling = _ceiling_module()
    numbers = {
        "generated_by": "scripts/writeup_numbers.py",
        "metric": "official_soft_min_votes_over_3_no_unk_credit",
        "artifacts": {
            "phase1_ceiling": "results/phase1/ceiling_check.json",
            **{c: f"results/phase2/{c}/" for c in CONDITIONS},
        },
        "flops_ladder_gflops": {
            "base": g(False), "tent_k1": g(True, 0),
            "memo2_k1": g(True, 2), "memo4_k1": g(True, 4),
            "source": "AdaptiveRouter.sample_flops, fusion backward charged per MEMO view",
            "landed_npz_memo4_k1": 138.6,
        },
        "adapter": {
            "params": "fusion LayerNorm affines",
            "adapted_values": int(adapted_values),
            "optimizer": "Adam, fresh per sample, restored after each sample",
            "lr": float(config.get("ttt_lr", 1e-4)),
            "k_steps": 1,
            "note": "Adam's first step moves every selected value by ~lr whatever the gradient",
        },
        "phase1": phase1_numbers(ceiling),
        "eval_8k": eval_8k,
        "priced_hours": {
            "blur_s3": _load(os.path.join(PHASE2, "blur_s3", "orch_summary.json"))["vm_hours"],
            "session_c_identity_and_noise_s5": _load(os.path.join(PHASE2, "orch_summary.json"))["vm_hours"],
        },
    }
    with open(OUT, "w") as fh:
        json.dump(numbers, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
