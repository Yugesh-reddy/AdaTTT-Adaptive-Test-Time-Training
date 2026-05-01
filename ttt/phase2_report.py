"""
Phase 2 reporting: sample_flops ladder, compact outcomes, Pareto, oracle, δ/p.

Always report AdaptiveRouter.sample_flops. The 58/93/128 GFLOPs ladder from an
earlier spec does not match those constants under CLIP and is stored only so
the gap is visible.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ttt.gate import AdaptiveRouter
from ttt.metrics import pareto_frontier
from ttt.shift_cache import LEGACY_LADDER_GFLOPS


def sample_flops_ladder(backend: str = "clip") -> Dict[str, Any]:
    AdaptiveRouter.configure_for_backend(backend)

    def gflops(adapted: bool, n_aug: int, k_steps: int = 1) -> float:
        return AdaptiveRouter.sample_flops(
            adapted=adapted, n_aug=n_aug, k_steps=k_steps
        ) / 1e9

    return {
        "backend": backend,
        "sample_flops_gflops": {
            "base": gflops(False, 0),
            "tent_k1": gflops(True, 0),
            "memo2_k1": gflops(True, 2),
            "memo4_k1": gflops(True, 4),
        },
        "legacy_ladder_gflops": dict(LEGACY_LADDER_GFLOPS),
        "used_for_reporting": "sample_flops",
        "legacy_ladder_note": (
            "The 58/93/128 GFLOPs ladder in the original spec does not match "
            "AdaptiveRouter.sample_flops. MEMO is charged a fusion forward and "
            "backward through every view on every step, plus the post-update "
            "prediction; results/phase2 npz files written before 2026-09-18 carry "
            "an older accounting that billed one backward per step (MEMO4 138.6G "
            "instead of ~175.8G) — scripts/writeup_numbers.py recomputes them. "
            "layernorm_only is Δ=0 FLOPs by design (optimizer state, not compute)."
        ),
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci_pp(delta: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list:
    """95% CI of mean(delta) in pp, resampling samples (deltas are paired per sample)."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.concatenate([
        delta[rng.integers(0, n, size=(200, n))].mean(axis=1) for _ in range(n_boot // 200)
    ])
    return [round(100.0 * float(np.percentile(means, q)), 3) for q in (2.5, 97.5)]


def oracle_recovery(base_soft: np.ndarray, adapted_soft: np.ndarray) -> Dict[str, float]:
    """Per-sample pick of the better soft score (upper bound on gating)."""
    base_soft = np.asarray(base_soft, dtype=float)
    adapted_soft = np.asarray(adapted_soft, dtype=float)
    oracle = np.maximum(base_soft, adapted_soft)
    return {
        "oracle_soft": float(oracle.mean()) if len(oracle) else 0.0,
        "base_soft": float(base_soft.mean()) if len(base_soft) else 0.0,
        "adapted_soft": float(adapted_soft.mean()) if len(adapted_soft) else 0.0,
    }


def delta_over_p(
    base_acc: float, gated_acc: float, adapt_rate: float
) -> Dict[str, Optional[float]]:
    """To beat base by δ at adapt rate p you need δ/p net on the touched samples."""
    delta = float(gated_acc) - float(base_acc)
    p = float(adapt_rate)
    return {
        "delta": delta,
        "p": p,
        "delta_over_p": (delta / p) if p > 0 else None,
    }


def compact_outcomes(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    soft_score: np.ndarray,
    adapted: np.ndarray,
    gate_score: np.ndarray,
    flops_g: np.ndarray,
) -> Dict[str, np.ndarray]:
    """~13 B/sample packed arrays for results/phase2/*.npz (not fat JSON)."""
    return {
        "prediction": np.asarray(prediction, dtype=np.uint16),
        "ground_truth": np.asarray(ground_truth, dtype=np.uint16),
        "soft_score": np.asarray(soft_score, dtype=np.float16),
        "adapted": np.asarray(adapted, dtype=np.uint8),
        "gate_score": np.asarray(gate_score, dtype=np.float16),
        "flops_g": np.asarray(flops_g, dtype=np.float16),
    }


def save_outcomes_npz(path: str, **arrays: np.ndarray) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_phase2_report(
    runs: List[Dict[str, Any]],
    out_dir: str,
    backend: str = "clip",
    oracle: Optional[Dict[str, Any]] = None,
    tau: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write summary / Pareto / FLOPs ladder JSON under out_dir (results/phase2/).

    `tau` records where the gate threshold came from (protocol held_out, fixed,
    fit or in_sample), so a gated number is never read without its provenance.
    """
    os.makedirs(out_dir, exist_ok=True)
    ladder = sample_flops_ladder(backend)
    with open(os.path.join(out_dir, "flops_ladder.json"), "w") as fh:
        json.dump(ladder, fh, indent=2)

    summarized: List[Dict[str, Any]] = []
    for run in runs:
        flops = np.asarray(run.get("flops", [run["avg_flops"]]), dtype=float)
        rec = {k: v for k, v in run.items() if k != "flops"}
        rec["flops_p50"] = float(np.percentile(flops, 50))
        rec["flops_p95"] = float(np.percentile(flops, 95))
        rec["avg_flops"] = float(run["avg_flops"])
        rec["accuracy"] = float(run["accuracy"])
        if "adapt_rate" in run and "soft" in run and "no_adapt" in {r.get("config") for r in runs}:
            base = next((r for r in runs if r.get("config") == "no_adapt"), None)
            if base is not None:
                rec["delta_over_p"] = delta_over_p(
                    base.get("soft", base["accuracy"]),
                    run.get("soft", run["accuracy"]),
                    run["adapt_rate"],
                )
        summarized.append(rec)

    pareto = pareto_frontier(summarized)
    with open(os.path.join(out_dir, "pareto.json"), "w") as fh:
        json.dump(_jsonable(pareto), fh, indent=2)
    summary = {
        "n_runs": len(runs),
        "runs": _jsonable(summarized),
        "flops_ladder": ladder,
        "backend": backend,
    }
    if tau is not None:
        summary["tau"] = _jsonable(tau)
    if oracle is not None:
        oracle_blob = _jsonable(oracle)
        summary["oracle"] = oracle_blob
        with open(os.path.join(out_dir, "oracle.json"), "w") as fh:
            json.dump(oracle_blob, fh, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
