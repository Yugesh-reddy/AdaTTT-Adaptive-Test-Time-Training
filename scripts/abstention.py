#!/usr/bin/env python3
"""
Selective prediction: answer only when confident (the Phase 2 fallback).

Visual TTA did not recover the noise-s5 drop (results/writeup/REPORT.md §2.8–2.9),
but the model's confidence ranks its own correctness well. This measures what
abstaining on the least confident questions buys. Fixed before looking at eval:

- Signal: the stored gate score, 0.5·normalized entropy + 0.5·(1 − MaxProb) of
  the unadapted model; lower = more confident. No extra compute.
- Thresholds: chosen on the noise-s5 gate-train split for 90% and 80% coverage.
- Applied unchanged to the eval 8k under identity, blur s3 and noise s5, so
  coverage drift across shifts is part of the result.
- Reported: realized coverage, soft and exact accuracy on the answered
  questions, the soft gain over answering everything (paired bootstrap 95% CI),
  and the soft-risk AURC.

    python scripts/abstention.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2 = os.path.join(ROOT, "results", "phase2")
CALIBRATION = os.path.join(PHASE2, "noise_s5_signals", "gate_train", "no_adapt.npz")
EVAL = {
    "identity": os.path.join(PHASE2, "identity", "no_adapt.npz"),
    "blur_s3": os.path.join(PHASE2, "blur_s3", "no_adapt.npz"),
    "noise_s5": os.path.join(PHASE2, "noise_s5_signals", "eval_sealed", "no_adapt.npz"),
}
TARGETS = (0.9, 0.8)
OUT = os.path.join(ROOT, "results", "writeup", "abstention.json")


def threshold_for_coverage(scores: np.ndarray, target: float) -> float:
    """Answer when score <= t; t gives at least `target` coverage on these scores."""
    return float(np.quantile(scores, target, method="higher"))


def selective(scores, soft, correct, t, n_boot=2000, seed=0) -> dict:
    """Accuracy on the answered questions, and its gain over answering all."""
    accept = scores <= t
    rng = np.random.default_rng(seed)
    n = len(soft)
    gains = []
    for _ in range(n_boot // 200):
        idx = rng.integers(0, n, size=(200, n))
        a, s = accept[idx], soft[idx]
        gains.append((s * a).sum(axis=1) / np.maximum(a.sum(axis=1), 1) - s.mean(axis=1))
    gains = np.concatenate(gains)
    return {
        "coverage": round(float(accept.mean()), 4),
        "soft_answered": round(100 * float(soft[accept].mean()), 3),
        "exact_answered": round(100 * float(correct[accept].mean()), 3),
        "soft_gain_pp": round(100 * float(soft[accept].mean() - soft.mean()), 3),
        "soft_gain_ci95_pp": [round(100 * float(np.percentile(gains, q)), 3) for q in (2.5, 97.5)],
    }


def risk_coverage(scores, soft, points: int = 20) -> dict:
    """Selective soft accuracy vs coverage, most confident first, plus soft-risk AURC."""
    order = np.argsort(scores, kind="mergesort")
    acc = np.cumsum(soft[order]) / np.arange(1, len(soft) + 1)
    cov = np.arange(1, len(soft) + 1) / len(soft)
    grid = np.linspace(1.0 / points, 1.0, points)
    at = np.searchsorted(cov, grid - 1e-12)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return {
        "coverage": [round(float(c), 3) for c in grid],
        "soft_answered": [round(100 * float(acc[i]), 3) for i in at],
        "aurc_soft_risk": round(float(trapz(1.0 - acc, cov)), 4),
    }


def _load(path):
    z = np.load(path)
    return (z["gate_score"].astype(float), z["soft_score"].astype(float),
            z["prediction"] == z["ground_truth"])


def main() -> int:
    calib_scores, _, _ = _load(CALIBRATION)
    thresholds = {f"{int(100 * t)}": threshold_for_coverage(calib_scores, t) for t in TARGETS}
    result = {
        "signal": "gate_score = 0.5*entropy_norm + 0.5*(1-maxprob), unadapted model; lower = more confident",
        "calibrated_on": os.path.relpath(CALIBRATION, ROOT),
        "thresholds": thresholds,
        "conditions": {},
    }
    for cond, path in EVAL.items():
        scores, soft, correct = _load(path)
        result["conditions"][cond] = {
            "all_soft": round(100 * float(soft.mean()), 3),
            "all_exact": round(100 * float(correct.mean()), 3),
            "at_target": {k: selective(scores, soft, correct, t) for k, t in thresholds.items()},
            "risk_coverage": risk_coverage(scores, soft),
        }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"thresholds from noise-s5 gate-train: {thresholds}")
    for cond, r in result["conditions"].items():
        print(f"{cond:9s} all {r['all_soft']:.2f}  AURC {r['risk_coverage']['aurc_soft_risk']:.4f}")
        for k, s in r["at_target"].items():
            print(f"   target {k}%: coverage {s['coverage']:.3f}  answered soft {s['soft_answered']:.2f} "
                  f"(exact {s['exact_answered']:.2f})  gain {s['soft_gain_pp']:+.2f} pp {s['soft_gain_ci95_pp']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
