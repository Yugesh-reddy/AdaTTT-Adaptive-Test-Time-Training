"""
A gate that predicts which samples MEMO helps (Phase 2, Session E).

MEMO resets its weights after every sample, so a gate's outcome on a dataset is
exactly where(adapt, MEMO outcome, skip outcome): gates can be fit and compared
offline from per-sample outcomes and signals. Everything here is fixed before
the Session E data exists:

- Candidate gates (GATES). "score" thresholds the existing weighted gate score.
  The others are logistic regressions on growing signal sets, each tier costing
  a deployed gate more to observe (TIER_PROBE_VIEWS; "post" means MEMO already
  ran and the gate only decides whether to keep its answer).
- Training uses the samples MEMO changed, labelled helped vs hurt and weighted
  by |soft change|: an unchanged sample's decision costs compute, not accuracy.
- The threshold maximizes gated gain on the training data; ties adapt fewer.
- 5-fold cross-validation on gate_train estimates each gate's gain. The best
  gain wins, a gate within TIE_PP of it that costs less wins instead, and if
  the best gain is below MIN_GAIN_PP the eval data is never opened.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from ttt.gate import AdaptiveRouter

FREE = ["maxprob", "entropy_norm", "margin"]
ONE_VIEW = FREE + ["probe_agree", "probe_kl"]
FOUR_VIEWS = ONE_VIEW + ["views_agree_frac", "views_marginal_entropy_norm"]
POST = FOUR_VIEWS + ["post_maxprob", "post_entropy_norm", "post_entropy_drop",
                     "post_answer_changed", "memo_loss"]
GATES: Dict[str, List[str]] = {
    "score": [], "free": FREE, "one_view": ONE_VIEW, "four_views": FOUR_VIEWS, "post": POST,
}
TIER_PROBE_VIEWS = {"score": 0, "free": 0, "one_view": 1, "four_views": 4, "post": None}

L2 = 0.1
N_FOLDS = 5
SEED = 0
TIE_PP = 0.05
MIN_GAIN_PP = 0.2


def load_part(directory: str) -> Dict[str, np.ndarray]:
    """Per-sample skip and MEMO outcomes plus gate signals for one subset."""
    skip = np.load(os.path.join(directory, "no_adapt.npz"))
    memo = np.load(os.path.join(directory, "memo.npz"))
    sig = np.load(os.path.join(directory, "memo_signals.npz"))
    out = {
        "skip_soft": skip["soft_score"].astype(float),
        "memo_soft": memo["soft_score"].astype(float),
        "skip_correct": skip["prediction"] == skip["ground_truth"],
        "memo_correct": memo["prediction"] == memo["ground_truth"],
        "gate_score": skip["gate_score"].astype(float),
    }
    out.update({k: sig[k].astype(float) for k in sig.files})
    return out


def features(data: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([data[n] for n in names]) if names else np.zeros((len(data["skip_soft"]), 0))


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = L2):
    """Weighted L2 logistic regression on standardized X. Returns a model dict."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (X - mean) / std
    s = np.where(y, 1.0, -1.0)
    w = w / w.sum()

    def loss(theta):
        coef, b = theta[:-1], theta[-1]
        m = s * (Z @ coef + b)
        value = np.sum(w * np.logaddexp(0.0, -m)) + 0.5 * l2 * coef @ coef
        g = -w * s * (1.0 / (1.0 + np.exp(m)))
        grad = np.concatenate([Z.T @ g + l2 * coef, [g.sum()]])
        return value, grad

    res = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return {"mean": mean.tolist(), "std": std.tolist(),
            "coef": res.x[:-1].tolist(), "intercept": float(res.x[-1])}


def gate_scores(name: str, model: Optional[dict], data: Dict[str, np.ndarray]) -> np.ndarray:
    """Higher = more likely MEMO helps."""
    if name == "score":
        return data["gate_score"]
    X = features(data, GATES[name])
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["std"])
    return Z @ np.asarray(model["coef"]) + model["intercept"]


def choose_threshold(scores: np.ndarray, gain: np.ndarray) -> float:
    """Threshold maximizing sum(gain[scores >= t]); ties adapt fewer. inf = never."""
    order = np.argsort(-scores, kind="mergesort")
    cum = np.concatenate([[0.0], np.cumsum(gain[order])])
    s_sorted = np.concatenate([[np.inf], scores[order]])
    # Only cut between distinct scores so every sample with score >= t is counted.
    valid = np.concatenate([[True], np.r_[s_sorted[2:] != s_sorted[1:-1], True]])
    best = np.max(cum[valid])
    k = int(np.flatnonzero(valid & (cum >= best - 1e-12))[0])
    return float(s_sorted[k])


def _train(name: str, data: Dict[str, np.ndarray], idx: np.ndarray):
    gain = data["memo_soft"][idx] - data["skip_soft"][idx]
    model = None
    if name != "score":
        changed = gain != 0
        X = features(data, GATES[name])[idx][changed]
        model = fit_logistic(X, gain[changed] > 0, np.abs(gain[changed]))
    sub = {k: v[idx] for k, v in data.items()}
    return model, choose_threshold(gate_scores(name, model, sub), gain)


def per_sample_gflops(name: str, adapt: np.ndarray) -> np.ndarray:
    """Deployment cost of a gate's decisions (AdaptiveRouter.sample_flops, CLIP)."""
    AdaptiveRouter.configure_for_backend("clip")
    base = AdaptiveRouter.sample_flops(adapted=False) / 1e9
    full = AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1) / 1e9
    views = TIER_PROBE_VIEWS[name]
    if views is None:  # MEMO always runs; the gate only chooses the answer
        return np.full(len(adapt), full)
    probe = views * (AdaptiveRouter.IMAGE_ENCODE_FLOPS + AdaptiveRouter.FUSION_FLOPS
                     + AdaptiveRouter.PRED_FLOPS) / 1e9
    return np.where(adapt, full, base + probe)


def cross_validate(name: str, data: Dict[str, np.ndarray],
                   n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
    """Held-out gated gain, adapt rate, cost and helped-vs-hurt AUROC on gate_train."""
    from ttt.score_gate import binary_auroc

    n = len(data["skip_soft"])
    folds = np.random.default_rng(seed).permutation(n) % n_folds
    gain = data["memo_soft"] - data["skip_soft"]
    adapt = np.zeros(n, dtype=bool)
    held_scores = np.zeros(n)
    for f in range(n_folds):
        train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        model, t = _train(name, data, train)
        sub = {k: v[test] for k, v in data.items()}
        s = gate_scores(name, model, sub)
        held_scores[test] = s
        adapt[test] = s >= t
    changed = gain != 0
    return {
        "gain_pp": round(100.0 * float(np.where(adapt, gain, 0.0).sum()) / n, 3),
        "adapt_rate": round(float(adapt.mean()), 4),
        "avg_gflops": round(float(per_sample_gflops(name, adapt).mean()), 2),
        "auroc_helped_vs_hurt": round(binary_auroc(gain[changed] > 0, held_scores[changed]), 3),
    }


def select_gate(cv: Dict[str, dict], tie_pp: float = TIE_PP,
                min_gain_pp: float = MIN_GAIN_PP) -> Tuple[Optional[str], dict]:
    """The pre-registered choice; None means stop before opening the eval data."""
    best = max(r["gain_pp"] for r in cv.values())
    near = [k for k, r in cv.items() if r["gain_pp"] >= best - tie_pp]
    chosen = min(near, key=lambda k: (cv[k]["avg_gflops"], -cv[k]["gain_pp"]))
    proceed = best >= min_gain_pp
    return (chosen if proceed else None), {
        "rule": (f"best cross-validated gate_train gain; a gate within {tie_pp} pp of it "
                 f"that costs less wins; stop if the best gain is under {min_gain_pp} pp"),
        "best_gain_pp": best,
        "candidate": chosen,
        "proceed": proceed,
    }


def fit_final(name: str, data: Dict[str, np.ndarray]) -> dict:
    """Refit the chosen gate on all of gate_train and freeze it."""
    model, t = _train(name, data, np.arange(len(data["skip_soft"])))
    return {"name": name, "features": GATES[name], "model": model, "threshold": t,
            "probe_views": TIER_PROBE_VIEWS[name]}


def apply_gate(gate: dict, data: Dict[str, np.ndarray]) -> np.ndarray:
    """Adapt mask for a frozen gate."""
    return gate_scores(gate["name"], gate["model"], data) >= gate["threshold"]
