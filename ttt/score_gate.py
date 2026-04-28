"""
Weighted-score gate and MaxProb AUROC/AURC for Phase 2.

The gate is a weighted sum of signals (entropy, 1-MaxProb, margin), compared
to a single τ — not a conjunction of thresholds. τ is tuned on corruption
conditions only; VQA-CP test is rejected so we cannot Goodhart it.

In gpu/eval_phase2.py the gate routes into memo_sar, whose SAR filter refuses
high-entropy samples, while this score rises with entropy. The rule actually
applied is score >= τ AND entropy < 0.4·ln C, so eval records gate_pass_rate
next to adapt_rate, and τ is tuned against memo_sar outcomes. The held-out
protocol fits τ on gate_train_subset_8k, never on the eval 8k it reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from ttt.tta import shannon_entropy


@dataclass
class ScoreWeights:
    entropy: float = 0.5
    maxprob: float = 0.5
    margin: float = 0.0


def signals_from_logits(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """MaxProb, entropy, normalized entropy, and top-1/top-2 margin."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    maxprob, _ = probs.max(dim=-1)
    entropy = shannon_entropy(probs, dim=-1)
    entropy_norm = entropy / math.log(logits.size(-1))
    k = min(2, logits.size(-1))
    top = probs.topk(k, dim=-1).values
    margin = top[:, 0] - top[:, 1] if k == 2 else top[:, 0]
    return {
        "probs": probs,
        "maxprob": maxprob,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "margin": margin,
    }


def weighted_score(
    signals: Dict[str, torch.Tensor],
    weights: ScoreWeights,
) -> torch.Tensor:
    """Higher score → more likely to adapt. A sum, not an AND of gates."""
    margin = signals["margin"].clamp(min=0.0, max=1.0)
    return (
        weights.entropy * signals["entropy_norm"]
        + weights.maxprob * (1.0 - signals["maxprob"])
        + weights.margin * (1.0 - margin)
    )


def route_adapt(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """True = ADAPT. Single threshold on the weighted score."""
    return scores >= tau


def _is_vqacp_source(source: str) -> bool:
    compact = source.lower().replace("-", "").replace("_", "")
    return "vqacp" in compact


def tune_tau(
    scores: Union[np.ndarray, Sequence[float]],
    base_metric: Union[np.ndarray, Sequence[float]],
    adapted_metric: Union[np.ndarray, Sequence[float]],
    source: str,
    taus: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Pick τ on a corruption split. VQA-CP test is forbidden."""
    if _is_vqacp_source(source):
        raise ValueError(
            "Do not tune τ on VQA-CP test (Goodhart). Tune on corruptions only."
        )
    scores_np = np.asarray(scores, dtype=float)
    base = np.asarray(base_metric, dtype=float)
    adapted = np.asarray(adapted_metric, dtype=float)
    if taus is None:
        grid = np.linspace(0.0, 1.0, 101)
        taus = np.unique(np.concatenate([grid, scores_np]))
    best: Optional[Dict[str, Any]] = None
    for tau in taus:
        adapt = scores_np >= float(tau)
        metric = float(np.where(adapt, adapted, base).mean()) if len(scores_np) else 0.0
        rec = {
            "tau": float(tau),
            "gated_metric": metric,
            "adapt_rate": float(adapt.mean()) if len(scores_np) else 0.0,
            "source": source,
        }
        if best is None:
            best = rec
            continue
        if rec["gated_metric"] > best["gated_metric"] + 1e-12:
            best = rec
        elif abs(rec["gated_metric"] - best["gated_metric"]) <= 1e-12 and rec["adapt_rate"] < best["adapt_rate"]:
            best = rec
    assert best is not None
    return best


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via Mann–Whitney U with average ranks for ties."""
    labels_b = np.asarray(labels).astype(bool)
    scores_a = np.asarray(scores, dtype=float)
    n_pos = int(labels_b.sum())
    n_neg = int((~labels_b).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores_a, kind="mergesort")
    ranks = np.empty(len(scores_a), dtype=float)
    i = 0
    n = len(scores_a)
    while i < n:
        j = i
        while j + 1 < n and scores_a[order[j + 1]] == scores_a[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    u = ranks[labels_b].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    if n == 0:
        return 0.0
    order = np.argsort(-confidence, kind="mergesort")
    correct = correct[order]
    coverage = np.arange(1, n + 1, dtype=float) / n
    risk = 1.0 - np.cumsum(correct) / np.arange(1, n + 1, dtype=float)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(risk, coverage))


def maxprob_metrics(logits: torch.Tensor, correct: np.ndarray) -> Dict[str, Any]:
    """MaxProb as a selective-prediction baseline (AUROC / AURC)."""
    sig = signals_from_logits(logits)
    maxp = sig["maxprob"].detach().cpu().numpy()
    correct_b = np.asarray(correct).astype(bool)
    return {
        "auroc": binary_auroc(correct_b, maxp),
        "aurc": aurc(maxp, correct_b.astype(float)),
        "maxprob": maxp,
    }
