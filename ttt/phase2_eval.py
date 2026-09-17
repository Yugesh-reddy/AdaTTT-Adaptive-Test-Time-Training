"""
Phase 2 evaluation loop over a multi-view cache.

Charges AdaptiveRouter.sample_flops on the deployment path (never cache-read
cost). Official soft credit is answer_scores[pred]; index 0 (<UNK>) is zero
when OOV votes were dropped, matching Phase 1.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ttt.gate import AdaptiveRouter
from ttt.score_gate import ScoreWeights, aurc, binary_auroc, signals_from_logits, weighted_score
from ttt.shift_cache import ORIGINAL_VIEW
from ttt.tta import TTAAdapter


def evaluate_condition(
    model: torch.nn.Module,
    samples: Dict[str, Any],
    method: str = "no_adapt",
    adapter: Optional[TTAAdapter] = None,
    tau: Optional[float] = None,
    weights: Optional[ScoreWeights] = None,
    k_steps: int = 1,
) -> Dict[str, np.ndarray]:
    """Run one method on in-memory multi-view tensors (CPU-safe).

    `samples["visual_tokens"]` is (N, 5, tokens, dim): view 0 = original.
    """
    weights = weights or ScoreWeights()
    vis_all = samples["visual_tokens"]
    text_all = samples["text_tokens"]
    mask_all = samples["attention_mask"]
    scores_all = samples.get("answer_scores")
    answers = samples["answer_idx"]
    n = vis_all.shape[0]
    n_answers = scores_all.shape[1] if scores_all is not None else None

    preds = np.empty(n, dtype=np.int32)
    gts = np.empty(n, dtype=np.int32)
    soft = np.empty(n, dtype=np.float32)
    adapted = np.empty(n, dtype=np.uint8)
    gate_score = np.empty(n, dtype=np.float32)
    flops_g = np.empty(n, dtype=np.float32)
    maxprob = np.empty(n, dtype=np.float32)

    model.eval()
    for i in range(n):
        orig = vis_all[i, ORIGINAL_VIEW : ORIGINAL_VIEW + 1]
        views = vis_all[i, 1:]
        text = text_all[i : i + 1]
        mask = mask_all[i : i + 1]
        with torch.no_grad():
            z = model.fusion(orig, text, mask)
            base_logits = model.prediction_head(z)
        sig = signals_from_logits(base_logits)
        score = float(weighted_score(sig, weights)[0].item())
        gate_score[i] = score
        maxprob[i] = float(sig["maxprob"][0].item())

        do_adapt = False
        n_aug = 0
        logits = base_logits
        if method == "no_adapt":
            do_adapt = False
        elif method == "gated_memo_sar":
            if adapter is None:
                raise ValueError("gated_memo_sar requires an adapter")
            threshold = 0.0 if tau is None else float(tau)
            if score >= threshold:
                logits, info = adapter.adapt_and_predict(
                    orig, text, mask, visual_views=views
                )
                do_adapt = bool(info["adapted"])
                n_aug = int(info["n_aug"])
        else:
            if adapter is None:
                raise ValueError(f"{method} requires an adapter")
            use_views = views if adapter.n_aug else None
            logits, info = adapter.adapt_and_predict(
                orig, text, mask, visual_views=use_views
            )
            do_adapt = bool(info["adapted"])
            n_aug = int(info["n_aug"])

        pred = int(logits.argmax(dim=-1)[0].item())
        preds[i] = pred
        gts[i] = int(answers[i])
        if scores_all is None or n_answers is None or pred >= n_answers:
            soft[i] = 0.0
        else:
            soft[i] = float(scores_all[i, pred])
        adapted[i] = 1 if do_adapt else 0
        flops_g[i] = AdaptiveRouter.sample_flops(
            adapted=bool(do_adapt),
            n_aug=n_aug if do_adapt else 0,
            k_steps=k_steps,
        ) / 1e9

    exact_correct = preds == gts
    return {
        "prediction": preds,
        "ground_truth": gts,
        "soft_score": soft,
        "adapted": adapted,
        "gate_score": gate_score,
        "flops_g": flops_g,
        "maxprob": maxprob,
        "maxprob_auroc": binary_auroc(exact_correct, maxprob),
        "maxprob_aurc": aurc(maxprob, exact_correct.astype(float)),
        "score_auroc": binary_auroc(~exact_correct.astype(bool), gate_score),
    }
