"""
Phase 2 evaluation loop over a multi-view cache.

Charges AdaptiveRouter.sample_flops on the deployment path (never cache-read
cost). Official soft credit is answer_scores[pred]; index 0 (<UNK>) is zero
when OOV votes were dropped, matching Phase 1.

Streams one sample at a time so an 8k fp16 cache is not stacked as float32
(~24 GB). Tensors are moved to the model's device.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from ttt.gate import AdaptiveRouter
from ttt.score_gate import ScoreWeights, aurc, binary_auroc, signals_from_logits, weighted_score
from ttt.shift_cache import ORIGINAL_VIEW
from ttt.tta import TTAAdapter


ProgressFn = Callable[[int, int], None]
SampleSource = Union[Mapping[str, Any], Sequence[Any]]


def order_methods(methods: Sequence[str]) -> list:
    """Run no_adapt and memo before gated_memo_sar so τ can be tuned first."""
    methods = list(methods)
    seen = set()
    ordered = []
    for name in ("no_adapt", "memo"):
        if name in methods and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in methods:
        if name == "gated_memo_sar" or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    if "gated_memo_sar" in methods:
        ordered.append("gated_memo_sar")
    return ordered


def _n_samples(samples: SampleSource) -> int:
    if isinstance(samples, Mapping) and "visual_tokens" in samples:
        return int(samples["visual_tokens"].shape[0])
    return len(samples)


def _get_sample(samples: SampleSource, index: int) -> Dict[str, Any]:
    if isinstance(samples, Mapping) and "visual_tokens" in samples:
        item = {
            "visual_tokens": samples["visual_tokens"][index],
            "text_tokens": samples["text_tokens"][index],
            "attention_mask": samples["attention_mask"][index],
            "answer_idx": samples["answer_idx"][index],
        }
        scores = samples.get("answer_scores")
        if scores is not None:
            item["answer_scores"] = scores[index]
        return item
    return samples[index]


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=device.type == "cuda")


def evaluate_condition(
    model: torch.nn.Module,
    samples: SampleSource,
    method: str = "no_adapt",
    adapter: Optional[TTAAdapter] = None,
    tau: Optional[float] = None,
    weights: Optional[ScoreWeights] = None,
    k_steps: int = 1,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, np.ndarray]:
    """Run one method, streaming samples (CPU-safe tests, CUDA on the VM).

    `samples` is either a batched dict with `visual_tokens` (N, 5, tokens, dim)
    or a Dataset/sequence yielding per-sample dicts. View 0 is the original.
    """
    weights = weights or ScoreWeights()
    n = _n_samples(samples)
    device = next(model.parameters()).device

    preds = np.empty(n, dtype=np.int32)
    gts = np.empty(n, dtype=np.int32)
    soft = np.empty(n, dtype=np.float32)
    adapted = np.empty(n, dtype=np.uint8)
    gate_score = np.empty(n, dtype=np.float32)
    flops_g = np.empty(n, dtype=np.float32)
    maxprob = np.empty(n, dtype=np.float32)

    model.eval()
    for i in range(n):
        sample = _get_sample(samples, i)
        vis = sample["visual_tokens"]
        if vis.dim() == 4:
            vis = vis[0]
        vis = _to_device(vis.float(), device)
        orig = vis[ORIGINAL_VIEW : ORIGINAL_VIEW + 1]
        views = vis[1:]
        text = sample["text_tokens"]
        if text.dim() == 2:
            text = text.unsqueeze(0)
        elif text.dim() == 3:
            text = text[:1]
        text = _to_device(text.float(), device)
        mask = sample["attention_mask"]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 2:
            mask = mask[:1]
        mask = _to_device(mask.long(), device)

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
        gts[i] = int(sample["answer_idx"])
        scores = sample.get("answer_scores")
        if scores is None or pred >= (scores.shape[-1] if hasattr(scores, "shape") else len(scores)):
            soft[i] = 0.0
        else:
            soft[i] = float(scores[pred])
        adapted[i] = 1 if do_adapt else 0
        flops_g[i] = AdaptiveRouter.sample_flops(
            adapted=bool(do_adapt),
            n_aug=n_aug if do_adapt else 0,
            k_steps=k_steps,
        ) / 1e9
        if on_progress is not None and ((i + 1) % 50 == 0 or i + 1 == n):
            on_progress(i + 1, n)

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


def merge_condition_outputs(
    parts: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Concatenate streamed chunks. Scalars (AUROC/AURC) are recomputed."""
    if not parts:
        raise ValueError("merge_condition_outputs needs at least one chunk")
    keys = ("prediction", "ground_truth", "soft_score", "adapted", "gate_score", "flops_g", "maxprob")
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    exact_correct = merged["prediction"] == merged["ground_truth"]
    merged["maxprob_auroc"] = binary_auroc(exact_correct, merged["maxprob"])
    merged["maxprob_aurc"] = aurc(merged["maxprob"], exact_correct.astype(float))
    merged["score_auroc"] = binary_auroc(~exact_correct.astype(bool), merged["gate_score"])
    return merged
