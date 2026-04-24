#!/usr/bin/env python3
"""
Evaluate Phase 2 TTA methods against a VM-local 5-view cache.

Writes compact npz + JSON aggregates to results/phase2/. Does not allocate a
VM, does not pull caches to the Mac, does not touch Phase 1 checkpoints
except to load weights read-only.

    python gpu/eval_phase2.py \\
        --features /content/phase2_cache/blur_s3.pt \\
        --checkpoint checkpoints/phase1_clip/best.pt \\
        --output results/phase2/blur_s3
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.gate import AdaptiveRouter
from ttt.models import FullVQAModel
from ttt.phase2_eval import evaluate_condition
from ttt.phase2_report import (
    compact_outcomes,
    save_outcomes_npz,
    write_phase2_report,
)
from ttt.score_gate import ScoreWeights, tune_tau
from ttt.shift_cache import (
    CachePathError,
    MultiViewCachedFeaturesDataset,
    assert_cache_path_allowed,
    multiview_collate_fn,
)
from ttt.tta import TTAAdapter
from ttt.utils import load_checkpoint, load_config, set_seed, setup_logging


METHODS = ("no_adapt", "tent", "eata", "memo", "memo_sar", "gated_memo_sar")


def _load_all(ds: MultiViewCachedFeaturesDataset) -> dict:
    loader = DataLoader(
        ds, batch_size=len(ds), shuffle=False, collate_fn=multiview_collate_fn
    )
    return next(iter(loader))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 TTA eval (no VM alloc)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--features", type=str, required=True,
                        help="VM-local 5-view cache (.pt)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/phase2")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--tau", type=float, default=None,
                        help="If omitted, tuned on this condition's base vs MEMO")
    parser.add_argument("--source", type=str, default="corruption",
                        help="τ-tuning source tag; VQA-CP is rejected")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        features_path = assert_cache_path_allowed(args.features)
    except CachePathError:
        raise

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    logger = setup_logging("logs")
    backend = config.get("encoder_backend", "clip")
    AdaptiveRouter.configure_for_backend(backend)

    model = FullVQAModel(config)
    # Read-only load of Phase 1 weights. Never writes checkpoints or retrains.
    load_checkpoint(model, args.checkpoint)
    model.eval()

    ds = MultiViewCachedFeaturesDataset(features_path)
    samples = _load_all(ds)
    logger.info("Loaded %d samples from %s", len(ds), features_path)

    weights = ScoreWeights()
    runs = []
    os.makedirs(args.output, exist_ok=True)

    base_soft = None
    memo_soft = None
    gate_scores = None

    for method in args.methods:
        adapter = None
        if method != "no_adapt":
            tta_method = "memo_sar" if method == "gated_memo_sar" else method
            adapter = TTAAdapter(
                model, config, method=tta_method, k_steps=args.k, layernorm_only=True
            )
        out = evaluate_condition(
            model,
            samples,
            method=method,
            adapter=adapter,
            tau=args.tau if method == "gated_memo_sar" else None,
            weights=weights,
            k_steps=args.k,
        )
        if method == "no_adapt":
            base_soft = out["soft_score"].copy()
            gate_scores = out["gate_score"].copy()
        if method == "memo":
            memo_soft = out["soft_score"].copy()

        packed = compact_outcomes(
            prediction=out["prediction"],
            ground_truth=out["ground_truth"],
            soft_score=out["soft_score"],
            adapted=out["adapted"],
            gate_score=out["gate_score"],
            flops_g=out["flops_g"],
        )
        save_outcomes_npz(os.path.join(args.output, f"{method}.npz"), **packed)

        exact = float((out["prediction"] == out["ground_truth"]).mean())
        soft = float(out["soft_score"].mean())
        adapt_rate = float(out["adapted"].mean())
        runs.append({
            "config": method,
            "method": method,
            "accuracy": soft,
            "soft": soft,
            "exact": exact,
            "avg_flops": float(out["flops_g"].mean()),
            "flops": out["flops_g"],
            "adapt_rate": adapt_rate,
            "maxprob_auroc": float(out["maxprob_auroc"]) if out["maxprob_auroc"] == out["maxprob_auroc"] else None,
            "maxprob_aurc": float(out["maxprob_aurc"]),
        })
        logger.info(
            "%s soft=%.4f exact=%.4f adapt=%.3f flops=%.1fG",
            method, soft, exact, adapt_rate, out["flops_g"].mean(),
        )

    if args.tau is None and base_soft is not None and memo_soft is not None and gate_scores is not None:
        chosen = tune_tau(gate_scores, base_soft, memo_soft, source=args.source)
        with open(os.path.join(args.output, "tau.json"), "w") as fh:
            import json
            json.dump(chosen, fh, indent=2)
        logger.info("Tuned τ=%s on %s (not VQA-CP)", chosen["tau"], args.source)

    write_phase2_report(runs, args.output, backend=backend)
    logger.info("Wrote report under %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
