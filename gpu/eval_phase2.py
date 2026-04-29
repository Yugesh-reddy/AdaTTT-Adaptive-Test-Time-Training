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
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.gate import AdaptiveRouter
from ttt.models import FullVQAModel
from ttt.phase2_eval import evaluate_condition, order_methods
from ttt.phase2_report import (
    compact_outcomes,
    oracle_recovery,
    save_outcomes_npz,
    write_phase2_report,
)
from ttt.score_gate import ScoreWeights, tune_tau
from ttt.shift_cache import (
    CachePathError,
    MultiViewCachedFeaturesDataset,
    assert_cache_path_allowed,
)
from ttt.tta import TTAAdapter
from ttt.utils import get_device, load_checkpoint, load_config, set_seed, setup_logging


METHODS = ("no_adapt", "tent", "eata", "memo", "memo_sar", "gated_memo_sar")


def _write_progress(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def resolve_tau_protocol(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Optional[Dict[str, Any]]:
    """Decide where the gate threshold comes from, before anything is loaded.

    The eval 8k is report-only. A τ picked by looking at the outcomes being
    reported makes every gated number an in-sample upper bound, and the gate
    "abstaining correctly" a tautology. The held-out path is --fit-tau on
    gate_train_subset_8k under the same corruption, then --tau-file here.
    """
    methods = set(args.methods)
    chosen = [flag for flag, on in (("--tau", args.tau is not None),
                                    ("--tau-file", args.tau_file is not None),
                                    ("--fit-tau", args.fit_tau)) if on]
    if len(chosen) > 1:
        parser.error(f"pick one τ source, got {' and '.join(chosen)}")
    if args.fit_tau and not {"no_adapt", "memo_sar"} <= methods:
        parser.error("--fit-tau tunes τ on no_adapt vs memo_sar, the adapter the gate "
                     "runs; include both methods")
    if "gated_memo_sar" in methods and not chosen:
        parser.error("gated_memo_sar needs a τ source: --tau-file with a tau.json from a "
                     "--fit-tau run on gate_train_subset_8k (held out), --tau for a fixed "
                     "value, or --fit-tau to tune on this run (gated numbers are then "
                     "in-sample).")
    if args.tau_file is not None:
        with open(args.tau_file) as fh:
            rec = json.load(fh)
        if rec.get("source") == args.source:
            parser.error(f"{args.tau_file} was fit on source '{args.source}', the data "
                         "being reported. Fit τ on gate_train_subset_8k instead.")
        if (rec.get("lr") is not None and args.lr is not None
                and abs(float(rec["lr"]) - args.lr) > 1e-12):
            parser.error(f"{args.tau_file} was fit at lr {rec['lr']}, not --lr {args.lr}: "
                         "a threshold fit for one step size does not carry to another.")
        return {
            "protocol": "held_out",
            "tau": float(rec["tau"]),
            "tuned_on": rec.get("source"),
            "tau_file": args.tau_file,
            "target_method": rec.get("target_method"),
            "fit_gated_metric": rec.get("gated_metric"),
            "fit_adapt_rate": rec.get("adapt_rate"),
            "lr": rec.get("lr"),
        }
    if args.tau is not None:
        return {"protocol": "fixed", "tau": float(args.tau)}
    return None


def fit_tau_record(gate_scores, base_soft, adapted_soft, source: str, reported_here: bool) -> Dict[str, Any]:
    """Tune τ against memo_sar, the adapter gated_memo_sar actually runs."""
    chosen = tune_tau(gate_scores, base_soft, adapted_soft, source=source)
    chosen["target_method"] = "memo_sar"
    chosen["protocol"] = "in_sample" if reported_here else "fit"
    if reported_here:
        chosen["note"] = ("τ was selected on the outcomes being reported; gated numbers "
                          "from this run are an in-sample upper bound.")
    return chosen


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 TTA eval (no VM alloc)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--features", type=str, required=True,
                        help="VM-local 5-view cache (.pt)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/phase2")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--tau", type=float, default=None,
                        help="Fixed τ for gated_memo_sar (protocol 'fixed')")
    parser.add_argument("--tau-file", type=str, default=None,
                        help="tau.json from a --fit-tau run on gate_train_subset_8k "
                             "(protocol 'held_out')")
    parser.add_argument("--fit-tau", action="store_true",
                        help="Fit τ on this run's no_adapt vs memo_sar outcomes and write "
                             "tau.json. Run it on gate_train_subset_8k; with "
                             "gated_memo_sar in the same run the gated numbers are in-sample")
    parser.add_argument("--source", type=str, default="corruption",
                        help="τ-tuning source tag; VQA-CP is rejected")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None,
                        help="Adam step size for the adapters (default: config ttt_lr)")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Optional JSON heartbeat (VM probe); never a cache path")
    args = parser.parse_args(argv)
    tau_info = resolve_tau_protocol(args, parser)

    try:
        features_path = assert_cache_path_allowed(args.features)
    except CachePathError:
        raise

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    logger = setup_logging("logs")
    backend = config.get("encoder_backend", "clip")
    lr = float(args.lr if args.lr is not None else config.get("ttt_lr", 1e-4))
    AdaptiveRouter.configure_for_backend(backend)
    device = get_device()

    model = FullVQAModel(config)
    # Read-only load of Phase 1 weights. Never writes checkpoints or retrains.
    try:
        load_checkpoint(model, args.checkpoint)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.fusion.load_state_dict(ckpt["fusion"])
        model.gate.load_state_dict(ckpt["gate"])
        model.prediction_head.load_state_dict(ckpt["prediction_head"])
    model.to(device)
    model.eval()

    # Stream the cache. Do not stack 8k rows as float32 (~24 GB).
    ds = MultiViewCachedFeaturesDataset(features_path)
    logger.info("Streaming %d samples from %s on %s", len(ds), features_path, device)

    weights = ScoreWeights()
    runs = []
    os.makedirs(args.output, exist_ok=True)

    base_soft = None
    base_pred = None
    memo_soft = None
    memo_sar_soft = None
    gate_scores = None
    tau = tau_info["tau"] if tau_info else None
    methods = order_methods(args.methods)

    for method in methods:
        if method == "gated_memo_sar" and tau is None:
            # Only reachable with --fit-tau: resolve_tau_protocol refused the rest.
            tau_info = fit_tau_record(gate_scores, base_soft, memo_sar_soft,
                                      source=args.source, reported_here=True)
            tau_info["lr"] = lr
            tau = float(tau_info["tau"])
            logger.warning("τ=%.4f tuned on the reported outcomes (%s): gated numbers "
                           "are in-sample", tau, args.source)

        adapter = None
        if method != "no_adapt":
            tta_method = "memo_sar" if method == "gated_memo_sar" else method
            adapter = TTAAdapter(
                model, config, method=tta_method, k_steps=args.k, layernorm_only=True,
                lr=lr,
            )

        def _tick(done: int, total: int, method=method) -> None:
            _write_progress(args.progress_file, {
                "stage": "eval",
                "step": f"{method} {done}/{total}",
                "method": method,
                "n": done,
                "n_total": total,
                "done": False,
                "crash": False,
            })

        _write_progress(args.progress_file, {
            "stage": "eval",
            "step": f"{method} 0/{len(ds)}",
            "method": method,
            "n": 0,
            "n_total": len(ds),
            "done": False,
            "crash": False,
        })
        out = evaluate_condition(
            model,
            ds,
            method=method,
            adapter=adapter,
            tau=tau if method == "gated_memo_sar" else None,
            weights=weights,
            k_steps=args.k,
            on_progress=_tick,
        )
        if method == "no_adapt":
            base_soft = out["soft_score"].copy()
            base_pred = out["prediction"].copy()
            gate_scores = out["gate_score"].copy()
        if method == "memo":
            memo_soft = out["soft_score"].copy()
        if method == "memo_sar":
            memo_sar_soft = out["soft_score"].copy()

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
        n_total = len(out["prediction"])
        # How far the adapter actually moved the model. An adapter that changes
        # <1% of answers cannot show a TTA effect either way.
        flips = None
        if base_pred is not None and method != "no_adapt":
            flips = int((out["prediction"] != base_pred).sum())
            if adapt_rate > 0 and flips < 0.01 * n_total:
                logger.warning("%s changed %d of %d predictions (<1%%): the adapter "
                               "barely moves the model; check the step size before "
                               "reading this as a TTA result", method, flips, n_total)
        gate_pass = None
        if method == "gated_memo_sar":
            # score >= τ passes the gate; SAR's entropy filter can still refuse,
            # so the realized adapt rate can be lower than the pass rate.
            gate_pass = float((out["gate_score"] >= tau).mean())
        runs.append({
            "lr": None if method == "no_adapt" else lr,
            "pred_flips_vs_no_adapt": flips,
            "gate_pass_rate": gate_pass,
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

    if args.fit_tau and tau_info is None:
        # Fit-only run (gate_train_subset_8k): the τ is for a different split.
        tau_info = fit_tau_record(gate_scores, base_soft, memo_sar_soft,
                                  source=args.source, reported_here=False)
        tau_info["lr"] = lr
        logger.info("Fit τ=%.4f on %s for a held-out eval", tau_info["tau"], args.source)
    if tau_info is not None:
        with open(os.path.join(args.output, "tau.json"), "w") as fh:
            json.dump(tau_info, fh, indent=2)

    oracle = None
    if base_soft is not None and memo_soft is not None:
        oracle = oracle_recovery(base_soft, memo_soft)
    write_phase2_report(runs, args.output, backend=backend, oracle=oracle, tau=tau_info)
    logger.info("Wrote report under %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
