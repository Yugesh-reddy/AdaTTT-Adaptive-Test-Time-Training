#!/usr/bin/env python3
"""
Compute all metrics from saved predictions. No GPU needed.

Usage:
    python scripts/02_analyze_results.py --config config/config.yaml
    python scripts/02_analyze_results.py --results-dir results/

Computes:
    1. Overall VQA accuracy for each configuration (K, objective, τ)
    2. Accuracy by question type (yes/no, number, other)
    3. Average FLOPs per sample for each configuration
    4. Pareto frontier points
    5. Gate routing statistics (% skip, % adapt, accuracy per group)
    6. McNemar's test: base vs best TTT configuration
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.utils import load_config, load_json, save_json
from ttt.metrics import (
    vqa_accuracy,
    accuracy_by_question_type,
    pareto_frontier,
    compute_gate_statistics,
    mcnemar_test,
)
from ttt.gate import AdaptiveRouter


def parse_config_from_filename(filename: str) -> dict:
    """Extract K, objective, threshold from result filename."""
    basename = os.path.basename(filename).replace(".json", "")

    config = {"filename": basename}

    # Parse k value
    k_match = re.search(r"k(\d+)", basename)
    if k_match:
        config["k"] = int(k_match.group(1))

    # Parse objective
    for obj in ["masked_patch", "rotation"]:
        if obj in basename:
            config["objective"] = obj

    # Parse threshold
    t_match = re.search(r"t([\d.]+)", basename)
    if t_match:
        config["threshold"] = float(t_match.group(1))

    return config


def main():
    parser = argparse.ArgumentParser(description="Analyze TTT results")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Results directory"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = args.results_dir or config.get("results_dir", "results/")

    print("=" * 70)
    print("  Efficient TTT — Results Analysis")
    print("=" * 70)

    all_results = []

    # 1. Base model predictions
    base_path = os.path.join(results_dir, "base_predictions.json")
    if os.path.exists(base_path):
        base_data = load_json(base_path)
        preds = [d["prediction"] for d in base_data]
        gts = [d["ground_truth"] for d in base_data]
        qtypes = [d.get("question_type", "other") for d in base_data]

        base_acc = vqa_accuracy(preds, gts)
        type_acc = accuracy_by_question_type(preds, gts, qtypes)

        print(f"\n📊 Base Model (no TTT)")
        print(f"   Overall accuracy: {base_acc*100:.2f}%")
        for qt, acc in type_acc.items():
            print(f"   {qt}: {acc*100:.2f}%")

        all_results.append({
            "config": "K=0 (no TTT)",
            "accuracy": base_acc,
            "avg_flops": AdaptiveRouter.SKIP_FLOPS / 1e9,
            "type_accuracy": type_acc,
        })

    # 2. TTT sweep predictions
    ttt_dir = os.path.join(results_dir, "ttt_predictions")
    if os.path.exists(ttt_dir):
        ttt_files = sorted(glob.glob(os.path.join(ttt_dir, "*.json")))
        for ttt_file in ttt_files:
            file_config = parse_config_from_filename(ttt_file)
            ttt_data = load_json(ttt_file)
            preds = [d["prediction"] for d in ttt_data]
            gts = [d["ground_truth"] for d in ttt_data]
            qtypes = [d.get("question_type", "other") for d in ttt_data]

            acc = vqa_accuracy(preds, gts)
            type_acc = accuracy_by_question_type(preds, gts, qtypes)
            k = file_config.get("k", 0)
            objective = file_config.get("objective", "unknown")

            flops = AdaptiveRouter.SKIP_FLOPS / 1e9 + k * AdaptiveRouter.TTT_STEP_FLOPS / 1e9

            print(f"\n📊 TTT: K={k}, {objective}")
            print(f"   Overall accuracy: {acc*100:.2f}%")
            for qt, a in type_acc.items():
                print(f"   {qt}: {a*100:.2f}%")
            print(f"   Est. GFLOPs/sample: {flops:.1f}")

            config_str = f"K={k}, {objective}"
            all_results.append({
                "config": config_str,
                "accuracy": acc,
                "avg_flops": flops,
                "type_accuracy": type_acc,
            })

    # 3. Adaptive inference results
    adaptive_files = sorted(glob.glob(os.path.join(results_dir, "adaptive_*.json")))
    for af in adaptive_files:
        file_config = parse_config_from_filename(af)
        adapt_data = load_json(af)
        preds = [d["prediction"] for d in adapt_data]
        gts = [d["ground_truth"] for d in adapt_data]

        acc = vqa_accuracy(preds, gts)
        threshold = file_config.get("threshold", 0.0)
        k = file_config.get("k", 0)

        # Compute routing stats if available
        skip = sum(1 for d in adapt_data if d.get("skipped", False))
        adapt = len(adapt_data) - skip
        skip_rate = skip / len(adapt_data) if adapt_data else 0

        adapt_flops = (AdaptiveRouter.SKIP_FLOPS + k * AdaptiveRouter.TTT_STEP_FLOPS) / 1e9
        avg_flops = (skip * AdaptiveRouter.SKIP_FLOPS / 1e9 + adapt * adapt_flops) / max(len(adapt_data), 1)

        print(f"\n📊 Adaptive: τ={threshold}, K={k}")
        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   Skip rate: {skip_rate*100:.1f}%")
        print(f"   Avg GFLOPs: {avg_flops:.1f}")

        all_results.append({
            "config": f"Adaptive τ={threshold}, K={k}",
            "accuracy": acc,
            "avg_flops": avg_flops,
        })

    # 4. Pareto frontier
    if all_results:
        pareto_points = pareto_frontier(all_results)
        print(f"\n{'='*70}")
        print(f"  Pareto Frontier ({len(pareto_points)} points)")
        print(f"{'='*70}")
        for p in pareto_points:
            print(f"  {p['config']}: acc={p['accuracy']*100:.2f}%, FLOPs={p['avg_flops']:.1f} G")

    # 5. McNemar's test (base vs best TTT)
    if os.path.exists(base_path) and os.path.exists(ttt_dir):
        ttt_files = sorted(glob.glob(os.path.join(ttt_dir, "*.json")))
        if ttt_files:
            best_ttt_file = ttt_files[0]  # Use first available
            ttt_data = load_json(best_ttt_file)
            base_data_reload = load_json(base_path)

            # Align by sample_id
            ttt_map = {d["sample_id"]: d for d in ttt_data}
            aligned_base = []
            aligned_ttt = []
            aligned_gt = []
            for d in base_data_reload:
                sid = d["sample_id"]
                if sid in ttt_map:
                    aligned_base.append(d["prediction"])
                    aligned_ttt.append(ttt_map[sid]["prediction"])
                    aligned_gt.append(d["ground_truth"])

            if aligned_base:
                mcnemar = mcnemar_test(aligned_base, aligned_ttt, aligned_gt)
                print(f"\n{'='*70}")
                print(f"  McNemar's Test (Base vs TTT)")
                print(f"{'='*70}")
                print(f"  Base ✓, TTT ✗: {mcnemar['b_base_correct_ttt_wrong']}")
                print(f"  Base ✗, TTT ✓: {mcnemar['c_base_wrong_ttt_correct']}")
                print(f"  χ² = {mcnemar['chi2']:.3f}, p = {mcnemar['p_value']:.4f}")
                print(f"  Significant (α=0.05): {mcnemar['significant_at_005']}")

    # Save summary
    summary = {"results": all_results}
    if all_results:
        summary["pareto_frontier"] = pareto_frontier(all_results)

    # 6. Memotion2 cross-task results
    memotion2_dir = os.path.join(results_dir, "memotion2")
    memotion2_results = []
    if os.path.exists(memotion2_dir):
        memo_files = sorted(glob.glob(os.path.join(memotion2_dir, "*.json")))
        if memo_files:
            print(f"\n{'='*70}")
            print(f"  Memotion2 Cross-Task Evaluation")
            print(f"{'='*70}")

        for mf in memo_files:
            file_config = parse_config_from_filename(mf)
            memo_data = load_json(mf)
            preds = [d["prediction"] for d in memo_data]
            gts = [d["ground_truth"] for d in memo_data]

            acc = vqa_accuracy(preds, gts)
            k = file_config.get("k", 0)
            objective = file_config.get("objective", "unknown")

            print(f"\n📊 Memotion2: K={k}, {objective}")
            print(f"   Accuracy: {acc*100:.2f}%")
            print(f"   Samples: {len(memo_data)}")

            memotion2_results.append({
                "config": f"Memotion2 K={k}, {objective}",
                "accuracy": acc,
                "num_samples": len(memo_data),
            })

        summary["memotion2_results"] = memotion2_results

    summary_path = os.path.join(results_dir, "analysis_summary.json")
    os.makedirs(results_dir, exist_ok=True)
    save_json(summary, summary_path)
    print(f"\n✅ Analysis saved to: {summary_path}")


if __name__ == "__main__":
    main()
