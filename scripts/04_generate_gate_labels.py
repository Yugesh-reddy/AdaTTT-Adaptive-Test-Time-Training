#!/usr/bin/env python3
"""
Generate training labels for the confidence gate.

Compares base model predictions vs TTT predictions to determine
which samples benefit from TTT adaptation.

This runs LOCALLY (no GPU needed — reads saved prediction files).

Usage:
    python scripts/04_generate_gate_labels.py --config config/config.yaml --split train

Prerequisites:
    - Base predictions for chosen split, e.g.:
      results/ttt_predictions/train/k0_baseline.json
    - TTT predictions for chosen split, e.g.:
      results/ttt_predictions/train/k1_masked_patch.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.utils import load_config, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="Generate gate training labels")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split for gate labels. Default: train.",
    )
    parser.add_argument(
        "--base-predictions",
        type=str,
        default=None,
        help="Path to base predictions JSON",
    )
    parser.add_argument(
        "--ttt-predictions",
        type=str,
        default=None,
        help="Path to TTT predictions JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for gate labels",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.get("results_dir", "results/")
    data_dir = config.get("data_dir", "data/")

    if args.base_predictions:
        base_path = args.base_predictions
    else:
        base_candidates = [
            os.path.join(results_dir, f"base_predictions_{args.split}.json"),
            os.path.join(results_dir, "ttt_predictions", args.split, "k0_baseline.json"),
        ]
        if args.split == "val":
            # Backward-compatibility with older base training outputs.
            base_candidates.append(os.path.join(results_dir, "base_predictions.json"))
        base_path = next((p for p in base_candidates if os.path.exists(p)), base_candidates[0])

    if args.ttt_predictions:
        ttt_path = args.ttt_predictions
    else:
        ttt_candidates = [
            os.path.join(results_dir, "ttt_predictions", args.split, "k1_masked_patch.json"),
        ]
        if args.split == "val":
            # Backward-compatibility with older TTT sweep outputs.
            ttt_candidates.append(os.path.join(results_dir, "ttt_predictions", "k1_masked_patch.json"))
        ttt_path = next((p for p in ttt_candidates if os.path.exists(p)), ttt_candidates[0])

    output_path = args.output or os.path.join(data_dir, f"gate_labels_{args.split}.json")

    print("=" * 60)
    print("Generating Gate Training Labels")
    print("=" * 60)
    print(f"  Base predictions: {base_path}")
    print(f"  TTT predictions:  {ttt_path}")
    print(f"  Output:           {output_path}")
    print()

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base predictions not found: {base_path}")
    if not os.path.exists(ttt_path):
        raise FileNotFoundError(f"TTT predictions not found: {ttt_path}")

    # Load predictions
    base_preds = load_json(base_path)
    ttt_preds = load_json(ttt_path)

    # Build lookup by sample_id
    base_map = {p["sample_id"]: p for p in base_preds}
    ttt_map = {p["sample_id"]: p for p in ttt_preds}

    # Generate labels
    gate_labels = []
    stats = {"base_correct": 0, "ttt_helps": 0, "ttt_hurts": 0, "neither": 0, "total": 0}

    for sample_id in base_map:
        if sample_id not in ttt_map:
            continue

        base_pred = base_map[sample_id]["prediction"]
        ttt_pred = ttt_map[sample_id]["prediction"]
        gt = base_map[sample_id]["ground_truth"]

        base_correct = (base_pred == gt)
        ttt_correct = (ttt_pred == gt)
        ttt_helps = (not base_correct) and ttt_correct
        ttt_hurts = base_correct and (not ttt_correct)

        # Gate label logic:
        # 1.0 (SKIP TTT) if base is already correct
        # 0.0 (APPLY TTT) if TTT helps
        # 1.0 if TTT hurts (protect base prediction)
        # 0.5 if neither helps (uncertain)
        if ttt_hurts:
            gate_label = 1.0
            stats["ttt_hurts"] += 1
        elif base_correct:
            gate_label = 1.0
            stats["base_correct"] += 1
        elif ttt_helps:
            gate_label = 0.0
            stats["ttt_helps"] += 1
        else:
            gate_label = 0.5
            stats["neither"] += 1

        gate_labels.append({
            "sample_id": sample_id,
            "split": args.split,
            "gate_label": gate_label,
            "base_correct": base_correct,
            "ttt_helps": ttt_helps,
            "ttt_hurts": ttt_hurts,
        })
        stats["total"] += 1

    # Save
    save_json(gate_labels, output_path)

    # Print statistics
    print(f"\nResults ({stats['total']} samples):")
    print(f"  Base correct (label=1.0):  {stats['base_correct']:6d}  ({100*stats['base_correct']/max(stats['total'],1):.1f}%)")
    print(f"  TTT helps (label=0.0):     {stats['ttt_helps']:6d}  ({100*stats['ttt_helps']/max(stats['total'],1):.1f}%)")
    print(f"  TTT hurts (label=1.0):     {stats['ttt_hurts']:6d}  ({100*stats['ttt_hurts']/max(stats['total'],1):.1f}%)")
    print(f"  Neither (label=0.5):       {stats['neither']:6d}  ({100*stats['neither']/max(stats['total'],1):.1f}%)")
    print(f"\n✅ Gate labels saved to: {output_path}")


if __name__ == "__main__":
    main()
