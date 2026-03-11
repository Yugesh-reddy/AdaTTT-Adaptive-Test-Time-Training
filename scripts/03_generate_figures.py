#!/usr/bin/env python3
"""
Generate 7 publication-quality figures. No GPU needed.

Usage:
    python scripts/03_generate_figures.py --config config/config.yaml
    python scripts/03_generate_figures.py --results-dir results/ --output-dir figures/

Figures:
    1. Pareto Frontier (accuracy vs GFLOPs) — THE MAIN FIGURE
    2. TTT Steps vs Accuracy (K vs accuracy per objective)
    3. Gate Routing Analysis (skip/adapt stacked bars per threshold)
    4. Per-Question-Type Breakdown (grouped bars)
    5. TTT Stabilization Ablation (bar chart)
    6. Confidence Distribution (histogram for correct vs incorrect)
    7. Memotion2 Cross-Task Generalization (VQA-v2 vs Memotion2 bar chart)
"""

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttt.utils import load_config, load_json
from ttt.metrics import pareto_frontier


# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig1_pareto_frontier(results: List[Dict], output_dir: str):
    """Figure 1: Pareto Frontier — accuracy vs compute cost."""
    if not results:
        print("  [skip] No results for Pareto frontier")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by K value
    colors = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#F44336", 5: "#9C27B0"}
    k_pattern = re.compile(r"K=(\d+)")

    for r in results:
        k_match = k_pattern.search(r["config"])
        k_val = int(k_match.group(1)) if k_match else 0
        color = colors.get(k_val, "#607D8B")
        ax.scatter(r["avg_flops"], r["accuracy"] * 100, c=color, s=80, zorder=5, alpha=0.8)
        ax.annotate(
            r["config"], (r["avg_flops"], r["accuracy"] * 100),
            textcoords="offset points", xytext=(8, 4), fontsize=7,
        )

    # Draw Pareto frontier line
    pareto = pareto_frontier(results)
    if len(pareto) > 1:
        ax.plot(
            [p["avg_flops"] for p in pareto],
            [p["accuracy"] * 100 for p in pareto],
            "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier",
        )

    ax.set_xlabel("Average GFLOPs per Sample")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Accuracy vs Compute Cost — Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "fig1_pareto_frontier.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig2_k_vs_accuracy(results: List[Dict], output_dir: str):
    """Figure 2: TTT Steps vs Accuracy (per objective)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    objectives = {}
    k_pattern = re.compile(r"K=(\d+)")
    for r in results:
        config = r["config"]
        k_match = k_pattern.search(config)
        if k_match:
            k_val = int(k_match.group(1))
            obj = "masked_patch" if "masked_patch" in config else (
                "rotation" if "rotation" in config else "baseline"
            )
            if obj not in objectives:
                objectives[obj] = []
            objectives[obj].append((k_val, r["accuracy"] * 100))

    markers = {"masked_patch": "o-", "rotation": "s--", "baseline": "D:"}
    colors = {"masked_patch": "#2196F3", "rotation": "#F44336", "baseline": "#4CAF50"}

    for obj, points in sorted(objectives.items()):
        points.sort(key=lambda x: x[0])
        ks = [p[0] for p in points]
        accs = [p[1] for p in points]
        ax.plot(ks, accs, markers.get(obj, "o-"), color=colors.get(obj, "#607D8B"),
                label=obj, linewidth=2, markersize=8)

    ax.set_xlabel("K (TTT Gradient Steps)")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("Effect of TTT Steps on Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 1, 2, 3, 5])

    path = os.path.join(output_dir, "fig2_k_vs_accuracy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig3_gate_routing(results: List[Dict], output_dir: str):
    """Figure 3: Gate Routing Analysis (skip/adapt bars per threshold)."""
    adaptive = [r for r in results if "Adaptive" in r.get("config", "")]
    if not adaptive:
        print("  [skip] No adaptive results for routing analysis")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = []
    skip_rates = []
    accuracies = []
    t_pattern = re.compile(r"τ=([\d.]+)")

    for r in sorted(adaptive, key=lambda x: x.get("config", "")):
        t_match = t_pattern.search(r["config"])
        if t_match:
            t = float(t_match.group(1))
            thresholds.append(t)
            skip_rates.append(r.get("skip_rate", 0) * 100)
            accuracies.append(r["accuracy"] * 100)

    x = np.arange(len(thresholds))
    width = 0.35

    bars1 = ax.bar(x - width / 2, skip_rates, width, label="Skip Rate (%)",
                   color="#2196F3", alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, accuracies, width, label="Accuracy (%)",
                    color="#4CAF50", alpha=0.8)

    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("Skip Rate (%)", color="#2196F3")
    ax2.set_ylabel("Accuracy (%)", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_title("Gate Routing Analysis")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    path = os.path.join(output_dir, "fig3_gate_routing.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig4_per_type_breakdown(results: List[Dict], output_dir: str):
    """Figure 4: Per-Question-Type Accuracy Breakdown."""
    configs_with_types = [r for r in results if "type_accuracy" in r]
    if not configs_with_types:
        print("  [skip] No per-type data available")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Select up to 3 configs for comparison
    selected = configs_with_types[:3]
    qtypes = list(selected[0]["type_accuracy"].keys())
    x = np.arange(len(qtypes))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, r in enumerate(selected):
        accs = [r["type_accuracy"].get(qt, 0) * 100 for qt in qtypes]
        ax.bar(x + i * width, accs, width, label=r["config"], color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Question Type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type")
    ax.set_xticks(x + width * (len(selected) - 1) / 2)
    ax.set_xticklabels(qtypes)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    path = os.path.join(output_dir, "fig4_per_type_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig5_ablation(results_dir: str, output_dir: str):
    """Figure 5: TTT Stabilization Ablation."""
    ablation_dir = os.path.join(results_dir, "ablation")
    if not os.path.exists(ablation_dir):
        print("  [skip] No ablation results found")
        return

    modes = ["ttt_only", "ttt_consistency", "ttt_mixup", "ttt_both"]
    labels = ["TTT Only", "TTT + Consistency", "TTT + Mixup", "TTT + Both"]
    accuracies = []
    found_modes = []

    for mode, label in zip(modes, labels):
        files = glob.glob(os.path.join(ablation_dir, f"{mode}_*.json"))
        if files:
            data = load_json(files[0])
            preds = [d["prediction"] for d in data]
            gts = [d["ground_truth"] for d in data]
            acc = sum(p == g for p, g in zip(preds, gts)) / len(preds) * 100
            accuracies.append(acc)
            found_modes.append(label)

    if not found_modes:
        print("  [skip] No ablation data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#F44336", "#2196F3", "#FF9800", "#4CAF50"]
    bars = ax.bar(range(len(found_modes)), accuracies, color=colors[:len(found_modes)], alpha=0.85)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(range(len(found_modes)))
    ax.set_xticklabels(found_modes, rotation=15, ha="right")
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("TTT Stabilization Ablation")
    ax.grid(True, alpha=0.2, axis="y")

    path = os.path.join(output_dir, "fig5_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig6_confidence_distribution(results_dir: str, output_dir: str):
    """Figure 6: Gate confidence distribution for correct vs incorrect predictions."""
    # Look for adaptive results that include confidence values
    adaptive_files = glob.glob(os.path.join(results_dir, "adaptive_*.json"))
    if not adaptive_files:
        print("  [skip] No adaptive results with confidence data")
        return

    data = load_json(adaptive_files[0])
    correct_conf = [d["confidence"] for d in data if d.get("prediction") == d.get("ground_truth")]
    incorrect_conf = [d["confidence"] for d in data if d.get("prediction") != d.get("ground_truth")]

    if not correct_conf and not incorrect_conf:
        print("  [skip] No confidence data in results")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 30)
    if correct_conf:
        ax.hist(correct_conf, bins=bins, alpha=0.6, color="#4CAF50", label="Correct (base)", density=True)
    if incorrect_conf:
        ax.hist(incorrect_conf, bins=bins, alpha=0.6, color="#F44336", label="Incorrect (base)", density=True)

    ax.set_xlabel("Gate Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "fig6_confidence_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def fig7_cross_task_generalization(results_dir: str, vqa_results: List[Dict], output_dir: str):
    """Figure 7: Cross-task generalization — VQA-v2 vs Memotion2."""
    memotion2_dir = os.path.join(results_dir, "memotion2")
    if not os.path.exists(memotion2_dir):
        print("  [skip] No Memotion2 results found")
        return

    memo_files = glob.glob(os.path.join(memotion2_dir, "*.json"))
    if not memo_files:
        print("  [skip] No Memotion2 result files")
        return

    # Compute Memotion2 accuracies
    memo_accs = {"base": None, "ttt": None, "adaptive": None}
    for mf in memo_files:
        basename = os.path.basename(mf)
        data = load_json(mf)
        preds = [d["prediction"] for d in data]
        gts = [d["ground_truth"] for d in data]
        acc = sum(p == g for p, g in zip(preds, gts)) / len(preds) * 100 if preds else 0

        if "k0" in basename or "baseline" in basename:
            memo_accs["base"] = acc
        elif "adaptive" in basename:
            memo_accs["adaptive"] = acc
        else:
            memo_accs["ttt"] = acc

    # Extract VQA accuracies from analysis summary
    vqa_accs = {"base": None, "ttt": None, "adaptive": None}
    for r in vqa_results:
        config = r.get("config", "")
        acc = r["accuracy"] * 100
        if "K=0" in config:
            vqa_accs["base"] = acc
        elif "Adaptive" in config and vqa_accs["adaptive"] is None:
            vqa_accs["adaptive"] = acc
        elif vqa_accs["ttt"] is None:
            vqa_accs["ttt"] = acc

    # Build figure
    fig, ax = plt.subplots(figsize=(9, 6))

    categories = []
    vqa_vals = []
    memo_vals = []
    labels = [("base", "Base (No TTT)"), ("ttt", "TTT"), ("adaptive", "Adaptive TTT")]

    for key, label in labels:
        if vqa_accs[key] is not None or memo_accs[key] is not None:
            categories.append(label)
            vqa_vals.append(vqa_accs[key] or 0)
            memo_vals.append(memo_accs[key] or 0)

    if not categories:
        print("  [skip] Insufficient data for cross-task comparison")
        plt.close(fig)
        return

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, vqa_vals, width, label="VQA-v2",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, memo_vals, width, label="Memotion2",
                   color="#FF9800", alpha=0.85)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-Task Generalization: VQA-v2 vs Memotion2")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    path = os.path.join(output_dir, "fig7_cross_task_generalization.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = args.results_dir or config.get("results_dir", "results/")
    output_dir = args.output_dir or config.get("figures_dir", "figures/")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating Publication Figures")
    print("=" * 60)

    # Load analysis summary
    summary_path = os.path.join(results_dir, "analysis_summary.json")
    results = []
    if os.path.exists(summary_path):
        summary = load_json(summary_path)
        results = summary.get("results", [])

    print("\n📈 Figure 1: Pareto Frontier")
    fig1_pareto_frontier(results, output_dir)

    print("\n📈 Figure 2: K vs Accuracy")
    fig2_k_vs_accuracy(results, output_dir)

    print("\n📈 Figure 3: Gate Routing Analysis")
    fig3_gate_routing(results, output_dir)

    print("\n📈 Figure 4: Per-Question-Type Breakdown")
    fig4_per_type_breakdown(results, output_dir)

    print("\n📈 Figure 5: TTT Stabilization Ablation")
    fig5_ablation(results_dir, output_dir)

    print("\n📈 Figure 6: Confidence Distribution")
    fig6_confidence_distribution(results_dir, output_dir)

    print("\n📈 Figure 7: Cross-Task Generalization (VQA-v2 vs Memotion2)")
    fig7_cross_task_generalization(results_dir, results, output_dir)

    print(f"\n✅ All figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
