#!/usr/bin/env python3
"""Phase 1 ceiling + Phase 2 visual-TTA figures. CPU only. Reads numbers.json."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NUMBERS = os.path.join(ROOT, "results", "writeup", "numbers.json")
OUT = os.path.join(ROOT, "results", "writeup", "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
RED = "#D55E00"
GRAY = "#949494"


def _save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.svg"))
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)


def fig_phase1_ceiling(n: dict) -> None:
    soft = n["phase1"]["soft"]
    labels = ["v1 (ViT+BERT)", "v2 control", "v2 CLIP"]
    vals = [soft["v1"], soft["control_vitbert"], soft["clip"]]
    colors = [GRAY, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylabel("Official VQA soft (%)")
    ax.set_title("Phase 1 ceiling on VQA-v2 val")
    ax.set_ylim(50, 72)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.2f}",
                ha="center", va="bottom")
    ax.axhline(soft["v1"], color=GRAY, linestyle=":", linewidth=1)
    fig.text(
        0.02, -0.02,
        "CLIP − control = +7.57 pp (encoder). Both CONTINUE. Exact: 56.99 / 49.56 / 49.56.",
        fontsize=8, color="#444",
    )
    _save(fig, "phase1_ceiling")


def fig_shift_drop(n: dict) -> None:
    e = n["eval_8k"]
    labels = ["ID (eval 8k)", "blur s3", "noise s5"]
    vals = [e["identity"]["soft"], e["blur_s3"]["soft"], e["noise_s5"]["soft"]]
    colors = [BLUE, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylabel("Official VQA soft (%)")
    ax.set_title("Skip accuracy under visual shift")
    ax.set_ylim(55, 72)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}",
                ha="center", va="bottom")
    ax.text(1, vals[1] - 1.35, "−1.99 pp vs ID", ha="center", fontsize=9, color="#555")
    ax.text(2, vals[2] - 1.35, "−6.40 pp vs ID", ha="center", fontsize=9, color="#555")
    _save(fig, "shift_drop")


def fig_pareto(n: dict) -> None:
    e = n["eval_8k"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    series = [
        (
            "blur s3",
            e["blur_s3"],
            [(24.8, e["blur_s3"]["soft"], "skip / gated", BLUE),
             (43.4, e["blur_s3"]["tent_soft"], "TENT", GRAY),
             (138.6, e["blur_s3"]["memo_soft"], "MEMO", ORANGE)],
            (65.85, 66.35),
        ),
        (
            "noise s5",
            e["noise_s5"],
            [(24.8, e["noise_s5"]["soft"], "skip", GREEN),
             (43.4, e["noise_s5"]["tent_soft"], "TENT", GRAY),
             (44.47, e["noise_s5"]["gated_soft"], "gated 17%", BLUE),
             (138.6, e["noise_s5"]["memo_soft"], "MEMO", RED)],
            (61.4, 61.9),
        ),
    ]
    for ax, (title, row, pts, ylim) in zip(axes, series):
        for x, y, label, c in pts:
            ax.scatter([x], [y], color=c, s=48, zorder=3, label=label)
        ax.axhline(row["oracle_soft"], color=BLUE, linestyle="--", linewidth=1,
                   label=f"oracle {row['oracle_soft']:.2f}")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xlim(10, 155)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel("GFLOPs (sample_flops)")
    axes[0].set_ylabel("Official VQA soft (%)")
    fig.suptitle("Pareto: accuracy vs deployment FLOPs", y=1.02)
    fig.text(
        0.02, -0.06,
        "Ladder 24.8 / 43.4 / 91.0 / 138.6. Y-axis zoomed; all dense deltas are << 0.5 pp.",
        fontsize=8, color="#444",
    )
    _save(fig, "pareto_flops")


def fig_oracle_vs_dense(n: dict) -> None:
    e = n["eval_8k"]
    labels = ["blur s3", "noise s5"]
    dense = [e["blur_s3"]["memo_delta_pp"], e["noise_s5"]["memo_delta_pp"]]
    oracle = [e["blur_s3"]["oracle_delta_pp"], e["noise_s5"]["oracle_delta_pp"]]
    x = [0, 1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    w = 0.32
    ax.bar([i - w / 2 for i in x], dense, width=w, color=ORANGE, label="dense MEMO − skip")
    ax.bar([i + w / 2 for i in x], oracle, width=w, color=BLUE, label="oracle − skip")
    ax.axhline(0.0, color=GRAY, linewidth=1)
    ax.axhline(0.5, color=RED, linestyle=":", linewidth=1, label="0.5 pp kill line")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Soft delta (pp)")
    ax.set_title("Dense MEMO vs per-sample oracle")
    ax.legend(fontsize=8)
    _save(fig, "oracle_vs_dense")


def fig_mcnemar(n: dict) -> None:
    b = n["eval_8k"]["blur_s3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].bar(["MEMO better", "MEMO worse"],
                [b["memo_better"], b["memo_worse"]],
                color=[GREEN, RED], width=0.55)
    axes[0].set_title("Soft flips (40 / 8000)")
    axes[0].set_ylabel("Samples")
    axes[1].bar(["skip right\nMEMO wrong", "skip wrong\nMEMO right"],
                [b["mcnemar_base_correct_memo_wrong"],
                 b["mcnemar_base_wrong_memo_correct"]],
                color=[ORANGE, BLUE], width=0.55)
    axes[1].set_title("McNemar 17 vs 15")
    for ax in axes:
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{int(bar.get_height())}", ha="center", va="bottom")
        ax.set_ylim(0, 24)
    fig.suptitle("Blur s3: MEMO vs skip", y=1.03)
    _save(fig, "blur_mcnemar")


def main() -> int:
    with open(NUMBERS) as fh:
        n = json.load(fh)
    fig_phase1_ceiling(n)
    fig_shift_drop(n)
    fig_pareto(n)
    fig_oracle_vs_dense(n)
    fig_mcnemar(n)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
