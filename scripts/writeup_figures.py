#!/usr/bin/env python3
"""Phase 1 ceiling + Phase 2 visual-TTA figures. CPU only.

Reads results/writeup/numbers.json (written by scripts/writeup_numbers.py);
every position and caption comes from it.
"""

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
PURPLE = "#CC78BC"
GRAY = "#949494"

SHORT = {
    "no_adapt": ("skip", GREEN),
    "tent": ("TENT-style", GRAY),
    "eata": ("EATA-style", PURPLE),
    "memo": ("MEMO", ORANGE),
    "memo_sar": ("MEMO+SAR", RED),
    "gated_memo_sar": ("gated (in-sample τ)", BLUE),
}
COND_LABEL = {"identity": "ID (eval 8k)", "blur_s3": "blur s3", "noise_s5": "noise s5"}


def _save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.svg"))
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)


def _label_bars(ax, bars, fmt, pad):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + pad, fmt.format(h),
                ha="center", va="bottom")


def fig_phase1_ceiling(n: dict) -> None:
    soft, exact = n["phase1"]["soft"], n["phase1"]["exact"]
    verdict = n["phase1"]["verdict"]
    vals = [soft["v1"], soft["control_vitbert"], soft["clip"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(["v1 (ViT+BERT)", "v2 control", "v2 CLIP"], vals,
                  color=[GRAY, ORANGE, BLUE], width=0.62)
    _label_bars(ax, bars, "{:.2f}", 0.4)
    ax.set_ylabel("Official VQA soft (%)")
    ax.set_title("Phase 1 ceiling on VQA-v2 val")
    ax.set_ylim(50, 72)
    ax.axhline(soft["v1"], color=GRAY, linestyle=":", linewidth=1)
    fig.text(0.02, -0.02,
             f"CLIP − control = {soft['encoder_delta_pp']:+.2f} pp soft / "
             f"{exact['encoder_delta_pp']:+.2f} pp exact. Verdicts: CLIP {verdict['clip']}, "
             f"control {verdict['control_vitbert']}.",
             fontsize=8, color="#444")
    _save(fig, "phase1_ceiling")


def fig_shift_drop(n: dict) -> None:
    e = n["eval_8k"]
    conds = ["identity", "blur_s3", "noise_s5"]
    vals = [e[c]["skip_soft"] for c in conds]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar([COND_LABEL[c] for c in conds], vals, color=[BLUE, ORANGE, RED], width=0.62)
    _label_bars(ax, bars, "{:.2f}", 0.35)
    for i, c in enumerate(conds[1:], start=1):
        ax.text(i, vals[i] - 1.35, f"{e[c]['drop_vs_id_pp']:+.2f} pp vs ID",
                ha="center", fontsize=9, color="#555")
    ax.set_ylabel("Official VQA soft (%)")
    ax.set_title("Skip accuracy under visual shift")
    ax.set_ylim(55, 72)
    _save(fig, "shift_drop")


def fig_pareto(n: dict) -> None:
    e = n["eval_8k"]
    ladder = n["flops_ladder_gflops"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
    xmax = max(r["avg_gflops"] for c in ("blur_s3", "noise_s5") for r in e[c]["methods"].values())
    for ax, cond in zip(axes, ("blur_s3", "noise_s5")):
        row = e[cond]
        ys = []
        for m, r in row["methods"].items():
            label, color = SHORT[m]
            if m == "gated_memo_sar":
                # Hollow ring: on blur it sits exactly on skip, which must stay visible.
                ax.scatter([r["avg_gflops"]], [r["soft"]], facecolors="none", edgecolors=color,
                           linewidths=2, s=130, zorder=4, label=label)
            else:
                ax.scatter([r["avg_gflops"]], [r["soft"]], color=color, s=48, zorder=3, label=label)
            ys.append(r["soft"])
        ax.axhline(row["oracle_soft"], color=BLUE, linestyle="--", linewidth=1,
                   label="per-sample oracle (needs MEMO on every sample)")
        ax.text(3, row["oracle_soft"] + 0.01, f"oracle {row['oracle_soft']:.2f}",
                fontsize=8, color=BLUE, va="bottom")
        ys.append(row["oracle_soft"])
        ax.set_title(COND_LABEL[cond])
        ax.set_ylim(min(ys) - 0.12, max(ys) + 0.12)
        ax.set_xlim(0, xmax + 15)
        ax.set_xlabel("avg GFLOPs per sample (sample_flops)")
    axes[0].set_ylabel("Official VQA soft (%)")
    fig.suptitle("Accuracy vs deployment FLOPs (eval 8k)", y=1.02)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.02, -0.16,
             f"Ladder at K=1: base {ladder['base']} / TENT-style {ladder['tent_k1']} / "
             f"MEMO2 {ladder['memo2_k1']} / MEMO4 {ladder['memo4_k1']} GFLOPs. "
             "Y-axis zoomed: every dense delta is under 0.05 pp.",
             fontsize=8, color="#444")
    _save(fig, "pareto_flops")


def fig_adapter_movement(n: dict) -> None:
    """MEMO's answer changes do not track the shift: the adapter barely moves."""
    e = n["eval_8k"]
    a = n["adapter"]
    conds = ["identity", "blur_s3", "noise_s5"]
    flips = [e[c]["methods"]["memo"]["pred_flips_vs_skip"] for c in conds]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar([COND_LABEL[c] for c in conds], flips, color=[BLUE, ORANGE, RED], width=0.62)
    _label_bars(ax, bars, "{:.0f}", 0.8)
    for i, c in enumerate(conds):
        drop = e[c]["drop_vs_id_pp"]
        ax.text(i, 3, "clean" if drop is None else f"skip {drop:+.2f} pp",
                ha="center", fontsize=9, color="white")
    ax.set_ylabel(f"MEMO answers changed (of {e['identity']['n']})")
    ax.set_title("MEMO moves ~0.5% of answers, with or without a shift")
    ax.set_ylim(0, max(flips) * 1.3)
    fig.text(0.02, -0.04,
             f"{a['optimizer']}; lr {a['lr']:g}, K={a['k_steps']}, {a['adapted_values']:,} "
             f"{a['params']} — each moves ~lr per step.",
             fontsize=8, color="#444")
    _save(fig, "adapter_movement")


def fig_oracle_vs_dense(n: dict) -> None:
    e = n["eval_8k"]
    conds = ["identity", "blur_s3", "noise_s5"]
    dense = [e[c]["methods"]["memo"]["delta_soft_pp"] for c in conds]
    oracle = [e[c]["oracle_delta_pp"] for c in conds]
    x = list(range(len(conds)))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar([i - w / 2 for i in x], dense, width=w, color=ORANGE, label="dense MEMO − skip")
    ax.bar([i + w / 2 for i in x], oracle, width=w, color=BLUE, label="oracle − skip")
    ax.axhline(0.0, color=GRAY, linewidth=1)
    ax.axhline(0.5, color=RED, linestyle=":", linewidth=1, label="0.5 pp kill line")
    ax.set_xticks(x, [COND_LABEL[c] for c in conds])
    ax.set_ylabel("Soft delta (pp)")
    ax.set_title("Dense MEMO vs per-sample oracle")
    ax.legend(fontsize=8)
    _save(fig, "oracle_vs_dense")


def fig_mcnemar(n: dict) -> None:
    b = n["eval_8k"]["blur_s3"]
    m = b["methods"]["memo"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].bar(["MEMO better", "MEMO worse"], [m["soft_better"], m["soft_worse"]],
                color=[GREEN, RED], width=0.55)
    axes[0].set_title(f"Soft changes ({m['pred_flips_vs_skip']} flips / {b['n']})")
    axes[0].set_ylabel("Samples")
    axes[1].bar(["skip right\nMEMO wrong", "skip wrong\nMEMO right"],
                [m["mcnemar_skip_right_method_wrong"], m["mcnemar_skip_wrong_method_right"]],
                color=[ORANGE, BLUE], width=0.55)
    axes[1].set_title(f"McNemar {m['mcnemar_skip_right_method_wrong']} vs "
                      f"{m['mcnemar_skip_wrong_method_right']} (p={m['mcnemar_exact_p']:.2f})")
    top = max(m["soft_better"], m["soft_worse"], m["mcnemar_skip_right_method_wrong"],
              m["mcnemar_skip_wrong_method_right"])
    for ax in axes:
        _label_bars(ax, ax.patches, "{:.0f}", 0.3)
        ax.set_ylim(0, top * 1.3)
    fig.suptitle("Blur s3: MEMO vs skip", y=1.03)
    _save(fig, "blur_mcnemar")


def main() -> int:
    with open(NUMBERS) as fh:
        n = json.load(fh)
    fig_phase1_ceiling(n)
    fig_shift_drop(n)
    fig_pareto(n)
    fig_adapter_movement(n)
    fig_oracle_vs_dense(n)
    fig_mcnemar(n)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
