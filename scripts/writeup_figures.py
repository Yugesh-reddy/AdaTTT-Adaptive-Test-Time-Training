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
    # Fixed salt for SVG element ids (random per run otherwise), so regenerating
    # an unchanged figure changes no bytes; see also Date=None in _save.
    "svg.hashsalt": "adattt-writeup",
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
    # No timestamp in the SVG, so regenerating unchanged figures changes no bytes.
    fig.savefig(os.path.join(OUT, f"{name}.svg"), metadata={"Date": None})
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


def fig_session_d_sweep(n: dict) -> None:
    """Session D, gate-train only: gain and ceiling vs step size, and how much MEMO moves."""
    d = n["session_d"]
    rows = d["sweep_gate_train"]
    lrs = [r["lr"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    ax.plot(lrs, [r["gain_pp"] for r in rows], "o-", color=ORANGE, label="dense MEMO − skip")
    ax.plot(lrs, [r["oracle_gain_pp"] for r in rows], "s--", color=BLUE, label="oracle − skip")
    ax.axhline(0.0, color=GRAY, linewidth=1)
    ax.axhline(d["min_gain_pp"], color=RED, linestyle=":", linewidth=1,
               label=f"proceed threshold {d['min_gain_pp']} pp")
    ax.axvline(d["chosen_lr"], color=GREEN, linestyle=":", linewidth=1.2,
               label=f"chosen lr {d['chosen_lr']:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Adam step size (lr), K=1")
    ax.set_ylabel("Soft delta vs skip (pp)")
    ax.set_title("Gate-train 8k, noise s5")
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.plot(lrs, [r["pred_flips"] for r in rows], "o-", color=ORANGE, label="gate-train 8k")
    c_flips = n["eval_8k"]["noise_s5"]["methods"]["memo"]["pred_flips_vs_skip"]
    ax.scatter([n["adapter"]["lr"]], [c_flips], facecolors="none", edgecolors=GRAY, s=70,
               linewidths=1.5, zorder=3, label=f"lr {n['adapter']['lr']:g}, Session C (eval 8k)")
    ax.set_xscale("log")
    ax.set_xlabel("Adam step size (lr), K=1")
    ax.set_ylabel("MEMO answers changed (of 8000)")
    ax.set_title("How far MEMO moves the model")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Session D: choosing the step size on gate-train only", y=1.03)
    _save(fig, "session_d_sweep")


def fig_session_d_eval(n: dict) -> None:
    """Session D, eval 8k scored once: gains with CIs, and who MEMO helps by gate score."""
    d = n["session_d"]
    e = d["eval_8k"]["methods"]
    c = n["eval_8k"]["noise_s5"]["methods"]["memo"]
    lr = d["chosen_lr"]
    rows = [
        (f"MEMO\nlr {n['adapter']['lr']:g}", c, GRAY),
        (f"MEMO\nlr {lr:g}", e["memo"], ORANGE),
        (f"MEMO+SAR\nlr {lr:g}", e["memo_sar"], RED),
        (f"gated lr {lr:g}\nheld-out τ", e["gated_memo_sar"], BLUE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    for i, (label, r, color) in enumerate(rows):
        lo, hi = r["delta_soft_ci95_pp"]
        mid = r["delta_soft_pp"]
        ax.errorbar([i], [mid], yerr=[[mid - lo], [hi - mid]], fmt="o", color=color,
                    capsize=5, markersize=7)
    ax.axhline(0.0, color=GRAY, linewidth=1)
    ax.axhline(d["eval_8k"]["oracle_delta_pp"], color=BLUE, linestyle="--", linewidth=1)
    ax.text(len(rows) - 0.5, d["eval_8k"]["oracle_delta_pp"] + 0.05,
            f"oracle at lr {lr:g}: {d['eval_8k']['oracle_delta_pp']:+.2f}", ha="right",
            fontsize=8, color=BLUE)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("Soft delta vs skip, 95% CI (pp)")
    ax.set_title(f"Eval 8k, noise s5 (skip {d['eval_8k']['skip_soft']:.2f}, "
                 f"{d['eval_8k']['drop_vs_id_pp']:+.2f} vs ID)", fontsize=10)
    ax = axes[1]
    q = d["benefit_signal"]["by_gate_score_quartile_post_hoc"]
    x = list(range(len(q)))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [r["helped"] for r in q], width=w, color=GREEN, label="MEMO helped")
    ax.bar([i + w / 2 for i in x], [r["hurt"] for r in q], width=w, color=RED, label="MEMO hurt")
    for i, r in enumerate(q):
        ax.text(i, max(r["helped"], r["hurt"]) + 2, f"{r['net_pp']:+.2f} pp", ha="center", fontsize=8)
    ax.set_xticks(x, [f"Q{r['quartile']}" for r in q])
    ax.set_xlabel("Gate-score quartile (Q1 = most confident)")
    ax.set_ylabel("Samples")
    ax.set_title(f"Post hoc: who MEMO helps (AUROC "
                 f"{d['benefit_signal']['gate_score_auroc_helped_vs_hurt']:.3f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Session D: MEMO at lr {lr:g} moves {e['memo']['pred_flips_vs_skip']} answers "
                 "but recovers nothing measurable", y=1.03)
    _save(fig, "session_d_eval")


def fig_session_e_gate(n: dict) -> None:
    """Session E: gates cross-validated on gate-train, the chosen one scored once."""
    e = n["session_e"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    names = list(e["cv"])
    labels = {"score": "gate score", "free": "free", "one_view": "+1 view",
              "four_views": "+4 views", "post": "after MEMO"}
    bars = ax.bar([labels[k] for k in names], [e["cv"][k]["gain_pp"] for k in names],
                  color=[BLUE if k == e["chosen"] else GRAY for k in names], width=0.6)
    for bar, k in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{e['cv'][k]['avg_gflops']:.0f}G", ha="center", va="bottom", fontsize=8)
    ax.axhline(e["gate_train"]["memo_gain_pp"], color=ORANGE, linestyle="--", linewidth=1,
               label=f"dense MEMO {e['gate_train']['memo_gain_pp']:+.2f}")
    ax.axhline(0.2, color=RED, linestyle=":", linewidth=1, label="stop threshold 0.2")
    ax.set_ylabel("Cross-validated gain vs skip (pp)")
    ax.set_title(f"Gate-train 8k, 5-fold CV (chosen: {labels[e['chosen']]})", fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(0, max(e["cv"][k]["gain_pp"] for k in names) * 1.18)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax = axes[1]
    ev = e["eval_8k"]
    ax.scatter([n["flops_ladder_gflops"]["base"]], [0.0], color=GREEN, s=48, zorder=3, label="skip")
    for key, color, label in (("dense_memo", ORANGE, "dense MEMO"),
                              ("gated", BLUE, f"benefit gate ({labels[e['chosen']]})")):
        r = ev[key]
        lo, hi = r["delta_ci95_pp"]
        ax.errorbar([r["avg_gflops"]], [r["delta_pp"]],
                    yerr=[[r["delta_pp"] - lo], [hi - r["delta_pp"]]],
                    fmt="o", color=color, capsize=5, markersize=7, label=label)
    ax.axhline(ev["oracle_gain_pp"], color=BLUE, linestyle="--", linewidth=1)
    ax.text(n["flops_ladder_gflops"]["base"], ev["oracle_gain_pp"] + 0.04,
            f"oracle {ev['oracle_gain_pp']:+.2f}", fontsize=8, color=BLUE)
    ax.axhline(0.0, color=GRAY, linewidth=1)
    ax.set_xlim(0, 195)
    ax.set_xlabel("avg GFLOPs per sample (sample_flops)")
    ax.set_ylabel("Soft delta vs skip, 95% CI (pp)")
    ax.set_title("Eval 8k, scored once", fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle(f"Session E: a benefit gate at lr {e['lr']:g}, noise s5", y=1.03)
    _save(fig, "session_e_gate")


def main() -> int:
    with open(NUMBERS) as fh:
        n = json.load(fh)
    fig_phase1_ceiling(n)
    fig_shift_drop(n)
    fig_pareto(n)
    fig_adapter_movement(n)
    fig_oracle_vs_dense(n)
    fig_mcnemar(n)
    if n.get("session_d"):
        fig_session_d_sweep(n)
        fig_session_d_eval(n)
    if n.get("session_e"):
        fig_session_e_gate(n)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
