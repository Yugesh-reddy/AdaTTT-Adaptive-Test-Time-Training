# Phase 2: visual TTA recovers no measurable accuracy

Eval: frozen `data/eval_subset_8k.json` (n = 8000). Episodic, fusion-LayerNorm-only adaptation (15,360 values; Adam, fresh per sample, lr 1e-4, K=1, restored after each sample). Methods: skip, TENT-style, EATA-style (no Fisher term), MEMO (4 AugMix views), MEMO + SAR's entropy filter, gated MEMO + SAR. The published TENT, EATA and SAR adapt online; these are single-step episodic variants.

τ was tuned on each condition's own eval-8k outcomes; `gate_train_subset_8k.json` was not used, so gated numbers are in-sample. FLOPs are `sample_flops` with MEMO's backward charged per view: 24.8 / 43.4 / 103.4 / 175.8 GFLOPs. Full tables: `REPORT.md`.

## Shift vs identity

| Condition | Skip soft | Skip exact | vs ID | Dense MEMO | Oracle | MEMO answer changes | Gate adapt |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity | 68.02 | 57.66 | — | 68.02 (0.00) | +0.10 | 39 | n/a |
| blur s3 | 66.03 | 55.76 | −1.99 | 66.00 (−0.03) | +0.14 | 40 | 0% (τ 0.83, in-sample) |
| noise s5 | 61.62 | 51.15 | **−6.40** | 61.58 (−0.04) | +0.10 | 45 | 17.3% (τ 0.35, in-sample; 24.6% pass τ) |

Identity skip 68.02 on the 8k slice is consistent with CLIP 67.50 on full val.

## What the runs show

- **The adapter barely moves the model.** Adam's first step moves each value by ~lr. At 1e-4 over the fusion LayerNorms, MEMO changes ~0.5% of answers whether or not the image is shifted. The oracle is bounded by that movement.
- **Nothing is significant.** No method differs from skip: exact McNemar p > 0.2 for every method, or 0.125 counting the in-sample gated run. Blur s3: 40 changes, 14 better / 19 worse, McNemar 17 vs 15 (p = 0.86).
- **Noise s5 has something to recover.** Skip falls 6.40 pp, and this operating point recovers none of it. That is a statement about lr 1e-4, K=1, fusion LayerNorms, not about visual TTA in general.
- **The gate's behaviour is not evidence.** τ saw the outcomes. On noise the gate score (high = uncertain) and SAR's filter (keeps confident samples) pull in opposite directions. τ was tuned against dense MEMO but applied to MEMO + SAR.

## Session D: a bigger step, chosen on gate-train (1.96 h)

Pre-registered sweep on `gate_train_subset_8k` (noise s5) over Adam step sizes 1e-3 / 3e-3 / 1e-2 / 3e-2. The largest gate-train MEMO gain wins, and the run stops if it is under 0.1 pp. 3e-3 won at +0.24 pp. τ was fit on gate-train, and the eval 8k was scored once:

| Method at 3e-3 | vs skip (95% CI) | Adapted | Answers changed | Helped / hurt |
|---|---:|---:|---:|---:|
| MEMO | +0.05 (−0.31, +0.45) | 100% | 584 | 174 / 166 |
| MEMO + SAR filter | +0.01 (−0.36, +0.41) | 93% | 474 | 155 / 155 |
| gated, held-out τ 0.226 | +0.07 (−0.25, +0.39) | 30% | 373 | 116 / 114 |
| oracle | +1.72 | — | — | — |

A bigger step makes MEMO move (584 answer changes vs 45 at 1e-4), but its effects cancel. The oracle shows +1.72 pp is there to recover if the helped samples could be picked in advance. The confidence-based gate score barely separates them (AUROC 0.585).

## Session E: a benefit gate (1.30 h)

Skip and MEMO at 3e-3 were run on both subsets with per-sample signals. Five gates were cross-validated on gate-train only; the cheapest near-best won (one AugMix view plus the free signals). It was frozen in git and scored once on the sealed eval 8k:

| | vs skip (95% CI) | Adapted | Avg GFLOPs | Helped / hurt |
|---|---:|---:|---:|---:|
| dense MEMO | +0.05 (−0.31, +0.45) | 100% | 175.8 | 174 / 166 |
| benefit gate | +0.18 (−0.11, +0.47) | 14% | 66.7 | 114 / 99 |
| oracle | +1.72 | — | — | — |

The gate beats dense MEMO at 38% of its compute, but not significantly beyond skip. No signal set, including four-view agreement and post-MEMO signals, predicts helped vs hurt above AUROC 0.57.

## Kill

Stop the visual-corruption grid. Neither the original step, a gate-train-chosen larger one, nor a pre-registered benefit gate recovers measurable accuracy on the noise-s5 drop. The limit is the benefit signal: these per-sample signals predict who MEMO helps only weakly.

## What works instead: abstention

The same model's confidence ranks correctness well. One threshold, fit on the noise-s5 gate-train split, keeps coverage near its 90% / 80% target on clean, blurred and noised images. It raises accuracy on the answered questions by +5.3 / +10.4 pp in every condition, at no extra compute; the unanswered 10–20% need a fallback. Details: `REPORT.md` §2.10.
