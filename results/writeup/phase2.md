# Phase 2: visual TTA shows no gain at the tested step size

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

## Kill

Stop the visual-corruption grid at this operating point. The open question, whether a larger update recovers the noise-s5 drop, needs one run with the step size and τ chosen on `gate_train_subset_8k` and the eval 8k scored once.
