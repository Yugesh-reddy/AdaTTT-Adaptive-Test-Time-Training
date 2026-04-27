# Phase 2: visual TTA is a priced negative

Eval: frozen `data/eval_subset_8k.json` (n=8000). Fusion LN-only MEMO / TENT / EATA / MEMO-SAR. τ from `gate_train_subset_8k.json` on the same corruption. FLOPs from `AdaptiveRouter.sample_flops`: 24.8 / 43.4 / 91.0 / 138.6 GFLOPs. Not 58 / 93 / 128.

## Shift vs identity

| Condition | Skip soft | Skip exact | vs ID | Dense MEMO | Oracle | Gate adapt |
|---|---:|---:|---:|---:|---:|---:|
| identity | 68.02 | 57.66 | — | 68.02 (0.00) | +0.10 | n/a |
| blur s3 | 66.03 | 55.76 | −1.99 | 66.00 (−0.03) | **+0.14** | **0%** (τ=0.83) |
| noise s5 | 61.62 | 51.15 | **−6.40** | 61.58 (−0.04) | +0.10 | 17% (τ=0.35) |

Identity skip 68.02 on the 8k slice is consistent with CLIP 67.50 on full val.

## Blur s3 (1.05 h): oracle +0.14 pp, 14 better / 19 worse

MEMO changes 40 / 8000 predictions. McNemar: **17 vs 15** (base-correct/MEMO-wrong vs base-wrong/MEMO-correct). Gate never adapts: τ sits above the max score (0.824). Dense TTA is a **priced no-op**. The encoder is almost as good as ID; TTA has nothing to recover.

## Noise s5 (with ID control, 1.65 h): −6.40 pp drop, TTA still dead

Skip falls 68.02 → 61.62. Dense MEMO −0.04 pp. Oracle +0.10 pp. TENT / gated ≈ +0.04 pp at 17% adapt. Kill rule: skip drop ≥ 3–5 pp **and** (MEMO > skip or oracle ≥ 0.5 pp). First clause holds; second fails. 17% adapt is the correct abstention: the gate mostly refuses a method that cannot help.

## MaxProb

MaxProb AUROC is 0.827 / 0.816 / 0.795 (ID / blur / noise). Calibration is not TTA benefit. The gate's MaxProb ranking does not create a recoverable subset.

## Kill

Stop the visual-corruption TTA grid. Remaining A100 hours go to a non-visual residual (question-side or fusion), not more ImageNet-C.
