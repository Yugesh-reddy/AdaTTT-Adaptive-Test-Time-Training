# AdaTTT writeup: Phase 1 win, Phase 2 visual-TTA negative

Numbers: `numbers.json`. Figures: `figures/`. Pointers to gitignored JSON: `ARTIFACTS.md`.

## 1. Phase 1 — CLIP ceiling (CONTINUE)

Official VQA-v2 val. Soft = min(votes/3, 1), no UNK credit.

| Model | Soft | Exact | vs v1 soft | vs control | Verdict |
|---|---:|---:|---:|---:|---|
| v1 ViT-B/16 + BERT-base | 54.30 | 49.56 | — | — | — |
| v2 ViT+BERT control | 59.93 | 49.56 | +5.63 | — | CONTINUE |
| v2 CLIP ViT-B/16 | **67.50** | **56.99** | +13.20 | **+7.57 / +7.43** | **CONTINUE** |

CLIP ep5 66.02 → ep8 67.50 (+1.48). Control ep5 59.00 → 59.93 (+0.93). 4.51 h A100. Encoder, not fusion, is the lift. v1 exact 49.56 contains 4.55 pp UNK==UNK.

![Phase 1 ceiling](figures/phase1_ceiling.svg)

## 2. Phase 2 — visual TTA negative

Frozen eval 8k. Fusion LN-only. τ fit on the matching gate-train 8k corruption. Reported FLOPs are `sample_flops`: **24.8 / 43.4 / 91.0 / 138.6** GFLOPs.

### 2.1 Accuracy vs identity

| Condition | Skip soft | Exact | vs ID | MEMO (dense) | Oracle | Gate |
|---|---:|---:|---:|---:|---:|---:|
| identity | 68.02 | 57.66 | — | 0.00 pp | +0.10 | — |
| blur s3 | 66.03 | 55.76 | −1.99 | −0.03 | **+0.14** | **0%** |
| noise s5 | 61.62 | 51.15 | **−6.40** | −0.04 | +0.10 | 17% |

![Shift drop](figures/shift_drop.svg)

### 2.2 Pareto (soft vs sample_flops)

δ in percentage points vs skip. p = adapt rate. δ/p is pp per unit adapt rate.

| Method | GFLOPs | blur s3 | δ | p | δ/p | noise s5 | δ | p | δ/p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| skip / no_adapt | 24.8 | 66.03 | 0 | 0 | — | 61.62 | 0 | 0 | — |
| TENT k=1 | 43.4 | 65.99 | −0.04 | 1.00 | −0.04 | 61.66 | +0.04 | 1.00 | +0.04 |
| EATA k=1 | ~41 | 65.99 | −0.04 | 0.86 | −0.05 | 61.66 | +0.03 | 0.85 | +0.04 |
| MEMO k=1, 4 views | 138.6 | 66.00 | −0.03 | 1.00 | −0.03 | 61.58 | −0.04 | 1.00 | −0.04 |
| MEMO-SAR | ~131 | 66.00 | −0.03 | 0.93 | −0.04 | 61.58 | −0.05 | 0.93 | −0.05 |
| gated MEMO-SAR | 24.8 / 44.5 | 66.03 | 0 | **0.00** | — | 61.66 | +0.04 | **0.17** | +0.22 |
| oracle | mixed | 66.17 | **+0.14** | ~0 | — | 61.73 | +0.10 | ~0 | — |

MEMO-2 (91.0G) is on the ladder and was not run; dense MEMO already sits on the 138.6G rung with a non-positive δ. Gated blur stays on the 24.8G skip arm (τ=0.83 > max score 0.824). Gated noise spends 17% of samples on MEMO-SAR (p95 138.6G) and still cannot beat skip by more than 0.04 pp.

![Pareto](figures/pareto_flops.svg)

### 2.3 Blur s3 McNemar (MEMO vs skip)

n=8000. Pred flips: 40. Soft-better **14** / soft-worse **19**. Exact McNemar: **17 vs 15** (base-correct/MEMO-wrong vs base-wrong/MEMO-correct). Oracle +0.14 pp. Priced at 1.05 h.

![McNemar](figures/blur_mcnemar.svg)

### 2.4 Noise s5 kill

ID 68.02 → skip 61.62 (−6.40 pp). MEMO −0.04. Oracle +0.10. Gate 17% adapt is the correct abstention pattern: it mostly refuses a method that cannot help. Kill rule fails the recovery clause. 1.65 h for ID + noise.

![Oracle vs dense](figures/oracle_vs_dense.svg)

### 2.5 MaxProb is not TTA

| Condition | MaxProb AUROC | MaxProb AURC | TTA oracle (pp) |
|---|---:|---:|---:|
| identity | 0.827 | 0.193 | +0.10 |
| blur s3 | 0.816 | 0.213 | +0.14 |
| noise s5 | 0.795 | 0.269 | +0.10 |

AUROC tracks difficulty. It does not imply a recoverable visual residual.

## 3. Decision

Phase 1: keep CLIP. Phase 2 visual grid: **stop**. Next residual is not ImageNet-C.
