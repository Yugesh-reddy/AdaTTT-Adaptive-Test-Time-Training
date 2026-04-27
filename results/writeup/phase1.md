# Phase 1: CLIP encoder ceiling

Official VQA-v2 val, official soft = min(votes/3, 1), no UNK credit.

| Model | Soft (%) | Exact (%) | vs v1 soft | Verdict |
|---|---:|---:|---:|---|
| v1 ViT-B/16 + BERT-base | 54.30 | 49.56 | — | — |
| v2 ViT+BERT control | 59.93 | 49.56 | +5.63 | CONTINUE |
| v2 CLIP ViT-B/16 | **67.50** | **56.99** | +13.20 | **CONTINUE** |

CLIP − control = **+7.57 pp** soft / **+7.43 pp** exact (encoder, not fusion).

Slope ep5→ep8: CLIP +1.48 pp (66.02 → 67.50), control +0.93 pp (59.00 → 59.93). Both still climbing. Wall time 4.51 h on A100.

v1 exact 49.56 includes 4.55 pp UNK==UNK. Official soft is the primary metric.

Figure: `figures/phase1_ceiling.svg`.
