# CLAUDE.md — Efficient TTT: Adaptive Test-Time Training for VQA

## Project Summary

Build an adaptive VQA system where a confidence gate routes "easy" test samples around expensive TTT adaptation and "hard" samples through it. Central deliverable: a Pareto frontier of accuracy vs compute cost at different gating thresholds. Cross-task evaluation on Memotion2 meme sentiment classification tests generalization beyond VQA.

**Authors:** Aishwarya Reddy Chinthalapudi, Yugesh Reddy Sappidi & Aryan Shetty
**Course:** CS 518 — Deep Learning for Computer Vision (Prof. Sathya N. Ravi, UIC)
**Execution:** Google Colab Pro (H100). No local GPU. All GPU work via `!python gpu/script.py`. Non-GPU work (data prep, eval, figures) runs locally.

---

## Quick Start

```bash
# Local setup
pip install -e .  # or: pip install -r requirements.txt

# Run tests
pytest tests/

# Local scripts (no GPU)
python scripts/01_prepare_data.py
python scripts/02_analyze_results.py
python scripts/03_generate_figures.py
python scripts/04_generate_gate_labels.py

# GPU scripts (Colab only)
!python gpu/train_base.py --config config/config.yaml
!python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
```

---

## Architecture

```
Image I + Question Q
    │            │
    ▼            ▼
 Frozen ViT-B/16    Frozen BERT-base
 v ∈ R^(197×768)    t ∈ R^(L×768)
    └──────┬─────────┘
           ▼
  Cross-Modal Fusion (θ_f, ~4.7M params)
  2-layer bidirectional cross-attention transformer
  Q=v,K=t,V=t + reverse Q=t,K=v,V=v + LayerNorm + FFN
           │
           ▼ z ∈ R^(768)
  Confidence Gate (θ_g, ~50K params)
  MLP: 768→256→ReLU→Dropout(0.1)→1→Sigmoid
  g(z) > τ → SKIP TTT | g(z) ≤ τ → ADAPT
      │                     │
      │              TTT Adaptation
      │              K gradient steps on θ_f, θ_d
      │              using L_self_sup (masked patch or rotation)
      │                     │ z' (adapted)
      ▼                     ▼
  Prediction Head (θ_d, ~2.3M params)
  MLP: 768→1024→ReLU→Dropout(0.2)→LayerNorm→N
  N = 3129 (VQA-v2) or num_classes (Memotion2)
```

**Trainable:** θ_f (fusion), θ_g (gate), θ_d (prediction head)
**Frozen:** ViT-B/16, BERT-base (always `requires_grad=False` + `torch.no_grad()` on forward)

---

## Repository Structure

```
AdaTTT/
├── CLAUDE.md
├── config/config.yaml              # Single source of truth for all hyperparams
├── ttt/                             # Core package
│   ├── __init__.py                  # Exports: FullVQAModel, FusionModule, ConfidenceGate, PredictionHead, TTTAdapter, AdaptiveRouter, LatencyProfiler, GracefulPredictor, datasets
│   ├── data.py                      # VQADataset, VizWizDataset, Memotion2Dataset, answer vocab builder, download utils
│   ├── models.py                    # FusionModule, ConfidenceGate, PredictionHead, FullVQAModel, frozen encoder loaders
│   ├── ttt_loop.py                  # TTTAdapter: adapt_and_predict, masked_patch_loss, rotation_loss, consistency_loss, mixup_anchor
│   ├── gate.py                      # AdaptiveRouter: routing logic + FLOPs computation
│   ├── metrics.py                   # vqa_accuracy, accuracy_by_question_type, pareto_frontier, gate statistics
│   ├── latency.py                   # LatencyProfiler: per-stage latency profiling with P50/P95
│   ├── fallback.py                  # GracefulPredictor: 4-level degradation (full→base→reduced_res→error)
│   └── utils.py                     # Config loading, JSON I/O, checkpointing (trainable params only), logging
├── scripts/                         # LOCAL (no GPU)
│   ├── 01_prepare_data.py           # Download + preprocess VQA-v2 / VizWiz / Memotion2
│   ├── 02_analyze_results.py        # Compute metrics from saved predictions, Pareto frontier, McNemar's test
│   ├── 03_generate_figures.py       # 7 publication figures
│   └── 04_generate_gate_labels.py   # Compare base vs TTT predictions → gate training labels
├── gpu/                             # COLAB ONLY (need GPU)
│   ├── train_base.py                # Train fusion + gate + prediction head (supervised, no TTT)
│   ├── train_gate.py                # Refine gate using proper labels from 04_generate_gate_labels
│   ├── run_ttt_sweep.py             # Sweep: K steps × TTT objective; saves per-(K,objective) result files
│   ├── run_inference.py             # Full adaptive pipeline with trained gate at threshold τ
│   ├── run_ablation.py              # TTT stabilization: ttt_only / ttt_consistency / ttt_mixup / ttt_both
│   ├── run_component_ablation.py    # Which modules to adapt: fusion_only / pred_only / both / all
│   ├── run_gate_sweep.py            # Single-pass efficient threshold sweep
│   ├── run_warmup_analysis.py       # Cumulative vs per-sample restore TTT analysis
│   └── run_latency_profile.py       # Per-stage latency profiling
├── notebooks/colab_runner.ipynb     # Mount drive → pip install → run gpu/ scripts
├── checkpoints/{base,gate}/         # Saved model weights (trainable params only)
├── results/                         # base_predictions.json, ttt_predictions/, ablation/, memotion2/
├── figures/                         # Generated publication figures
├── demo/app.py                      # Gradio interactive demo
├── tests/                           # test_models, test_ttt, test_gate, test_metrics, test_latency, test_fallback, test_integration
├── setup.py                         # pip install -e . for clean imports
└── requirements.txt                 # torch, torchvision, transformers, datasets, pillow, pyyaml, numpy, matplotlib, scipy, tqdm, gradio
```

---

## Key Config Parameters (`config/config.yaml`)

| Category | Parameter | Value |
|----------|-----------|-------|
| **Encoders** | vision_encoder | google/vit-base-patch16-224 (frozen) |
| | text_encoder | bert-base-uncased (frozen) |
| **Fusion** | layers / dim / heads / dropout | 2 / 768 / 12 / 0.1 |
| **Prediction** | hidden / num_answers | 1024 / 3129 (VQA-v2) |
| **Gate** | hidden / threshold sweep | 256 / [0.5, 0.7, 0.8, 0.9, 0.95] |
| **TTT** | objectives | masked_patch, rotation |
| | k_steps sweep | [0, 1, 2, 3, 5] |
| | lr / mask_ratio | 1e-4 / 0.25 |
| | adapt_modules | fusion + prediction_head |
| **Stabilization** | consistency_weight | 0.1 (symmetric KL on 2 augmented views) |
| | consistency_augs | random_crop, color_jitter |
| | mixup_alpha_range | [0.7, 1.0] (anchor interpolation) |
| **Base training** | batch/lr/epochs/optimizer | 64 / 1e-4 / 15 / AdamW (wd=0.01, cosine schedule, 10% warmup) |
| **Gate training** | batch/lr/epochs | 128 / 5e-4 / 5 |
| **Reproducibility** | seed | 42 |
| **Data** | image_size / max_question_length | 224 / 20 |
| | strict_images | true (fail fast on missing/corrupt) |
| **Latency** | warmup_runs / num_samples | 5 / 100 |
| **Fallback** | ttt_timeout_ms / reduced_resolution | 500 / 160 |
| **Memotion2** | num_classes | 3 (positive/negative/neutral) |
| **TTT** | grad_clip | 1.0 (max gradient norm) |

---

## Module Responsibilities

### `ttt/data.py`
- **VQA-v2:** COCO images + VQA-v2 questions/annotations. Answer vocab = top 3129 answers. Samples: `{image, input_ids, attention_mask, answer_idx, question_type, sample_id}`. Question types (yes/no, number, other) needed for per-type analysis.
- **VizWiz:** Same format. Has "unanswerable" class as natural difficulty signal.
- **Memotion2:** Meme images + OCR text, sentiment labels (positive/negative/neutral). Reuses same encoders + fusion; only prediction head output dim changes.
- Image preprocessing: resize 224, ImageNet normalize. Text: BERT WordPiece, pad/truncate to 20.
- Answer vocab built once from training annotations, saved to `data/answer_vocab.json`.

### `ttt/models.py`
- **FusionModule:** 2-layer bidirectional cross-attention (Q=visual,K=text,V=text + reverse). Outputs pooled z ∈ R^768. Optionally returns full sequence (for masked patch loss).
- **ConfidenceGate:** 768→256→ReLU→Dropout→1→Sigmoid. `route(z, threshold)` returns boolean skip mask.
- **PredictionHead:** 768→1024→ReLU→Dropout(0.2)→LayerNorm→num_answers.
- **FullVQAModel:** Combines all components. `forward()` returns (logits, confidence, z). Key methods: `get_ttt_params()` (fusion + pred head, NOT gate), `encode()` (frozen encoders only), `fuse_and_predict()` (fusion + pred on pre-encoded tokens).

### `ttt/ttt_loop.py` — Most Critical File
**TTTAdapter** performs per-sample test-time training:
1. Save anchor state (clone of θ_f, θ_d)
2. Create fresh Adam optimizer per adaptation episode
3. K gradient steps with self-supervised loss on θ_f and θ_d only
4. Predict with adapted params
5. **RESTORE** params to anchor state (critical — prevents drift)

**Self-supervised objectives:**
- **Masked patch:** Mask 25% of visual tokens (indices 1-196), predict masked token mean from pooled z via Linear(768,768), MSE loss
- **Rotation:** Rotate image by {0,90,180,270}°, re-encode through frozen ViT, predict rotation class via Linear(768,4), CrossEntropy loss

**Stabilization techniques (for ablation):**
- **Consistency regularization:** Two augmented views (RandomResizedCrop + ColorJitter) → fuse both → symmetric KL divergence on prediction distributions. Added with weight λ_cons=0.1.
- **Mixup anchoring:** z_mixed = α*z_current + (1-α)*z_anchor.detach(), α ∈ [0.7, 1.0]. Keeps adapted representation close to original.

### `ttt/latency.py`
- **LatencyBudget** dataclass: Stores per-stage timing (image_preprocess_ms, vision_encode_ms, text_encode_ms, fusion_predict_ms, ttt_adaptation_ms, total_ms). Property `ttt_fraction` = ttt_adaptation_ms / total_ms.
- **LatencyProfiler** class: `profile_single(image_pil, question, threshold, k_steps)` → `LatencyBudget`. Times each stage independently (ViT, BERT, fusion, TTT). `profile_batch(dataset, n_samples, warmup_runs)` → dict with P50/P95 per stage. Uses `torch.cuda.synchronize()` on CUDA for accurate timing.

### `ttt/fallback.py`
- **FallbackLevel** IntEnum: FULL_ADATTT=0, BASE_ONLY=1, REDUCED_RESOLUTION=2, ERROR=3.
- **FallbackResult** dataclass: answer_idx, logits, level, reason, latency_ms.
- **GracefulPredictor** class: Wraps AdaptiveRouter with fallback chain. Level 0→1: skip TTT on timeout/error. Level 1→2: catch OOM, retry at reduced resolution (160px→224px). Level 2→3: catch-all, return error. Uses `torch.cuda.empty_cache()` after OOM.

### `ttt/gate.py`
**AdaptiveRouter:** Encodes all → fuses → gate splits batch into SKIP (high conf) and ADAPT (low conf) → TTT on ADAPT only → recombine predictions.

**FLOPs estimates:** SKIP ≈ 40.9 GFLOPs (encode+fuse+predict). ADAPT ≈ 40.9 + K×3.2 GFLOPs. Average = weighted by skip/adapt counts.

### `ttt/metrics.py`
- VQA accuracy: top-1 match against most-frequent ground truth answer
- Per-question-type breakdown (yes/no, number, other)
- Pareto frontier: filters dominated (accuracy, FLOPs) points
- Gate statistics: skip/adapt counts, rates, per-group accuracy

---

## GPU Scripts

### `gpu/train_base.py`
Trains θ_f + θ_g + θ_d with dual loss: `L = CrossEntropy(logits, answers) + 0.1 * BCE(gate_confidence, correct_prediction_indicator)`. Gate proxy: predict whether base model gets it right. Saves `checkpoints/base/best.pt` + `results/base_predictions.json`.

### `gpu/train_gate.py`
Refines θ_g only using proper gate labels (from `04_generate_gate_labels.py`). Fusion + pred head frozen. BCE loss against gate labels. Saves `checkpoints/gate/best.pt`.

### `gpu/run_ttt_sweep.py`
Sweeps K×objective on val set. One results file per (K, objective): `results/ttt_predictions/k{K}_{objective}.json`. Threshold τ applied post-hoc in analysis. Times: K=0 ~10min, K=1 ~25min, K=3 ~55min, K=5 ~90min (estimated, will be faster on H100).

### `gpu/run_ablation.py`
Fixed K=1, masked_patch. Four modes: ttt_only, ttt_consistency, ttt_mixup, ttt_both. Saves `results/ablation/{mode}_k{K}.json`.

### `gpu/run_inference.py`
Full adaptive pipeline with trained gate. Per threshold τ: encode → fuse → gate route → SKIP/ADAPT → save predictions + routing info + FLOPs. Run once per τ to build Pareto curve.

### `gpu/run_component_ablation.py`
Ablation: which modules to adapt during TTT. Modes: fusion_only, pred_only, both (default), all (+gate). Overrides `config["ttt_adapt_modules"]` per mode. Saves `results/component_ablation/{mode}_k{K}.json`.

### `gpu/run_gate_sweep.py`
Efficient single-pass threshold sweep. Runs TTT on ALL samples once, saves base_logits and ttt_logits, then applies each threshold post-hoc. Much faster than N runs of run_inference.py. Saves `results/gate_sweep.json`.

### `gpu/run_warmup_analysis.py`
Tests accumulated TTT vs per-sample restore. Mode "cumulative" skips the restore step, letting params drift. Tracks rolling accuracy and parameter drift (L2 from anchor). Saves `results/warmup_analysis.json`.

### `gpu/run_latency_profile.py`
Profiles per-stage latency. Creates LatencyProfiler, runs warmup + N samples, reports P50/P95 per stage. Saves `results/latency_profiles.json`.

---

## Local Scripts

### `scripts/04_generate_gate_labels.py`
Compares base vs TTT predictions per sample:
- base_correct AND ttt_correct → label 1.0 (SKIP)
- NOT base_correct AND ttt_correct → label 0.0 (ADAPT, TTT helps)
- base_correct AND NOT ttt_correct → label 1.0 (SKIP, TTT hurts)
- Neither correct → label 0.5 (uncertain)
Saves `data/gate_labels.json`.

### `scripts/02_analyze_results.py`
Computes: per-config accuracy, per-question-type accuracy, avg FLOPs, Pareto frontier, gate routing stats, McNemar's test. Saves `results/analysis_summary.json`.

### `scripts/03_generate_figures.py`
7 figures: (1) Pareto frontier [main deliverable], (2) TTT steps vs accuracy, (3) Gate routing analysis by τ, (4) Per-question-type breakdown, (5) TTT stabilization ablation bars, (6) Confidence distribution for correct vs incorrect, (7) Memotion2 cross-task generalization comparison.

---

## End-to-End Workflow

```
LOCAL:  python scripts/01_prepare_data.py
COLAB:  !python gpu/train_base.py                                    # ~4-6 hrs
COLAB:  !python gpu/run_ttt_sweep.py --k {0,1,2,3,5} --objective {masked_patch,rotation}
LOCAL:  python scripts/04_generate_gate_labels.py
COLAB:  !python gpu/train_gate.py                                    # ~30 min
COLAB:  !python gpu/run_inference.py --threshold {0.5,0.7,0.8,0.9,0.95} --k 1
COLAB:  !python gpu/run_ablation.py --mode {ttt_only,ttt_consistency,ttt_mixup,ttt_both}
COLAB:  !python gpu/run_component_ablation.py --checkpoint ... --k 1 --mode {fusion_only,pred_only,both,all}
COLAB:  !python gpu/run_gate_sweep.py --checkpoint ... --k 1
COLAB:  !python gpu/run_warmup_analysis.py --checkpoint ... --k 1 --mode {cumulative,restore}
COLAB:  !python gpu/run_latency_profile.py --checkpoint ... --k {0,1}
COLAB:  (Memotion2) train_base --dataset memotion2 → run_ttt_sweep → run_inference
LOCAL:  python scripts/02_analyze_results.py
LOCAL:  python scripts/03_generate_figures.py
LOCAL:  python demo/app.py --checkpoint checkpoints/base/best.pt
```

---

## Memory Budget (H100 80GB)

| Component | VRAM |
|-----------|------|
| ViT-B/16 + BERT-base (frozen) | ~760 MB |
| Trainable modules (fusion + gate + pred head) | ~55 MB |
| Activations (batch=64, training) | ~4 GB |
| Gradients + optimizer states | ~600 MB |
| **Training total** | **~5.5 GB** |
| TTT inference (K=1, batch=1) | **~7 GB** |

H100 has 80GB VRAM — massive headroom. Can use larger batches freely.

---

## Critical Implementation Gotchas

1. **TTT parameter restore is CRITICAL.** After every adaptation, restore θ_f and θ_d via `param.data.copy_(saved_state)`. Without this, params drift and accuracy collapses after ~100 samples.
2. **Frozen encoders must stay frozen.** `requires_grad=False` for all ViT/BERT params. Wrap encoder forwards in `torch.no_grad()`.
3. **TTT creates a NEW optimizer per sample/batch.** Fresh `Adam(model.get_ttt_params(), lr=ttt_lr)` each time. Never reuse the training optimizer.
4. **Gate is frozen at test time.** Only θ_f and θ_d are adapted by TTT. θ_g makes routing decisions but is never updated during inference.
5. **FLOPs tracking is per-sample.** SKIP = constant cost, ADAPT = cost proportional to K.
6. **Lazy image loading.** Never load all VQA images into memory. Load per-batch from disk/Drive.
7. **Answer vocab.** Build once, save to `data/answer_vocab.json`, load everywhere.
# Update CLAUDE.md with new workflow docs
