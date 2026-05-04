# AdaTTT — Adaptive Test-Time Training for VQA

A confidence-gated VQA system that decides *when* to do test-time training, so adaptation cost is paid only on samples that need it.

**Course:** CS 518 — Deep Learning for Computer Vision (Prof. Sathya N. Ravi, UIC)
**Authors:** Aishwarya Reddy Chinthalapudi, Yugesh Reddy Sappidi, Aryan Shetty

---

## Idea

Test-Time Training (TTT) takes a few self-supervised gradient steps on each test sample to improve accuracy, but it runs on every sample — including the easy ones. AdaTTT adds a learned confidence gate that routes each input either through a fast base path or through the TTT loop. The threshold τ is a runtime knob: one trained model gives you a Pareto frontier between speed and accuracy, no retraining required.

---

## Results

Evaluated on VQA-v2 validation (214,354 samples) and Memotion2 (2,797 samples, cross-task).

### VQA-v2 — accuracy vs. compute

| Configuration | Accuracy (95% CI) | Avg GFLOPs | Skip Rate |
|---|---|---|---|
| Base (no TTT) | 0.4956 ± 0.002 | 46.31 | 100% |
| TTT K=1, all samples | 0.4849 ± 0.002 | 64.91 | 0% |
| **AdaTTT τ=0.95, K=1** | **0.4952 ± 0.002** | **47.34** | **94.5%** |
| AdaTTT τ=0.9, K=1 | 0.4956 ± 0.002 | 46.49 | 99.0% |
| TTT K=3, all samples | 0.4666 ± 0.002 | 102.11 | 0% |
| TTT K=5, all samples | 0.4648 ± 0.002 | 139.31 | 0% |

The gate at τ=0.95 matches base accuracy at ~+1 GFLOP. Naively running more TTT steps on every sample actually *hurts* accuracy, which is the result the gate is built to address.

### Memotion2 — cross-task transfer

| Setting | Accuracy | Skip Rate |
|---|---|---|
| AdaTTT τ=0.8, K=1 | 0.7165 | 50.1% |

The gate trained on VQA-v2 transfers to meme sentiment without retraining.

### Latency (H100, batch=1, FP16)

| Stage | p50 (ms) | p95 (ms) |
|---|---|---|
| Image preprocess | 3.0 | 3.4 |
| ViT-B/16 (frozen) | 7.9 | 8.3 |
| BERT-base (frozen) | 10.6 | 11.1 |
| Fusion + predict | 4.0 | 4.2 |
| TTT step (when triggered) | 0.0 | 18.8 |
| **End-to-end** | **25.9** | **45.0** |

---

## Architecture

```
   Image ─▶ Frozen ViT-B/16 ─┐
                             ├─▶ Fusion Transformer ─▶ Confidence Gate
   Question ─▶ Frozen BERT ──┘                              │
                                                ┌───────────┴───────────┐
                                          conf ≥ τ                 conf < τ
                                            │                          │
                                       Prediction MLP           TTT Loop (K steps)
                                            │                  ├─ Masked patch SSL
                                            │                  ├─ Rotation
                                            │                  ├─ Consistency
                                            │                  └─ MixUp
                                            ▼                          ▼
                                          Answer                    Answer
```

Encoders stay frozen; TTT only updates the fusion + prediction head. The gate is supervised on prediction-correctness *deltas* between base and TTT-adapted forward passes — it learns "would TTT help here?" rather than "is the base right?", which is what makes it transfer.

---

## Repository

```
config/config.yaml          single source of truth for all hyperparameters
ttt/                        models, TTT loop, gate, data, metrics, latency, fallback
scripts/                    CPU: data prep, analysis, figures, gate labels
gpu/                        GPU: train, sweep, inference, ablation, profiling
demo/app.py                 Gradio demo
notebooks/                  hybrid Colab/local runner + tiny-dataset demo
report/                     IEEE conference paper (LaTeX + PDF)
figures/  results/  tests/  9 figures, JSON results with CIs, 94 tests
```

24 Python modules, ~8.6K LOC, 94 tests, 9 figures.

---

## Reproducing

CPU only — runs the test suite and the tiny demo:

```bash
pip install -e .
pytest tests/ -q
jupyter lab notebooks/tiny_dataset_demo.ipynb
```

Full pipeline (H100 recommended):

```bash
python scripts/01_prepare_data.py --config config/config.yaml
python gpu/train_base.py --config config/config.yaml
python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
python scripts/04_generate_gate_labels.py --config config/config.yaml --split train
python gpu/train_gate.py --base-checkpoint checkpoints/base/best.pt --split train
python gpu/run_inference.py \
    --base-checkpoint checkpoints/base/best.pt \
    --gate-checkpoint checkpoints/gate/best.pt \
    --threshold 0.8 --k 1
python scripts/02_analyze_results.py --config config/config.yaml
python scripts/03_generate_figures.py --config config/config.yaml
```

`gpu/precompute_features.py` caches frozen encoder activations; downstream eval scripts accept `--features` for a 5–10× speedup on sweeps.

Demo:

```bash
python demo/app.py --checkpoint checkpoints/base/best.pt --gate-checkpoint checkpoints/gate/best.pt
```

The Gradio UI exposes τ and K as sliders so you can move along the Pareto frontier interactively.

---

## Notes on the implementation

- Every reported number ships with a 95% bootstrap CI; pairwise comparisons use McNemar's test (`ttt/metrics.py`).
- `ttt/latency.py` includes a `LatencyBudget` context manager that falls through to the base path if a TTT step would exceed a configured wall-clock SLO.
- `ttt/fallback.py` implements a 4-level degradation chain (Full AdaTTT → Base → Reduced resolution → Error) for inference under load.
- Hyperparameters live entirely in `config/config.yaml`; seeds are pinned across NumPy, PyTorch, CUDA, and `random`.

---

## Findings

1. More TTT is not better — K=3 and K=5 on every sample underperform the base model on VQA-v2.
2. The gate transfers across tasks without retraining (VQA-v2 → Memotion2).
3. Among the four SSL objectives, masked-patch and mixup are roughly tied; consistency-only is weakest.
4. τ ∈ [0.9, 0.95] gives base accuracy at <2% compute overhead.

Full discussion in `report/conference_101719.pdf`.

---

## Stack

PyTorch 2.0+, HuggingFace Transformers (ViT-B/16, BERT-base), Gradio, pytest, Colab (H100), LaTeX (IEEEtran).

---

## Citation

```bibtex
@inproceedings{adattt2026,
  title     = {AdaTTT: Adaptive Test-Time Training for Visual Question Answering},
  author    = {Chinthalapudi, Aishwarya Reddy and Sappidi, Yugesh Reddy and Shetty, Aryan},
  booktitle = {CS 518 Final Project, University of Illinois Chicago},
  year      = {2026}
}
```

---

## Contact

Yugesh Reddy Sappidi — yugeshreddysappidi@gmail.com
