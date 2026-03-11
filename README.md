# Efficient TTT: Adaptive Test-Time Training for VQA

An adaptive VQA system that learns to selectively apply test-time training using a confidence gate, producing a Pareto frontier of accuracy vs compute cost. Also evaluated on Memotion2 meme sentiment classification for cross-task generalization.

**Authors:** Aishwarya Reddy Chinthalapudi, Yugesh Reddy Sappidi & Aryan Shetty  
**Course:** CS 518 — Deep Learning for Computer Vision (Prof. Sathya N. Ravi, UIC)

## Architecture

```
Image + Question → [Frozen ViT + BERT] → Cross-Modal Fusion → Confidence Gate
                                                                    │
                                           ┌────────────────────────┤
                                           │ SKIP (confident)       │ ADAPT (uncertain)
                                           ↓                        ↓
                                     Prediction MLP          TTT Adaptation (K steps)
                                           ↓                        ↓
                                         Answer                   Answer
```

## Project Structure

```
AdaTTT/
├── config/config.yaml          # All hyperparameters
├── ttt/                        # Core Python package
│   ├── models.py               # FusionModule, ConfidenceGate, PredictionHead, FullVQAModel
│   ├── ttt_loop.py             # TTT adaptation (masked patch, rotation, consistency, mixup)
│   ├── gate.py                 # AdaptiveRouter (skip/adapt routing)
│   ├── data.py                 # VQA-v2 / VizWiz / Memotion2 dataset classes
│   ├── metrics.py              # VQA accuracy, Pareto frontier, McNemar's test
│   └── utils.py                # Config, checkpointing, I/O
├── scripts/                    # Local CLI scripts (no GPU)
│   ├── 01_prepare_data.py      # Download & preprocess
│   ├── 02_analyze_results.py   # Compute metrics
│   ├── 03_generate_figures.py  # 6 publication figures
│   └── 04_generate_gate_labels.py  # Gate label generation
├── gpu/                        # Colab GPU scripts
│   ├── train_base.py           # Train base VQA model
│   ├── train_gate.py           # Refine confidence gate
│   ├── run_ttt_sweep.py        # Sweep K × objective
│   ├── run_inference.py        # Adaptive inference
│   └── run_ablation.py         # Stabilization ablation
├── notebooks/colab_runner.ipynb # Colab notebook
└── tests/                      # Unit tests (37 passing)
```

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (local)
```bash
python scripts/01_prepare_data.py
```

### 3. Train Base Model (Colab GPU)
```bash
!python gpu/train_base.py --config config/config.yaml
```

### 4. Run TTT Sweep (Colab GPU)
```bash
!python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
```

### 5. Generate Gate Labels (local)
```bash
python scripts/04_generate_gate_labels.py
```

### 6. Train Gate + Adaptive Inference (Colab GPU)
```bash
!python gpu/train_gate.py --base-checkpoint checkpoints/base/best.pt
!python gpu/run_inference.py --threshold 0.8 --k 1
```

### 7. Analyze & Visualize (local)
```bash
python scripts/02_analyze_results.py
python scripts/03_generate_figures.py
```

### 8. Memotion2 Cross-Task Evaluation (Colab GPU)
```bash
!python gpu/train_base.py --config config/config.yaml --dataset memotion2
!python gpu/run_inference.py --threshold 0.8 --k 1 --dataset memotion2
```

## Tests

```bash
python -m pytest tests/ -v
```

37 tests passing, 1 skipped (requires torchvision for rotation transforms).

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Transformers ≥ 4.36
- Google Colab Pro (T4 or A100) for GPU tasks
