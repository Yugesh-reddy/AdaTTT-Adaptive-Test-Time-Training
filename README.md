# Efficient TTT: Adaptive Test-Time Training for VQA

An adaptive VQA system that learns to selectively apply test-time training using a confidence gate, producing a Pareto frontier of accuracy vs compute cost. Also evaluated on Memotion2 meme sentiment classification for cross-task generalization.

**Authors:** Aishwarya Reddy Chinthalapudi, Yugesh Reddy Sappidi & Aryan Shetty
**Course:** CS 518 — Deep Learning for Computer Vision (Prof. Sathya N. Ravi, UIC)

## Key Results

| Configuration | VQA Accuracy | Avg GFLOPs | Skip Rate |
|--------------|-------------|------------|-----------|
| Base (no TTT) | X% | 40.9 | 100% |
| TTT K=1 (masked patch) | X% | 44.1 | 0% |
| Adaptive τ=0.8, K=1 | X% | X | X% |

*Results will be filled after full training runs on H100.*

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
│   ├── utils.py                # Config, checkpointing, I/O
│   ├── latency.py              # LatencyProfiler + LatencyBudget
│   └── fallback.py             # GracefulPredictor (4-level degradation)
├── scripts/                    # Local CLI scripts (no GPU)
│   ├── 01_prepare_data.py      # Download & preprocess
│   ├── 02_analyze_results.py   # Compute metrics
│   ├── 03_generate_figures.py  # 11 publication figures
│   └── 04_generate_gate_labels.py  # Gate label generation
├── gpu/                        # Colab GPU scripts
│   ├── train_base.py           # Train base VQA model
│   ├── train_gate.py           # Refine confidence gate
│   ├── run_ttt_sweep.py        # Sweep K × objective
│   ├── run_inference.py        # Adaptive inference
│   ├── run_ablation.py         # Stabilization ablation
│   ├── run_component_ablation.py  # Which-modules-to-adapt ablation
│   ├── run_gate_sweep.py       # Single-pass threshold sweep
│   ├── run_warmup_analysis.py  # TTT warmup cost analysis
│   └── run_latency_profile.py  # Per-stage latency profiling
├── demo/app.py                 # Gradio interactive demo
├── notebooks/colab_runner.ipynb # Colab notebook
├── tests/                      # Unit + integration tests
└── setup.py                    # pip install -e .
```

## Quick Start

### 1. Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
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

## Demo

Launch the interactive Gradio demo:
```bash
python demo/app.py --checkpoint checkpoints/base/best.pt
```
Upload an image, type a question, and adjust TTT steps / gate threshold to explore the accuracy-latency tradeoff in real time.

## Experiments & Findings

### Latency Budget
Profile per-stage latency with:
```bash
!python gpu/run_latency_profile.py --checkpoint checkpoints/base/best.pt --k 1
```
See Figure 8 for the stacked latency breakdown.

### Graceful Degradation
The `GracefulPredictor` provides a 4-level fallback chain (Full AdaTTT → Base Only → Reduced Resolution → Error) for production robustness.

### Component Ablation
```bash
!python gpu/run_component_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode fusion_only
```
Tests which modules benefit most from TTT adaptation.

### Gate Threshold Sweep
```bash
!python gpu/run_gate_sweep.py --checkpoint checkpoints/base/best.pt --k 1
```
Efficient single-pass sweep over all thresholds.

### TTT Warmup Analysis
```bash
!python gpu/run_warmup_analysis.py --checkpoint checkpoints/base/best.pt --k 1 --mode cumulative
```
Tests whether accumulated TTT adaptations help or hurt.

## Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- Transformers >= 4.36
- Gradio >= 4.0 (for demo)
- Google Colab Pro (H100) for GPU tasks
