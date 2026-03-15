#!/usr/bin/env python3
"""
Gradio interactive demo for AdaTTT.

Allows uploading an image and asking a question, with controls for
TTT steps, gate threshold, and gate enable/disable. Displays the
base answer, adapted answer, gate routing info, and latency breakdown.

Usage:
    python demo/app.py
    python demo/app.py --checkpoint checkpoints/base/best.pt
    python demo/app.py --share  # public Gradio link
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Globals (loaded once at startup)
_model = None
_ttt_adapter = None
_profiler = None
_image_transform = None
_tokenizer = None
_answer_vocab_reverse = None
_config = None
_device = None


def load_model(checkpoint_path, gate_checkpoint_path=None, config_path="config/config.yaml"):
    """Load model and all components."""
    global _model, _ttt_adapter, _profiler, _image_transform
    global _tokenizer, _answer_vocab_reverse, _config, _device

    from ttt.models import FullVQAModel
    from ttt.ttt_loop import TTTAdapter
    from ttt.latency import LatencyProfiler
    from ttt.data import get_image_transform
    from ttt.utils import load_config, load_checkpoint, get_device, set_seed

    _config = load_config(config_path)
    set_seed(_config.get("seed", 42))
    _device = get_device()

    if _device.type != "cuda":
        print("WARNING: No CUDA device found. Running on CPU (slow).")

    # Load model
    _model = FullVQAModel(_config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        _model.load_encoders(_config)
        load_checkpoint(_model, checkpoint_path)
        if gate_checkpoint_path and os.path.exists(gate_checkpoint_path):
            gate_ckpt = torch.load(gate_checkpoint_path, map_location="cpu", weights_only=True)
            _model.gate.load_state_dict(gate_ckpt["gate"])
        _model = _model.to(_device)
        _model.eval()
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Model has random weights.")
        _model.load_encoders(_config)
        _model = _model.to(_device)
        _model.eval()

    # Create TTT adapter
    _ttt_adapter = TTTAdapter(_model, _config, objective="masked_patch", k_steps=1)

    # Image transform and tokenizer
    _image_transform = get_image_transform(_config.get("image_size", 224))
    from transformers import BertTokenizer
    _tokenizer = BertTokenizer.from_pretrained(
        _config.get("text_encoder", "bert-base-uncased")
    )

    # Latency profiler
    _profiler = LatencyProfiler(_model, _ttt_adapter, _image_transform, _tokenizer, _config)

    # Answer vocab
    vocab_path = os.path.join(_config.get("data_dir", "data/"), "answer_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            answer_vocab = json.load(f)
        _answer_vocab_reverse = {v: k for k, v in answer_vocab.items()}
    else:
        _answer_vocab_reverse = None
        print(f"WARNING: Answer vocab not found at {vocab_path}. Will show indices.")


def idx_to_answer(idx):
    """Convert answer index to string."""
    if _answer_vocab_reverse and idx in _answer_vocab_reverse:
        return _answer_vocab_reverse[idx]
    return f"answer_{idx}"


def create_latency_figure(budget):
    """Create a horizontal stacked bar chart of latency breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["Preprocess", "ViT Encode", "BERT Encode", "Fusion+Predict"]
    values = [
        budget.image_preprocess_ms,
        budget.vision_encode_ms,
        budget.text_encode_ms,
        budget.fusion_predict_ms,
    ]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800"]

    if budget.ttt_triggered:
        stages.append("TTT Adapt")
        values.append(budget.ttt_adaptation_ms)
        colors.append("#F44336")

    fig, ax = plt.subplots(figsize=(8, 1.8))

    left = 0
    for stage, val, color in zip(stages, values, colors):
        ax.barh(0, val, left=left, height=0.5, color=color, label=f"{stage}: {val:.1f}ms")
        left += val

    ax.set_xlim(0, max(left * 1.1, 1))
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Latency Breakdown — Total: {budget.total_ms:.1f}ms")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def predict(image_pil, question, num_ttt_steps, threshold, enable_gate):
    """Run prediction and return results."""
    if _model is None:
        return "Model not loaded", "", "{}", None

    if image_pil is None:
        return "Please upload an image", "", "{}", None
    if not question or not question.strip():
        return "Please enter a question", "", "{}", None

    # Profile and get latency
    _ttt_adapter.k_steps = int(num_ttt_steps)
    budget = _profiler.profile_single(
        image_pil, question, threshold=threshold, k_steps=int(num_ttt_steps),
    )

    # Get actual predictions
    image_tensor = _image_transform(image_pil).unsqueeze(0).to(_device)
    max_len = _config.get("max_question_length", 20)
    encoding = _tokenizer(
        question, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        visual_tokens, text_tokens = _model.encode(image_tensor, input_ids, attention_mask)
        z = _model.fusion(visual_tokens, text_tokens, attention_mask)
        confidence = _model.gate(z).item()
        base_logits = _model.prediction_head(z)
        base_pred = base_logits.argmax(dim=-1).item()

    base_answer = idx_to_answer(base_pred)

    # TTT adaptation
    adapted_answer = base_answer
    if int(num_ttt_steps) > 0:
        skip = enable_gate and confidence > threshold
        if not skip:
            _ttt_adapter.k_steps = int(num_ttt_steps)
            adapted_logits, ttt_loss = _ttt_adapter.adapt_and_predict(
                image_tensor, visual_tokens, text_tokens, attention_mask,
            )
            adapted_pred = adapted_logits.argmax(dim=-1).item()
            adapted_answer = idx_to_answer(adapted_pred)
        else:
            adapted_answer = f"{base_answer} (gate: SKIP)"

    # Gate info
    gate_info = {
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "decision": "SKIP" if (enable_gate and confidence > threshold) else "ADAPT",
        "ttt_steps": int(num_ttt_steps),
        "latency_ms": round(budget.total_ms, 1),
        "ttt_fraction": round(budget.ttt_fraction, 3),
    }

    # Latency figure
    latency_fig = create_latency_figure(budget)

    return base_answer, adapted_answer, json.dumps(gate_info, indent=2), latency_fig


def main():
    parser = argparse.ArgumentParser(description="AdaTTT Gradio Demo")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base/best.pt")
    parser.add_argument("--gate-checkpoint", type=str, default=None)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import gradio as gr

    # Load model
    load_model(args.checkpoint, args.gate_checkpoint, args.config)

    # Build demo examples
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    examples = []
    if os.path.exists(examples_dir):
        for img_file in sorted(os.listdir(examples_dir)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                examples.append([
                    os.path.join(examples_dir, img_file),
                    "What is in this image?",
                    1, 0.8, True,
                ])

    # Build Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(label="Question", placeholder="What is in this image?"),
            gr.Slider(0, 10, value=1, step=1, label="TTT Steps (K)"),
            gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Gate Threshold (τ)"),
            gr.Checkbox(value=True, label="Enable Gate"),
        ],
        outputs=[
            gr.Textbox(label="Base Answer"),
            gr.Textbox(label="Adapted Answer"),
            gr.Textbox(label="Gate Info (JSON)"),
            gr.Plot(label="Latency Breakdown"),
        ],
        title="AdaTTT: Adaptive Test-Time Training for VQA",
        description=(
            "Upload an image and ask a question. The model uses a confidence gate "
            "to decide whether to apply test-time training (TTT) for harder questions. "
            "Adjust TTT steps and gate threshold to explore the accuracy-latency tradeoff."
        ),
        examples=examples if examples else None,
        allow_flagging="never",
    )

    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
