#!/usr/bin/env python3
"""
Gradio interactive demo for AdaTTT.

Supports two interactive task demos:
    - VQA: upload an image and ask a question
    - Memotion2: upload a meme image and provide meme/OCR text for sentiment

Usage:
    python demo/app.py
    python demo/app.py --checkpoint checkpoints/base/best.pt
    python demo/app.py --memotion-checkpoint checkpoints/memotion2/best.pt
    python demo/app.py --share
"""

import argparse
import copy
import json
import os
import sys
from functools import partial
from html import escape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# Globals loaded once at startup
_CONFIG = None
_DEVICE = None
_TOKENIZER = None
_IMAGE_TRANSFORM = None
_TASKS = {}


def _load_shared_runtime(config_path: str):
    """Load config, device, tokenizer, and transform shared by all task demos."""
    global _CONFIG, _DEVICE, _TOKENIZER, _IMAGE_TRANSFORM

    from transformers import BertTokenizer

    from ttt.data import get_image_transform
    from ttt.utils import get_device, load_config, set_seed

    _CONFIG = load_config(config_path)
    set_seed(_CONFIG.get("seed", 42))
    _DEVICE = get_device()

    if _DEVICE.type != "cuda":
        print("WARNING: No CUDA device found. Running on CPU (slow).")

    tokenizer_name = _CONFIG.get("text_encoder", "bert-base-uncased")
    try:
        _TOKENIZER = BertTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=True,
        )
    except Exception:
        _TOKENIZER = BertTokenizer.from_pretrained(tokenizer_name)
    _IMAGE_TRANSFORM = get_image_transform(_CONFIG.get("image_size", 224))


def _load_task_context(
    task_name: str,
    checkpoint_path: str,
    gate_checkpoint_path: str | None = None,
):
    """Load a task-specific model, adapter, profiler, and label decoder."""
    from ttt.data import build_memotion2_label_map
    from ttt.latency import LatencyProfiler
    from ttt.models import FullVQAModel
    from ttt.ttt_loop import TTTAdapter
    from ttt.utils import load_checkpoint

    config = copy.deepcopy(_CONFIG)
    if task_name == "memotion2":
        config["num_answers"] = config.get("memotion2_num_classes", 3)

    model = FullVQAModel(config)
    model.load_encoders(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            load_checkpoint(model, checkpoint_path)
            if gate_checkpoint_path and os.path.exists(gate_checkpoint_path):
                gate_ckpt = torch.load(gate_checkpoint_path, map_location="cpu", weights_only=True)
                model.gate.load_state_dict(gate_ckpt["gate"])
            print(f"[{task_name}] model loaded from {checkpoint_path}")
        except RuntimeError as e:
            print(f"[{task_name}] ERROR: Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    else:
        print(f"[{task_name}] WARNING: checkpoint not found at {checkpoint_path}; using random weights")

    model = model.to(_DEVICE)
    model.eval()

    ttt_adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)
    profiler = LatencyProfiler(model, ttt_adapter, _IMAGE_TRANSFORM, _TOKENIZER, config)

    if task_name == "memotion2":
        label_map = build_memotion2_label_map()
        label_decoder = {idx: label for label, idx in label_map.items()}
    else:
        vocab_path = os.path.join(config.get("data_dir", "data/"), "answer_vocab.json")
        label_decoder = None
        if os.path.exists(vocab_path):
            with open(vocab_path) as f:
                answer_vocab = json.load(f)
            label_decoder = {v: k for k, v in answer_vocab.items()}
        else:
            print(f"[{task_name}] WARNING: answer vocab not found at {vocab_path}; showing indices")

    return {
        "name": task_name,
        "config": config,
        "model": model,
        "ttt_adapter": ttt_adapter,
        "profiler": profiler,
        "label_decoder": label_decoder,
    }


def _decode_prediction(task_name: str, idx: int) -> str:
    """Convert classifier index to a user-facing label."""
    ctx = _TASKS.get(task_name)
    if not ctx:
        return "model_not_loaded"

    decoder = ctx["label_decoder"]
    if decoder and idx in decoder:
        return decoder[idx]

    if task_name == "memotion2":
        return f"sentiment_{idx}"
    return f"answer_{idx}"


def _shared_card_css():
    return """
    <style>
      .adattt-card {
        background: #18181b;
        border: 1px solid #2f2f35;
        border-radius: 10px;
        padding: 14px 16px;
        color: #f5f5f5;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      }
      .adattt-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }
      .adattt-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #d4d4d8;
      }
      .adattt-badge {
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
      }
      .adattt-value {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.25;
        word-break: break-word;
      }
      .adattt-subtitle {
        margin-top: 8px;
        color: #a1a1aa;
        font-size: 0.83rem;
      }
      .adattt-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
      }
      .adattt-metric {
        background: #111114;
        border: 1px solid #2a2a30;
        border-radius: 10px;
        padding: 10px 12px;
      }
      .adattt-metric-label {
        color: #9ca3af;
        font-size: 0.76rem;
        margin-bottom: 6px;
      }
      .adattt-metric-value {
        color: #f5f5f5;
        font-size: 1rem;
        font-weight: 700;
      }
      .adattt-latency-list {
        display: grid;
        gap: 12px;
      }
      .adattt-latency-row {
        display: grid;
        grid-template-columns: 120px 72px minmax(0, 1fr) 52px;
        align-items: center;
        gap: 10px;
      }
      .adattt-latency-label {
        color: #d4d4d8;
        font-size: 0.83rem;
      }
      .adattt-latency-value,
      .adattt-latency-pct {
        color: #a1a1aa;
        font-size: 0.78rem;
        text-align: right;
      }
      .adattt-bar {
        width: 100%;
        height: 9px;
        background: #24242a;
        border-radius: 999px;
        overflow: hidden;
      }
      .adattt-bar-fill {
        height: 100%;
        border-radius: 999px;
      }
    </style>
    """


def _render_message_card(title: str, message: str, tone: str = "neutral") -> str:
    return _render_answer_card(title, message, tone=tone, badge_text="INFO", subtitle="")


def _render_answer_card(
    title: str,
    value: str,
    tone: str = "neutral",
    badge_text: str | None = None,
    subtitle: str = "",
) -> str:
    palette = {
        "neutral": {
            "border": "#32323a",
            "glow": "rgba(255,255,255,0.04)",
            "text": "#f5f5f5",
            "badge_bg": "#27272a",
            "badge_text": "#d4d4d8",
        },
        "green": {
            "border": "#14532d",
            "glow": "rgba(34,197,94,0.14)",
            "text": "#dcfce7",
            "badge_bg": "#14532d",
            "badge_text": "#dcfce7",
        },
        "red": {
            "border": "#7f1d1d",
            "glow": "rgba(239,68,68,0.14)",
            "text": "#fee2e2",
            "badge_bg": "#7f1d1d",
            "badge_text": "#fee2e2",
        },
    }
    colors = palette[tone]
    badge_html = ""
    if badge_text:
        badge_html = (
            f"<span class='adattt-badge' style='background:{colors['badge_bg']};"
            f"color:{colors['badge_text']};'>{escape(badge_text)}</span>"
        )
    subtitle_html = ""
    if subtitle:
        subtitle_html = f"<div class='adattt-subtitle'>{escape(subtitle)}</div>"
    return f"""
    {_shared_card_css()}
    <div class="adattt-card" style="border-color:{colors['border']}; box-shadow: inset 0 0 0 1px {colors['glow']};">
      <div class="adattt-header">
        <div class="adattt-title">{escape(title)}</div>
        {badge_html}
      </div>
      <div class="adattt-value" style="color:{colors['text']};">{escape(value)}</div>
      {subtitle_html}
    </div>
    """


def _render_metric_cards(title: str, metrics: list[tuple[str, str]], badge_text: str, badge_color: str) -> str:
    metric_html = "".join(
        f"""
        <div class="adattt-metric">
          <div class="adattt-metric-label">{escape(label)}</div>
          <div class="adattt-metric-value">{escape(value)}</div>
        </div>
        """
        for label, value in metrics
    )
    return f"""
    {_shared_card_css()}
    <div class="adattt-card">
      <div class="adattt-header">
        <div class="adattt-title">{escape(title)}</div>
        <span class="adattt-badge" style="background:{badge_color}; color:white;">{escape(badge_text)}</span>
      </div>
      <div class="adattt-grid">
        {metric_html}
      </div>
    </div>
    """


def _render_latency_cards(budget, decision: str) -> str:
    total_ms = max(float(budget.total_ms), 1e-6)
    stages = [
        ("Preprocess", float(budget.image_preprocess_ms), "#6b7280"),
        ("ViT Encode", float(budget.vision_encode_ms), "#3b82f6"),
        ("BERT Encode", float(budget.text_encode_ms), "#22c55e"),
        ("Fusion+Predict", float(budget.fusion_predict_ms), "#f59e0b"),
    ]
    if budget.ttt_triggered:
        stages.append(("TTT Adapt", float(budget.ttt_adaptation_ms), "#ef4444"))

    rows = []
    for label, value, color in stages:
        width = min((value / total_ms) * 100.0, 100.0)
        pct = (value / total_ms) * 100.0
        rows.append(
            f"""
            <div class="adattt-latency-row">
              <div class="adattt-latency-label">{escape(label)}</div>
              <div class="adattt-latency-value">{value:.1f} ms</div>
              <div class="adattt-bar"><div class="adattt-bar-fill" style="width:{width:.1f}%; background:{color};"></div></div>
              <div class="adattt-latency-pct">{pct:.0f}%</div>
            </div>
            """
        )

    if decision == "SKIP":
        summary = "Gate skipped TTT cost"
        badge_color = "#166534"
    elif decision == "ADAPT":
        summary = "TTT adaptation was applied"
        badge_color = "#991b1b"
    else:
        summary = "Base path only"
        badge_color = "#3f3f46"
    stage_html = "".join(rows)
    return f"""
    {_shared_card_css()}
    <div class="adattt-card">
      <div class="adattt-header">
        <div class="adattt-title">Latency Breakdown</div>
        <span class="adattt-badge" style="background:{badge_color}; color:white;">{escape(decision)}</span>
      </div>
      <div class="adattt-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 14px;">
        <div class="adattt-metric">
          <div class="adattt-metric-label">Total Latency</div>
          <div class="adattt-metric-value">{budget.total_ms:.1f} ms</div>
        </div>
        <div class="adattt-metric">
          <div class="adattt-metric-label">TTT Share</div>
          <div class="adattt-metric-value">{budget.ttt_fraction * 100:.1f}%</div>
        </div>
      </div>
      <div class="adattt-latency-list">
        {stage_html}
      </div>
      <div class="adattt-subtitle">{escape(summary)}</div>
    </div>
    """


def predict(task_name: str, image_pil, text_input: str, num_ttt_steps, threshold, enable_gate):
    """Run prediction for a selected task context and return UI outputs."""
    if task_name not in _TASKS:
        message = _render_message_card("Model Status", "Model not loaded", tone="red")
        return message, message, message, message

    if image_pil is None:
        message = _render_message_card("Input Required", "Please upload an image")
        return message, message, message, message
    if not text_input or not text_input.strip():
        message = "Please enter a question"
        if task_name == "memotion2":
            message = "Please enter meme/OCR text"
        card = _render_message_card("Input Required", message)
        return card, card, card, card

    ctx = _TASKS[task_name]
    model = ctx["model"]
    ttt_adapter = ctx["ttt_adapter"]
    profiler = ctx["profiler"]
    config = ctx["config"]

    ttt_adapter.k_steps = int(num_ttt_steps)
    budget = profiler.profile_single(
        image_pil,
        text_input,
        threshold=threshold,
        k_steps=int(num_ttt_steps),
    )

    image_tensor = _IMAGE_TRANSFORM(image_pil).unsqueeze(0).to(_DEVICE)
    max_len = config.get("max_question_length", 20)
    encoding = _TOKENIZER(
        text_input,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(_DEVICE)
    attention_mask = encoding["attention_mask"].to(_DEVICE)

    with torch.no_grad():
        visual_tokens, text_tokens = model.encode(image_tensor, input_ids, attention_mask)
        z = model.fusion(visual_tokens, text_tokens, attention_mask)
        confidence = model.gate(z).item()
        base_logits = model.prediction_head(z)
        base_pred = base_logits.argmax(dim=-1).item()

    base_answer = _decode_prediction(task_name, base_pred)

    decision = "BASE"
    adapted_answer = base_answer
    if int(num_ttt_steps) > 0:
        skip = enable_gate and confidence > threshold
        if skip:
            decision = "SKIP"
        else:
            decision = "ADAPT"
            ttt_adapter.k_steps = int(num_ttt_steps)
            adapted_logits, _ = ttt_adapter.adapt_and_predict(
                image_tensor,
                visual_tokens,
                text_tokens,
                attention_mask,
            )
            adapted_pred = adapted_logits.argmax(dim=-1).item()
            adapted_answer = _decode_prediction(task_name, adapted_pred)
    elif not enable_gate:
        decision = "BASE"

    adapted_tone = "neutral"
    adapted_subtitle = "Base prediction shown"
    if decision == "SKIP":
        adapted_tone = "green"
        adapted_subtitle = "Gate was confident, so TTT was skipped"
    elif decision == "ADAPT":
        adapted_tone = "red"
        adapted_subtitle = "TTT ran on this sample before the final prediction"
    elif int(num_ttt_steps) == 0:
        adapted_subtitle = "TTT steps set to 0"
    elif not enable_gate:
        adapted_subtitle = "Gate disabled, so the adapted path is always used"

    base_card = _render_answer_card(
        title="Base Answer" if task_name == "vqa" else "Base Sentiment",
        value=base_answer,
        tone="neutral",
        badge_text="BASE",
        subtitle="Prediction before any test-time adaptation",
    )
    adapted_card = _render_answer_card(
        title="Adapted Answer" if task_name == "vqa" else "Adapted Sentiment",
        value=adapted_answer,
        tone=adapted_tone,
        badge_text=decision,
        subtitle=adapted_subtitle,
    )

    badge_color = "#166534" if decision == "SKIP" else "#991b1b" if decision == "ADAPT" else "#3f3f46"
    gate_metrics = [
        ("Confidence", f"{confidence:.3f}"),
        ("Threshold", f"{threshold:.2f}"),
        ("TTT Steps", str(int(num_ttt_steps))),
        ("Total Latency", f"{budget.total_ms:.1f} ms"),
        ("TTT Share", f"{budget.ttt_fraction * 100:.1f}%"),
        ("Task", task_name.upper()),
    ]
    gate_card = _render_metric_cards("Routing Summary", gate_metrics, decision, badge_color)
    latency_card = _render_latency_cards(budget, decision)
    return base_card, adapted_card, gate_card, latency_card


def _build_vqa_examples():
    """Use demo/examples if present."""
    examples = []
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    if os.path.exists(examples_dir):
        for img_file in sorted(os.listdir(examples_dir)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                examples.append([
                    os.path.join(examples_dir, img_file),
                    "What is in this image?",
                    1,
                    0.8,
                    True,
                ])
    return examples


def _build_memotion_examples(limit: int = 5):
    """Build Memotion2 examples from the validation annotations if available."""
    if _CONFIG is None:
        return []

    memo_dir = _CONFIG.get("memotion2_data_dir", os.path.join(_CONFIG.get("data_dir", "data/"), "memotion2"))
    ann_path = os.path.join(memo_dir, "val.json")
    image_dir = os.path.join(memo_dir, "images")
    if not os.path.exists(ann_path):
        return []

    with open(ann_path) as f:
        annotations = json.load(f)

    examples = []
    for item in annotations:
        image_name = item.get("image", "")
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue
        text = item.get("text", "").strip()
        if not text:
            continue
        examples.append([image_path, text, 1, 0.8, True])
        if len(examples) >= limit:
            break
    return examples


def _build_interface(
    task_name: str,
    title: str,
    description: str,
    text_label: str,
    placeholder: str,
    output_label: str,
    examples,
):
    import gradio as gr

    return gr.Interface(
        fn=partial(predict, task_name),
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(label=text_label, placeholder=placeholder),
            gr.Slider(0, 10, value=1, step=1, label="TTT Steps (K)"),
            gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Gate Threshold (τ)"),
            gr.Checkbox(value=True, label="Enable Gate"),
        ],
        outputs=[
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
        ],
        title=title,
        description=description,
        examples=examples if examples else None,
        flagging_mode="never",
    )


def main():
    parser = argparse.ArgumentParser(description="AdaTTT Gradio Demo")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base/best.pt")
    parser.add_argument("--gate-checkpoint", type=str, default=None)
    parser.add_argument(
        "--memotion-checkpoint",
        type=str,
        default="checkpoints/memotion2/best.pt",
        help="Checkpoint for the Memotion2 sentiment demo.",
    )
    parser.add_argument(
        "--memotion-gate-checkpoint",
        type=str,
        default=None,
        help="Optional refined gate checkpoint for the Memotion2 demo.",
    )
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import gradio as gr

    _load_shared_runtime(args.config)

    _TASKS["vqa"] = _load_task_context("vqa", args.checkpoint, args.gate_checkpoint)

    if args.memotion_checkpoint and os.path.exists(args.memotion_checkpoint):
        memotion_ctx = _load_task_context(
            "memotion2",
            args.memotion_checkpoint,
            args.memotion_gate_checkpoint,
        )
        if memotion_ctx is not None:
            _TASKS["memotion2"] = memotion_ctx
        else:
            print("[memotion2] checkpoint could not be loaded; Memotion2 tab will be omitted.")
    else:
        print(
            "[memotion2] checkpoint missing; Memotion2 tab will be omitted. "
            f"Expected: {args.memotion_checkpoint}"
        )

    vqa_demo = _build_interface(
        task_name="vqa",
        title="AdaTTT: Adaptive Test-Time Training for VQA",
        description=(
            "Upload an image and ask a question. The model uses a confidence gate "
            "to decide whether to apply test-time training (TTT) for harder questions. "
            "Adjust TTT steps and gate threshold to explore the accuracy-latency tradeoff."
        ),
        text_label="Question",
        placeholder="What is in this image?",
        output_label="Answer",
        examples=_build_vqa_examples(),
    )

    interfaces = [vqa_demo]
    tab_names = ["VQA"]

    if _TASKS.get("memotion2") is not None:
        memotion_demo = _build_interface(
            task_name="memotion2",
            title="AdaTTT: Adaptive Test-Time Training for Memotion2",
            description=(
                "Upload a meme image and provide the meme/OCR text. "
                "This demo uses the Memotion2-trained checkpoint to predict sentiment "
                "while exposing the same adaptive routing and latency controls."
            ),
            text_label="Meme / OCR Text",
            placeholder="Type the meme text or OCR text here",
            output_label="Sentiment",
            examples=_build_memotion_examples(),
        )
        interfaces.append(memotion_demo)
        tab_names.append("Memotion2")

    demo = gr.TabbedInterface(interfaces, tab_names, title="AdaTTT Interactive Demo")
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
# Add demo examples routing display
# Improve Gradio demo card UI layout
