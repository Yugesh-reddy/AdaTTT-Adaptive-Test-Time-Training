#!/usr/bin/env python3
"""
AdaTTT demo: answer a question about an image, or abstain when unsure.

Runs the Phase 1 CLIP model (frozen CLIP ViT-B/16 encoders, trained fusion;
67.50 official VQA soft on VQA-v2 val) and applies the confidence rule from
results/writeup/REPORT.md §2.10: the model answers when its gate score is below
a threshold and abstains otherwise. The thresholds in
results/writeup/abstention.json were fit on a held-out split for 90% and 80%
coverage, never on the reported evaluation split.

    pip install gradio
    python demo/app.py                       # http://127.0.0.1:7860
    python demo/app.py --coverage 80         # abstain more often
    python demo/app.py --checkpoint path/to/best.pt --share

The frozen CLIP encoders are downloaded on first run (~600 MB) and cached by
transformers. Answers come from the fixed 3,129-answer VQA-v2 vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttt.score_gate import ScoreWeights, signals_from_logits, weighted_score  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "config", "config.yaml")
DEFAULT_CHECKPOINT = os.path.join(ROOT, "checkpoints", "phase1_clip", "best.pt")
ABSTENTION = os.path.join(ROOT, "results", "writeup", "abstention.json")


@dataclass
class Runtime:
    """Everything one prediction needs."""

    config: Dict[str, Any]
    model: torch.nn.Module
    tokenizer: Any
    transform: Any
    answers: List[str]
    device: torch.device
    thresholds: Dict[str, float]


def load_thresholds(path: str = ABSTENTION) -> Dict[str, float]:
    """Coverage target -> gate-score threshold, as fit on the held-out split."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {k: float(v) for k, v in json.load(fh)["thresholds"].items()}


def load_runtime(config_path: str = DEFAULT_CONFIG,
                 checkpoint: str = DEFAULT_CHECKPOINT,
                 device: Optional[torch.device] = None) -> Runtime:
    """Load config, frozen encoders, the trained fusion, and the answer vocabulary."""
    from ttt.data import build_tokenizer, get_image_transform
    from ttt.models import FullVQAModel
    from ttt.utils import get_device, load_checkpoint, load_config

    from transformers.utils import logging as hf_logging

    # Loading CLIP's vision/text tower out of the full checkpoint logs every
    # key belonging to the other tower; that is expected, not a problem.
    hf_logging.set_verbosity_error()
    config = load_config(config_path)
    device = device or get_device()
    model = FullVQAModel(config)
    model.load_encoders(config)
    if os.path.exists(checkpoint):
        load_checkpoint(model, checkpoint)
    else:
        print(f"WARNING: no checkpoint at {checkpoint}; the fusion is untrained.")
    model.to(device).eval()

    vocab_path = os.path.join(ROOT, config.get("answer_vocab", "data/answer_vocab.json"))
    with open(vocab_path) as fh:
        vocab = json.load(fh)
    answers = [""] * len(vocab)
    for answer, idx in vocab.items():
        answers[int(idx)] = answer

    return Runtime(
        config=config,
        model=model,
        tokenizer=build_tokenizer(config),
        transform=get_image_transform(config.get("image_size", 224),
                                      backend=config.get("encoder_backend", "clip")),
        answers=answers,
        device=device,
        thresholds=load_thresholds(),
    )


@torch.no_grad()
def answer_question(rt: Runtime, image, question: str,
                    threshold: Optional[float] = None, top_k: int = 5) -> Dict[str, Any]:
    """Answer one question, or mark it as one the model should decline.

    `threshold` is a gate-score cutoff; None never abstains. The gate score is
    the same quantity the evaluation used: 0.5*normalized entropy +
    0.5*(1 - MaxProb), so higher means less confident.
    """
    pixel_values = rt.transform(image).unsqueeze(0).to(rt.device)
    encoded = rt.tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=rt.config.get("max_question_length", 20),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(rt.device)
    attention_mask = encoded["attention_mask"].to(rt.device)

    logits, _, _ = rt.model(pixel_values, input_ids, attention_mask)
    signals = signals_from_logits(logits)
    score = float(weighted_score(signals, ScoreWeights())[0].item())
    probs = signals["probs"][0]
    k = min(top_k, probs.numel())
    top = probs.topk(k)
    ranked: List[Tuple[str, float]] = [
        (rt.answers[int(i)], float(p)) for p, i in zip(top.values, top.indices)
    ]
    return {
        "answer": ranked[0][0],
        "confidence": ranked[0][1],
        "gate_score": score,
        "threshold": threshold,
        "abstained": threshold is not None and score > threshold,
        "top": ranked,
    }


def verdict_markdown(result: Dict[str, Any]) -> str:
    """The headline the demo shows above the answer distribution."""
    confidence = f"{100 * result['confidence']:.1f}%"
    if result["abstained"]:
        return (
            f"### I'd rather not answer\n"
            f"Its best guess is **{result['answer']}** ({confidence} confidence), but the "
            f"uncertainty score is {result['gate_score']:.3f}, above the "
            f"{result['threshold']:.3f} cutoff. On the evaluation set, declining questions "
            f"like this is what raises accuracy on the ones it does answer."
        )
    return f"### {result['answer']}\n{confidence} confidence (uncertainty {result['gate_score']:.3f})"


COVERAGE_CHOICES = {
    "Answer ~90% of questions": "90",
    "Answer ~80% of questions (more cautious)": "80",
    "Always answer": None,
}


def build_ui(rt: Runtime, default_coverage: Optional[str] = "90"):
    """The Gradio page. Imported lazily so the module stays importable without it."""
    import gradio as gr

    labels = list(COVERAGE_CHOICES)
    default_label = next(
        (lab for lab, key in COVERAGE_CHOICES.items() if key == default_coverage), labels[0]
    )

    def run(image, question, coverage_label):
        if image is None or not (question or "").strip():
            return "### Upload an image and ask a question", {}
        key = COVERAGE_CHOICES.get(coverage_label)
        threshold = rt.thresholds.get(key) if key else None
        result = answer_question(rt, image, question, threshold)
        return verdict_markdown(result), {a: p for a, p in result["top"]}

    with gr.Blocks(title="AdaTTT — answer or abstain") as page:
        gr.Markdown(
            "# AdaTTT\n"
            "Frozen CLIP ViT-B/16 encoders with a trained fusion stack: **67.50** official "
            "VQA soft on VQA-v2 val. It answers from a fixed 3,129-answer vocabulary, and "
            "declines the questions it is least sure about.\n\n"
            "Thresholds were fit on a held-out split (`results/writeup/abstention.json`). "
            "Declining the least confident 10% raises accuracy on the rest by ~5.3 points; "
            "declining 20% raises it by ~10.4."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                question = gr.Textbox(label="Question", placeholder="What color is the car?")
                coverage = gr.Radio(labels, value=default_label, label="How often should it answer?")
                ask = gr.Button("Ask", variant="primary")
            with gr.Column():
                verdict = gr.Markdown("### Upload an image and ask a question")
                distribution = gr.Label(num_top_classes=5, label="Top answers")
        ask.click(run, [image, question, coverage], [verdict, distribution])
        question.submit(run, [image, question, coverage], [verdict, distribution])
    return page


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--coverage", choices=["90", "80"], default="90",
                        help="target fraction of questions answered (default: 90)")
    parser.add_argument("--no-abstain", action="store_true", help="always answer")
    parser.add_argument("--share", action="store_true", help="public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    rt = load_runtime(args.config, args.checkpoint)
    if not rt.thresholds:
        print(f"WARNING: {ABSTENTION} missing; run scripts/abstention.py to enable abstention.")
    page = build_ui(rt, default_coverage=None if args.no_abstain else args.coverage)
    page.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
