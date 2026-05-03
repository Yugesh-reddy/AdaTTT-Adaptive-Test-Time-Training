"""The demo's inference and abstention logic (demo/app.py), without Gradio."""

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from ttt.models import FullVQAModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("demo_app", os.path.join(ROOT, "demo", "app.py"))
app = importlib.util.module_from_spec(_spec)
sys.modules["demo_app"] = app  # @dataclass resolves its types through sys.modules
_spec.loader.exec_module(app)


class _StubEncoder:
    """Stands in for a frozen CLIP tower: same call shape, fixed output."""

    def __init__(self, length, dim, logits_bias=None):
        self.length, self.dim, self.logits_bias = length, dim, logits_bias

    def eval(self):
        return self

    def __call__(self, **kwargs):
        key = "pixel_values" if "pixel_values" in kwargs else "input_ids"
        batch = kwargs[key].shape[0]
        return SimpleNamespace(last_hidden_state=torch.zeros(batch, self.length, self.dim))


def _runtime(peaked: bool):
    config = {
        "fusion_dim": 768, "fusion_heads": 12, "fusion_layers": 1, "fusion_dropout": 0.0,
        "prediction_hidden": 32, "num_answers": 4, "gate_hidden": 16, "num_query_tokens": 1,
        "text_dim": 512, "encoder_backend": "clip", "max_question_length": 8,
    }
    model = FullVQAModel(config)
    model.vit = _StubEncoder(197, 768)
    model.bert = _StubEncoder(8, 512)
    model.eval()
    # Force a confident or an unsure distribution, whatever the stub features are.
    # Distinct values: equal logits leave topk's tie order platform-dependent.
    bias = torch.tensor([9.0, 3.0, 2.0, 1.0]) if peaked else torch.zeros(4)
    with torch.no_grad():
        final = model.prediction_head.classifier[-1]
        final.weight.zero_()
        final.bias.copy_(bias)

    def tokenizer(text, **kwargs):
        length = kwargs.get("max_length", 8)
        return {"input_ids": torch.ones(1, length, dtype=torch.long),
                "attention_mask": torch.ones(1, length, dtype=torch.long)}

    return app.Runtime(
        config=config, model=model, tokenizer=tokenizer,
        transform=lambda image: torch.zeros(3, 224, 224),
        answers=["yes", "no", "blue", "two"], device=torch.device("cpu"),
        thresholds={"90": 0.5, "80": 0.4},
    )


def test_confident_question_is_answered():
    result = app.answer_question(_runtime(peaked=True), object(), "what color?", threshold=0.5)
    assert result["abstained"] is False
    assert result["answer"] == "yes" and result["confidence"] > 0.9
    assert result["gate_score"] < 0.5
    assert [a for a, _ in result["top"]] == ["yes", "no", "blue", "two"]
    assert result["top"] == sorted(result["top"], key=lambda t: -t[1])


def test_unsure_question_is_declined_but_still_reports_its_guess():
    result = app.answer_question(_runtime(peaked=False), object(), "what color?", threshold=0.5)
    assert result["abstained"] is True
    assert result["gate_score"] > 0.5
    assert result["answer"] in {"yes", "no", "blue", "two"}
    assert "rather not answer" in app.verdict_markdown(result)


def test_no_threshold_always_answers():
    result = app.answer_question(_runtime(peaked=False), object(), "what color?", threshold=None)
    assert result["abstained"] is False
    assert app.verdict_markdown(result).startswith(f"### {result['answer']}")


def test_top_k_is_capped_by_the_vocabulary():
    result = app.answer_question(_runtime(peaked=True), object(), "q", threshold=None, top_k=10)
    assert len(result["top"]) == 4


def test_ui_coverage_options_match_the_shipped_thresholds():
    shipped = app.load_thresholds()
    assert shipped, "results/writeup/abstention.json should ship with the repo"
    for key in app.COVERAGE_CHOICES.values():
        assert key is None or key in shipped


def test_missing_abstention_file_disables_abstention(tmp_path):
    assert app.load_thresholds(str(tmp_path / "nope.json")) == {}
