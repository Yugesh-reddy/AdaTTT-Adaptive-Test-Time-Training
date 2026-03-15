"""
End-to-end integration test with a tiny synthetic dataset.

Exercises: model construction → fuse_and_predict → TTT adapt → gate route → predict.
No HuggingFace downloads required (uses random tensors as encoder outputs).
"""

import pytest
import torch

from ttt.models import FullVQAModel, FusionModule, ConfidenceGate, PredictionHead
from ttt.ttt_loop import TTTAdapter
from ttt.gate import AdaptiveRouter
from ttt.metrics import vqa_accuracy


@pytest.fixture
def config():
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.1,
        "prediction_hidden": 1024,
        "num_answers": 100,
        "gate_hidden": 256,
        "ttt_objectives": ["masked_patch"],
        "ttt_k_steps_sweep": [1],
        "ttt_lr": 1e-4,
        "ttt_mask_ratio": 0.25,
        "ttt_adapt_modules": ["fusion", "prediction_head"],
        "ttt_grad_clip": 1.0,
        "consistency_weight": 0.1,
        "mixup_alpha_range": [0.7, 1.0],
    }


@pytest.fixture
def synthetic_batch():
    """5-sample synthetic batch mimicking pre-encoded outputs."""
    B = 5
    return {
        "images": torch.randn(B, 3, 224, 224),
        "visual_tokens": torch.randn(B, 197, 768),
        "text_tokens": torch.randn(B, 20, 768),
        "attention_mask": torch.ones(B, 20, dtype=torch.bool),
        "answers": torch.randint(0, 100, (B,)),
    }


class TestEndToEndPipeline:
    def test_base_prediction(self, config, synthetic_batch):
        """Model produces predictions from pre-encoded tokens."""
        model = FullVQAModel(config)
        logits, z = model.fuse_and_predict(
            synthetic_batch["visual_tokens"],
            synthetic_batch["text_tokens"],
            synthetic_batch["attention_mask"],
        )
        assert logits.shape == (5, 100)
        preds = logits.argmax(dim=-1)
        assert preds.shape == (5,)

    def test_ttt_adaptation(self, config, synthetic_batch):
        """TTTAdapter adapts and predicts without crashing."""
        model = FullVQAModel(config)
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)

        logits, loss = adapter.adapt_and_predict(
            synthetic_batch["images"][:1],
            synthetic_batch["visual_tokens"][:1],
            synthetic_batch["text_tokens"][:1],
            synthetic_batch["attention_mask"][:1],
        )
        assert logits.shape == (1, 100)
        assert isinstance(loss, float)

    def test_ttt_restores_params(self, config, synthetic_batch):
        """TTT restores parameters after adaptation."""
        model = FullVQAModel(config)
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=2)

        # Snapshot params before
        before = {n: p.data.clone() for n, p in model.fusion.named_parameters()}

        adapter.adapt_and_predict(
            synthetic_batch["images"][:1],
            synthetic_batch["visual_tokens"][:1],
            synthetic_batch["text_tokens"][:1],
            synthetic_batch["attention_mask"][:1],
        )

        # Params should be restored
        for n, p in model.fusion.named_parameters():
            assert torch.allclose(p.data, before[n]), f"Parameter {n} was not restored!"

    def test_gate_routing(self, config, synthetic_batch):
        """ConfidenceGate produces valid routing decisions."""
        model = FullVQAModel(config)
        z = model.fusion(
            synthetic_batch["visual_tokens"],
            synthetic_batch["text_tokens"],
            synthetic_batch["attention_mask"],
        )
        confidence = model.gate(z)
        assert confidence.shape == (5, 1)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

        mask = model.gate.route(z, threshold=0.5)
        assert mask.shape == (5,)
        assert mask.dtype == torch.bool

    def test_full_pipeline_with_metrics(self, config, synthetic_batch):
        """Full pipeline: base predict → check accuracy metric."""
        model = FullVQAModel(config)
        logits, z = model.fuse_and_predict(
            synthetic_batch["visual_tokens"],
            synthetic_batch["text_tokens"],
            synthetic_batch["attention_mask"],
        )
        preds = logits.argmax(dim=-1).tolist()
        gt = synthetic_batch["answers"].tolist()

        acc = vqa_accuracy(preds, gt)
        assert 0.0 <= acc <= 1.0

    def test_k0_no_adaptation(self, config, synthetic_batch):
        """K=0 skips adaptation entirely."""
        model = FullVQAModel(config)
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=0)

        logits, loss = adapter.adapt_and_predict(
            synthetic_batch["images"][:1],
            synthetic_batch["visual_tokens"][:1],
            synthetic_batch["text_tokens"][:1],
            synthetic_batch["attention_mask"][:1],
        )
        assert logits.shape == (1, 100)
        assert loss == 0.0

    def test_multiple_samples_independent(self, config, synthetic_batch):
        """TTT on sample i doesn't affect sample j (params are restored)."""
        model = FullVQAModel(config)
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)

        # Run two samples with the same seed so random masking is identical
        torch.manual_seed(42)
        logits1, _ = adapter.adapt_and_predict(
            synthetic_batch["images"][:1],
            synthetic_batch["visual_tokens"][:1],
            synthetic_batch["text_tokens"][:1],
            synthetic_batch["attention_mask"][:1],
        )
        torch.manual_seed(42)
        logits2, _ = adapter.adapt_and_predict(
            synthetic_batch["images"][:1],
            synthetic_batch["visual_tokens"][:1],
            synthetic_batch["text_tokens"][:1],
            synthetic_batch["attention_mask"][:1],
        )
        # Same input + same seed should give same output (params were restored)
        assert torch.allclose(logits1, logits2, atol=1e-4)
