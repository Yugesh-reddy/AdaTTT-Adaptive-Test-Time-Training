"""
Tests for model components: FusionModule, ConfidenceGate, PredictionHead, FullVQAModel.

All tests use random tensors — no HuggingFace model downloads required.
"""

import pytest
import torch
import torch.nn as nn

from ttt.models import (
    CrossAttention,
    FusionLayer,
    FusionModule,
    ConfidenceGate,
    PredictionHead,
    MaskedPatchProjection,
    RotationHead,
    FullVQAModel,
)


@pytest.fixture
def config():
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.1,
        "prediction_hidden": 1024,
        "num_answers": 3129,
        "gate_hidden": 256,
    }


@pytest.fixture
def batch_tensors():
    """Random batch tensors mimicking encoded outputs."""
    B = 4
    return {
        "visual": torch.randn(B, 197, 768),
        "text": torch.randn(B, 20, 768),
        "text_mask": torch.ones(B, 20, dtype=torch.bool),
    }


class TestCrossAttention:
    def test_output_shape(self):
        attn = CrossAttention(dim=768, num_heads=12)
        q = torch.randn(2, 10, 768)
        k = torch.randn(2, 20, 768)
        v = torch.randn(2, 20, 768)
        out = attn(q, k, v)
        assert out.shape == (2, 10, 768)

    def test_with_mask(self):
        attn = CrossAttention(dim=768, num_heads=12)
        q = torch.randn(2, 10, 768)
        k = torch.randn(2, 20, 768)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 15:] = False  # Mask last 5 positions
        out = attn(q, k, k, key_mask=mask)
        assert out.shape == (2, 10, 768)


class TestFusionModule:
    def test_output_shape(self, batch_tensors):
        """FusionModule(visual(B,197,768), text(B,20,768)) → (B, 768)"""
        fusion = FusionModule(dim=768, num_heads=12, num_layers=2)
        z = fusion(batch_tensors["visual"], batch_tensors["text"], batch_tensors["text_mask"])
        assert z.shape == (4, 768)

    def test_return_sequence(self, batch_tensors):
        """return_sequence=True returns full visual sequence."""
        fusion = FusionModule(dim=768, num_heads=12, num_layers=2)
        seq = fusion(
            batch_tensors["visual"], batch_tensors["text"],
            batch_tensors["text_mask"], return_sequence=True
        )
        assert seq.shape == (4, 197, 768)

    def test_parameter_count(self):
        """Fusion module has substantial parameters (2 layers × dual-stream)."""
        fusion = FusionModule(dim=768, num_heads=12, num_layers=2)
        count = sum(p.numel() for p in fusion.parameters())
        # 2 layers × (2 cross-attn + 2 FFN with 4x expansion) ≈ 28M
        assert count > 20_000_000
        assert count < 35_000_000


class TestConfidenceGate:
    def test_output_range(self):
        """ConfidenceGate output is in [0, 1]"""
        gate = ConfidenceGate(768, 256)
        z = torch.randn(8, 768)
        confidence = gate(z)
        assert confidence.shape == (8, 1)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_route(self):
        """Gate route() returns correct boolean mask."""
        gate = ConfidenceGate(768, 256)
        z = torch.randn(10, 768)
        mask = gate.route(z, threshold=0.5)
        assert mask.shape == (10,)
        assert mask.dtype == torch.bool


class TestPredictionHead:
    def test_output_shape(self):
        """PredictionHead(B, 768) → (B, 3129)"""
        head = PredictionHead(768, 1024, 3129)
        z = torch.randn(4, 768)
        logits = head(z)
        assert logits.shape == (4, 3129)

    def test_custom_num_answers(self):
        """Custom num_answers works."""
        head = PredictionHead(768, 1024, 1000)
        z = torch.randn(2, 768)
        logits = head(z)
        assert logits.shape == (2, 1000)


class TestAuxiliaryHeads:
    def test_mask_proj(self):
        proj = MaskedPatchProjection(768)
        z = torch.randn(4, 768)
        out = proj(z)
        assert out.shape == (4, 768)

    def test_rotation_head(self):
        head = RotationHead(768, 4)
        z = torch.randn(4, 768)
        out = head(z)
        assert out.shape == (4, 4)


class TestFullVQAModel:
    def test_construction(self, config):
        """Model can be constructed without loading encoders."""
        model = FullVQAModel(config)
        assert model.vit is None
        assert model.bert is None
        assert model.fusion is not None
        assert model.gate is not None
        assert model.prediction_head is not None

    def test_fuse_and_predict(self, config, batch_tensors):
        """fuse_and_predict works with pre-encoded tokens."""
        model = FullVQAModel(config)
        logits, z = model.fuse_and_predict(
            batch_tensors["visual"], batch_tensors["text"], batch_tensors["text_mask"]
        )
        assert logits.shape == (4, 3129)
        assert z.shape == (4, 768)

    def test_get_ttt_params(self, config):
        """get_ttt_params returns fusion + prediction_head params."""
        model = FullVQAModel(config)
        ttt_params = model.get_ttt_params()
        # Should include fusion and prediction head params
        fusion_count = sum(p.numel() for p in model.fusion.parameters())
        pred_count = sum(p.numel() for p in model.prediction_head.parameters())
        ttt_count = sum(p.numel() for p in ttt_params)
        assert ttt_count == fusion_count + pred_count

    def test_get_ttt_params_named(self, config):
        """get_ttt_params_named returns properly prefixed names."""
        model = FullVQAModel(config)
        named_params = model.get_ttt_params_named()
        for name, param in named_params:
            assert name.startswith("fusion.") or name.startswith("prediction_head.")

    def test_frozen_encoders_assertion(self, config):
        """encode() raises if encoders not loaded."""
        model = FullVQAModel(config)
        images = torch.randn(2, 3, 224, 224)
        ids = torch.ones(2, 20, dtype=torch.long)
        mask = torch.ones(2, 20, dtype=torch.long)
        with pytest.raises(AssertionError, match="load_encoders"):
            model.encode(images, ids, mask)
