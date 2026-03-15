"""
Tests for the graceful degradation (fallback) module.

Uses mock objects — no HuggingFace downloads required.
"""

import pytest
import torch

from ttt.fallback import FallbackLevel, FallbackResult, GracefulPredictor


class TestFallbackLevel:
    def test_ordering(self):
        """Fallback levels have correct ordering."""
        assert FallbackLevel.FULL_ADATTT < FallbackLevel.BASE_ONLY
        assert FallbackLevel.BASE_ONLY < FallbackLevel.REDUCED_RESOLUTION
        assert FallbackLevel.REDUCED_RESOLUTION < FallbackLevel.ERROR

    def test_values(self):
        assert FallbackLevel.FULL_ADATTT == 0
        assert FallbackLevel.BASE_ONLY == 1
        assert FallbackLevel.REDUCED_RESOLUTION == 2
        assert FallbackLevel.ERROR == 3


class TestFallbackResult:
    def test_fields_exist(self):
        """FallbackResult has all expected fields."""
        result = FallbackResult(
            answer_idx=42,
            logits=torch.randn(1, 100),
            level=FallbackLevel.FULL_ADATTT,
            reason="success",
            latency_ms=10.5,
        )
        assert result.answer_idx == 42
        assert result.logits is not None
        assert result.level == FallbackLevel.FULL_ADATTT
        assert result.reason == "success"
        assert result.latency_ms == 10.5

    def test_error_result(self):
        """Error result has answer_idx=-1 and no logits."""
        result = FallbackResult(
            answer_idx=-1,
            logits=None,
            level=FallbackLevel.ERROR,
            reason="all failed",
            latency_ms=100.0,
        )
        assert result.answer_idx == -1
        assert result.logits is None
        assert result.level == FallbackLevel.ERROR


class TestGracefulPredictor:
    @pytest.fixture
    def mock_components(self):
        """Create mock model, adapter, and router for testing."""
        from ttt.models import FullVQAModel

        config = {
            "fusion_dim": 768,
            "fusion_heads": 12,
            "fusion_layers": 2,
            "fusion_dropout": 0.1,
            "prediction_hidden": 1024,
            "num_answers": 100,
            "gate_hidden": 256,
            "fallback_ttt_timeout_ms": 500,
            "fallback_reduced_resolution": 160,
        }
        model = FullVQAModel(config)
        return model, None, None, config

    def test_construction(self, mock_components):
        """GracefulPredictor can be constructed."""
        model, adapter, router, config = mock_components

        class MockRouter:
            def predict(self, images, input_ids, attention_mask):
                B = images.shape[0]
                return torch.randn(B, 100), {"skip_count": B, "adapt_count": 0}

        predictor = GracefulPredictor(model, adapter, MockRouter(), config)
        assert predictor.ttt_timeout_ms == 500
        assert predictor.reduced_resolution == 160

    def test_level0_normal_prediction(self, mock_components):
        """Level 0 returns FULL_ADATTT on success."""
        model, adapter, router, config = mock_components

        class MockRouter:
            def predict(self, images, input_ids, attention_mask):
                B = images.shape[0]
                logits = torch.randn(B, 100)
                return logits, {"skip_count": B, "adapt_count": 0}

        predictor = GracefulPredictor(model, adapter, MockRouter(), config)
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.ones(1, 20, dtype=torch.long)
        attention_mask = torch.ones(1, 20, dtype=torch.long)

        result = predictor.predict_with_fallback(images, input_ids, attention_mask)
        assert result.level == FallbackLevel.FULL_ADATTT
        assert result.answer_idx >= 0
        assert result.logits is not None
        assert result.latency_ms > 0

    def test_level3_on_router_error(self, mock_components):
        """Returns ERROR level when router raises non-OOM and base also fails."""
        model, adapter, router, config = mock_components
        # Remove encoders so base prediction also fails
        model.vit = None
        model.bert = None

        class FailRouter:
            def predict(self, images, input_ids, attention_mask):
                raise RuntimeError("test error")

        predictor = GracefulPredictor(model, adapter, FailRouter(), config)
        images = torch.randn(1, 3, 224, 224)
        input_ids = torch.ones(1, 20, dtype=torch.long)
        attention_mask = torch.ones(1, 20, dtype=torch.long)

        result = predictor.predict_with_fallback(images, input_ids, attention_mask)
        assert result.level == FallbackLevel.ERROR
        assert result.answer_idx == -1
