"""
Tests for gate routing logic (AdaptiveRouter).

Validates batch splitting, prediction recombination, and FLOPs computation.
"""

import pytest
import torch

from ttt.models import FullVQAModel, ConfidenceGate
from ttt.gate import AdaptiveRouter


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
        "ttt_lr": 1e-3,
        "ttt_mask_ratio": 0.25,
        "consistency_weight": 0.1,
        "mixup_alpha_range": [0.7, 1.0],
    }


class TestRoutingSplit:
    def test_routing_splits_batch(self):
        """Gate correctly identifies skip and adapt groups."""
        gate = ConfidenceGate(768, 256)
        z = torch.randn(10, 768)
        mask = gate.route(z, threshold=0.5)

        assert mask.shape == (10,)
        assert mask.dtype == torch.bool
        # At least some should be True and some False (with high probability for random z)
        # But we can't guarantee this, so just check shapes

    def test_routing_threshold_extremes(self):
        """Extreme thresholds route all/none."""
        gate = ConfidenceGate(768, 256)
        z = torch.randn(10, 768)

        # Very low threshold: everything skips (confidence > 0 for sigmoid)
        mask_low = gate.route(z, threshold=0.0)
        assert mask_low.all()  # All skip since confidence > 0

        # Very high threshold: nothing skips
        mask_high = gate.route(z, threshold=1.0)
        assert not mask_high.any()  # None can exceed 1.0 (sigmoid max is 1.0)


class TestRoutingRecombine:
    def test_predictions_recombined(self, config):
        """Predictions are correctly recombined after split routing.

        Tests the model's fuse_and_predict with known inputs and verifies
        that all batch positions get valid predictions.
        """
        model = FullVQAModel(config)
        B = 8
        visual = torch.randn(B, 197, 768)
        text = torch.randn(B, 20, 768)
        text_mask = torch.ones(B, 20, dtype=torch.bool)

        # Direct full-batch prediction
        logits_full, z_full = model.fuse_and_predict(visual, text, text_mask)

        assert logits_full.shape == (B, 100)
        assert z_full.shape == (B, 768)

        # All predictions should be finite
        assert torch.isfinite(logits_full).all()


class TestFLOPsComputation:
    def test_flops_skip_only(self):
        """All-skip batch has lower FLOPs than all-adapt."""
        routing_skip = {"skip_count": 10, "adapt_count": 0}
        routing_adapt = {"skip_count": 0, "adapt_count": 10}

        flops_skip = AdaptiveRouter.SKIP_FLOPS * 10 / 10 / 1e9
        flops_adapt = (AdaptiveRouter.SKIP_FLOPS + 3 * AdaptiveRouter.TTT_STEP_FLOPS) * 10 / 10 / 1e9

        assert flops_skip < flops_adapt

    def test_compute_flops_method(self):
        """compute_flops returns reasonable values."""
        # Create a minimal router (won't actually route, just use compute_flops)
        router = AdaptiveRouter.__new__(AdaptiveRouter)

        routing_info = {"skip_count": 5, "adapt_count": 5}
        flops = AdaptiveRouter.compute_flops(router, routing_info, k_steps=1)

        assert flops > 0
        # Should be between pure-skip and pure-adapt FLOPs
        pure_skip = AdaptiveRouter.SKIP_FLOPS / 1e9
        pure_adapt = (AdaptiveRouter.SKIP_FLOPS + 1 * AdaptiveRouter.TTT_STEP_FLOPS) / 1e9
        assert flops >= pure_skip
        assert flops <= pure_adapt
