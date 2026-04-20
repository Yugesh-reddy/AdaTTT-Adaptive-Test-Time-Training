"""
Tests for the latency profiler module.

Uses mock data — no HuggingFace downloads required.
"""

import pytest

from ttt.latency import LatencyBudget


class TestLatencyBudget:
    def test_fields_exist(self):
        """LatencyBudget has all expected fields."""
        budget = LatencyBudget()
        assert hasattr(budget, "image_preprocess_ms")
        assert hasattr(budget, "vision_encode_ms")
        assert hasattr(budget, "text_encode_ms")
        assert hasattr(budget, "fusion_predict_ms")
        assert hasattr(budget, "ttt_adaptation_ms")
        assert hasattr(budget, "total_ms")
        assert hasattr(budget, "ttt_triggered")
        assert hasattr(budget, "num_ttt_steps")

    def test_default_values(self):
        """Default values are zero/false."""
        budget = LatencyBudget()
        assert budget.image_preprocess_ms == 0.0
        assert budget.total_ms == 0.0
        assert budget.ttt_triggered is False
        assert budget.num_ttt_steps == 0

    def test_ttt_fraction_no_time(self):
        """ttt_fraction is 0 when total_ms is 0."""
        budget = LatencyBudget()
        assert budget.ttt_fraction == 0.0

    def test_ttt_fraction_calculation(self):
        """ttt_fraction correctly computes ratio."""
        budget = LatencyBudget(
            ttt_adaptation_ms=25.0,
            total_ms=100.0,
            ttt_triggered=True,
        )
        assert budget.ttt_fraction == pytest.approx(0.25)

    def test_ttt_fraction_no_ttt(self):
        """ttt_fraction is 0 when no TTT time."""
        budget = LatencyBudget(
            ttt_adaptation_ms=0.0,
            total_ms=50.0,
        )
        assert budget.ttt_fraction == 0.0

    def test_custom_values(self):
        """LatencyBudget accepts custom values."""
        budget = LatencyBudget(
            image_preprocess_ms=5.0,
            vision_encode_ms=20.0,
            text_encode_ms=15.0,
            fusion_predict_ms=3.0,
            ttt_adaptation_ms=30.0,
            total_ms=73.0,
            ttt_triggered=True,
            num_ttt_steps=3,
        )
        assert budget.num_ttt_steps == 3
        assert budget.ttt_triggered is True
        assert budget.ttt_fraction == pytest.approx(30.0 / 73.0)
