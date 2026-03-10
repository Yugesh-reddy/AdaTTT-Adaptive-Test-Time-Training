"""
Tests for evaluation metrics: VQA accuracy, Pareto frontier, gate stats.
"""

import pytest

from ttt.metrics import (
    vqa_accuracy,
    accuracy_by_question_type,
    pareto_frontier,
    compute_gate_statistics,
    mcnemar_test,
)


class TestVQAAccuracy:
    def test_perfect_accuracy(self):
        """All correct → accuracy = 1.0"""
        preds = [0, 1, 2, 3, 4]
        gts = [0, 1, 2, 3, 4]
        assert vqa_accuracy(preds, gts) == 1.0

    def test_zero_accuracy(self):
        """All wrong → accuracy = 0.0"""
        preds = [1, 2, 3, 4, 0]
        gts = [0, 1, 2, 3, 4]
        assert vqa_accuracy(preds, gts) == 0.0

    def test_partial_accuracy(self):
        """3/5 correct → accuracy = 0.6"""
        preds = [0, 1, 2, 99, 99]
        gts = [0, 1, 2, 3, 4]
        assert abs(vqa_accuracy(preds, gts) - 0.6) < 1e-6

    def test_empty_predictions(self):
        """Empty input → accuracy = 0.0"""
        assert vqa_accuracy([], []) == 0.0


class TestAccuracyByType:
    def test_breakdown(self):
        """Per-type accuracy is computed correctly."""
        preds = [0, 0, 1, 1, 2]
        gts = [0, 1, 1, 1, 0]
        types = ["yes/no", "yes/no", "number", "number", "other"]

        result = accuracy_by_question_type(preds, gts, types)
        assert abs(result["yes/no"] - 0.5) < 1e-6     # 1/2 correct
        assert abs(result["number"] - 1.0) < 1e-6     # 2/2 correct
        assert abs(result["other"] - 0.0) < 1e-6      # 0/1 correct


class TestParetoFrontier:
    def test_pareto_filters_dominated(self):
        """Pareto frontier correctly filters dominated points."""
        results = [
            {"accuracy": 0.5, "avg_flops": 40.0, "config": "A"},
            {"accuracy": 0.7, "avg_flops": 50.0, "config": "B"},
            {"accuracy": 0.6, "avg_flops": 55.0, "config": "C"},  # dominated by B
            {"accuracy": 0.8, "avg_flops": 60.0, "config": "D"},
        ]

        pareto = pareto_frontier(results)

        # C is dominated by B (lower acc, higher flops)
        configs = [p["config"] for p in pareto]
        assert "C" not in configs
        assert "A" in configs
        assert "B" in configs
        assert "D" in configs

    def test_pareto_empty(self):
        """Empty input returns empty."""
        assert pareto_frontier([]) == []

    def test_pareto_single(self):
        """Single point is always Pareto-optimal."""
        results = [{"accuracy": 0.5, "avg_flops": 40.0, "config": "A"}]
        assert len(pareto_frontier(results)) == 1

    def test_pareto_sorted_by_flops(self):
        """Result is sorted by FLOPs ascending."""
        results = [
            {"accuracy": 0.9, "avg_flops": 60.0, "config": "C"},
            {"accuracy": 0.5, "avg_flops": 30.0, "config": "A"},
            {"accuracy": 0.7, "avg_flops": 45.0, "config": "B"},
        ]
        pareto = pareto_frontier(results)
        flops = [p["avg_flops"] for p in pareto]
        assert flops == sorted(flops)


class TestGateStatistics:
    def test_basic_stats(self):
        """Gate stats are computed correctly."""
        import torch

        routing_infos = [
            {
                "skip_count": 3,
                "adapt_count": 2,
                "confidences": torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2]),
                "skip_mask": torch.tensor([True, True, True, False, False]),
            }
        ]

        stats = compute_gate_statistics(routing_infos)
        assert stats["total_skip"] == 3
        assert stats["total_adapt"] == 2
        assert abs(stats["skip_rate"] - 0.6) < 1e-6


class TestMcNemarTest:
    def test_identical_models(self):
        """Identical predictions → not significant."""
        preds = [0, 1, 2, 3, 4]
        result = mcnemar_test(preds, preds, preds)
        assert result["b_base_correct_ttt_wrong"] == 0
        assert result["c_base_wrong_ttt_correct"] == 0
        assert result["p_value"] == 1.0
        assert not result["significant_at_005"]

    def test_asymmetric_improvement(self):
        """TTT strictly improves on base → should detect difference."""
        base = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ttt = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        gt = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        result = mcnemar_test(base, ttt, gt)
        assert result["b_base_correct_ttt_wrong"] == 0
        assert result["c_base_wrong_ttt_correct"] == 9
