"""
Tests for training utilities and logic.
"""

import math


class TestGradAccumStepCount:
    def test_accum_1_steps_every_batch(self):
        """accum=1 steps on every batch."""
        assert self._count(10, 1) == 10

    def test_accum_4_with_10_batches(self):
        """accum=4, 10 batches → steps at 4, 8, 10 = 3 steps."""
        assert self._count(10, 4) == 3

    def test_accum_4_with_12_batches(self):
        """accum=4, 12 batches → steps at 4, 8, 12 = 3 steps."""
        assert self._count(12, 4) == 3

    def test_accum_equals_batches(self):
        """accum == num_batches → exactly 1 step."""
        assert self._count(5, 5) == 1

    def test_scheduler_total_steps_matches_optimizer_steps_single_epoch(self):
        """Scheduler total steps should follow optimizer steps, not micro-batches."""
        num_batches = 10
        accum = 4
        assert self._scheduler_total_steps(num_batches, accum, epochs=1) == self._count(num_batches, accum)

    def test_scheduler_total_steps_scale_by_epochs(self):
        """Per-epoch optimizer step count should scale linearly across epochs."""
        num_batches = 10
        accum = 4
        epochs = 3
        assert self._scheduler_total_steps(num_batches, accum, epochs=epochs) == self._count(num_batches, accum) * epochs

    def test_scheduler_total_steps_do_not_use_raw_batch_count(self):
        """Regression check: accum>1 must shorten the scheduler horizon."""
        num_batches = 10
        accum = 4
        assert self._scheduler_total_steps(num_batches, accum, epochs=1) == 3
        assert self._scheduler_total_steps(num_batches, accum, epochs=1) != num_batches

    @staticmethod
    def _count(num_batches, accum):
        """Count optimizer steps using the same boundary logic as train_base.py."""
        return sum(1 for i in range(num_batches)
                   if (i + 1) % accum == 0 or (i + 1) == num_batches)

    @staticmethod
    def _scheduler_total_steps(num_batches, accum, epochs=1):
        """Mirror the scheduler horizon used in train_base.py."""
        steps_per_epoch = math.ceil(num_batches / accum)
        return steps_per_epoch * epochs
