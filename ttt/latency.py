"""
Latency profiling for the Efficient TTT pipeline.

Profiles each stage of inference (encode, fuse, predict, TTT)
and reports P50/P95 latencies for performance analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class LatencyBudget:
    """Breakdown of latency for a single prediction."""

    image_preprocess_ms: float = 0.0
    vision_encode_ms: float = 0.0
    text_encode_ms: float = 0.0
    fusion_predict_ms: float = 0.0
    ttt_adaptation_ms: float = 0.0
    total_ms: float = 0.0
    ttt_triggered: bool = False
    num_ttt_steps: int = 0

    @property
    def ttt_fraction(self) -> float:
        """Fraction of total time spent on TTT adaptation."""
        if self.total_ms <= 0:
            return 0.0
        return self.ttt_adaptation_ms / self.total_ms


class LatencyProfiler:
    """Profiles per-stage latency of the AdaTTT pipeline.

    Usage:
        profiler = LatencyProfiler(model, ttt_adapter, image_transform, tokenizer, config)
        budget = profiler.profile_single(image_pil, question, threshold=0.8, k_steps=1)
        summary = profiler.profile_batch(dataset, n_samples=100)
    """

    def __init__(
        self,
        model: nn.Module,
        ttt_adapter: Any,
        image_transform: Any,
        tokenizer: Any,
        config: Dict[str, Any],
    ):
        self.model = model
        self.ttt_adapter = ttt_adapter
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

    def _sync_time(self) -> float:
        """Synchronize CUDA (if applicable) and return current time."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def profile_single(
        self,
        image_pil,
        question: str,
        threshold: float = 0.8,
        k_steps: int = 1,
    ) -> LatencyBudget:
        """Profile a single prediction through the full pipeline.

        Args:
            image_pil: PIL Image.
            question: Question string.
            threshold: Gate threshold for skip/adapt decision.
            k_steps: Number of TTT gradient steps.

        Returns:
            LatencyBudget with per-stage timings.
        """
        budget = LatencyBudget(num_ttt_steps=k_steps)
        max_len = self.config.get("max_question_length", 20)

        # Image preprocessing
        t0 = self._sync_time()
        image_tensor = self.image_transform(image_pil).unsqueeze(0).to(self.device)
        t1 = self._sync_time()
        budget.image_preprocess_ms = (t1 - t0) * 1000

        # Text tokenization + encoding
        encoding = self.tokenizer(
            question, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Vision encoding
        t2 = self._sync_time()
        with torch.no_grad():
            self.model.vit.eval()
            visual_tokens = self.model.vit(pixel_values=image_tensor).last_hidden_state
        t3 = self._sync_time()
        budget.vision_encode_ms = (t3 - t2) * 1000

        # Text encoding
        t4 = self._sync_time()
        with torch.no_grad():
            self.model.bert.eval()
            text_output = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_tokens = text_output.last_hidden_state
        t5 = self._sync_time()
        budget.text_encode_ms = (t5 - t4) * 1000

        # Fusion + gate + predict
        t6 = self._sync_time()
        z = self.model.fusion(visual_tokens, text_tokens, attention_mask)
        confidence = self.model.gate(z)
        skip = confidence.squeeze(-1).item() > threshold
        logits = self.model.prediction_head(z)
        t7 = self._sync_time()
        budget.fusion_predict_ms = (t7 - t6) * 1000

        # TTT adaptation (if not skipped and k_steps > 0)
        ttt_ms = 0.0
        if not skip and k_steps > 0:
            budget.ttt_triggered = True
            old_k = self.ttt_adapter.k_steps
            self.ttt_adapter.k_steps = k_steps

            t8 = self._sync_time()
            self.ttt_adapter.adapt_and_predict(
                image_tensor, visual_tokens, text_tokens, attention_mask,
            )
            t9 = self._sync_time()
            ttt_ms = (t9 - t8) * 1000
            self.ttt_adapter.k_steps = old_k

        budget.ttt_adaptation_ms = ttt_ms
        budget.total_ms = (
            budget.image_preprocess_ms
            + budget.vision_encode_ms
            + budget.text_encode_ms
            + budget.fusion_predict_ms
            + budget.ttt_adaptation_ms
        )
        return budget

    def profile_batch(
        self,
        dataset,
        n_samples: int = 100,
        warmup_runs: int = 5,
        threshold: float = 0.8,
        k_steps: int = 1,
    ) -> Dict[str, Any]:
        """Profile multiple samples and compute P50/P95 statistics.

        Args:
            dataset: Dataset returning dicts with 'image' (PIL) and 'question'.
            n_samples: Number of samples to profile.
            warmup_runs: Number of warmup runs before timing.
            threshold: Gate threshold.
            k_steps: TTT steps.

        Returns:
            Dict with P50/P95 for each stage.
        """
        import numpy as np
        from PIL import Image

        n_samples = min(n_samples, len(dataset))

        # Warmup
        for i in range(min(warmup_runs, n_samples)):
            sample = dataset[i]
            image = sample.get("image_pil") or sample.get("image")
            question = sample.get("question", "What is this?")
            if isinstance(image, torch.Tensor):
                # Convert back to PIL for profiling
                from torchvision.transforms.functional import to_pil_image
                # Undo normalization approximately
                image = to_pil_image(image * 0.5 + 0.5)
            self.profile_single(image, question, threshold, k_steps)

        # Collect budgets
        budgets: List[LatencyBudget] = []
        for i in range(n_samples):
            sample = dataset[i]
            image = sample.get("image_pil") or sample.get("image")
            question = sample.get("question", "What is this?")
            if isinstance(image, torch.Tensor):
                from torchvision.transforms.functional import to_pil_image
                image = to_pil_image(image * 0.5 + 0.5)
            budget = self.profile_single(image, question, threshold, k_steps)
            budgets.append(budget)

        # Aggregate
        stages = [
            "image_preprocess_ms", "vision_encode_ms", "text_encode_ms",
            "fusion_predict_ms", "ttt_adaptation_ms", "total_ms",
        ]
        summary: Dict[str, Any] = {
            "n_samples": n_samples,
            "k_steps": k_steps,
            "threshold": threshold,
        }
        for stage in stages:
            values = np.array([getattr(b, stage) for b in budgets])
            summary[stage] = {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        ttt_triggered = sum(1 for b in budgets if b.ttt_triggered)
        summary["ttt_trigger_rate"] = ttt_triggered / n_samples if n_samples > 0 else 0.0

        return summary
