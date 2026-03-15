"""
Graceful degradation for the AdaTTT inference pipeline.

Provides a 4-level fallback chain:
    Level 0: Full AdaTTT (encode + gate + TTT if needed)
    Level 1: Base only (skip TTT due to timeout)
    Level 2: Reduced resolution (after OOM, retry at lower res)
    Level 3: Error (catch-all, return error info)
"""

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("efficient_ttt")


class FallbackLevel(IntEnum):
    """Fallback degradation levels, from best to worst."""

    FULL_ADATTT = 0
    BASE_ONLY = 1
    REDUCED_RESOLUTION = 2
    ERROR = 3


@dataclass
class FallbackResult:
    """Result from graceful prediction."""

    answer_idx: int
    logits: Optional[torch.Tensor]
    level: FallbackLevel
    reason: str
    latency_ms: float


class GracefulPredictor:
    """Wraps the AdaTTT pipeline with graceful degradation.

    Falls back through increasingly cheaper prediction modes
    when the full pipeline fails or exceeds time budgets.

    Usage:
        predictor = GracefulPredictor(model, ttt_adapter, router, config)
        result = predictor.predict_with_fallback(images, input_ids, attention_mask)
    """

    def __init__(
        self,
        model: nn.Module,
        ttt_adapter: Any,
        router: Any,
        config: Dict[str, Any],
    ):
        self.model = model
        self.ttt_adapter = ttt_adapter
        self.router = router
        self.config = config
        self.device = next(model.parameters()).device
        self.ttt_timeout_ms = config.get("fallback_ttt_timeout_ms", 500)
        self.reduced_resolution = config.get("fallback_reduced_resolution", 160)

    def _log_fallback(self, level: FallbackLevel, reason: str, latency_ms: float):
        logger.warning(
            f"Fallback to level {level.name}: {reason} (latency={latency_ms:.1f}ms)"
        )

    def predict_with_fallback(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ttt_timeout_ms: Optional[float] = None,
    ) -> FallbackResult:
        """Run prediction with graceful degradation.

        Args:
            images: (B, 3, 224, 224)
            input_ids: (B, L)
            attention_mask: (B, L)
            ttt_timeout_ms: Override timeout for TTT adaptation.

        Returns:
            FallbackResult with prediction and degradation info.
        """
        timeout = ttt_timeout_ms if ttt_timeout_ms is not None else self.ttt_timeout_ms
        t0 = time.perf_counter()

        # Level 0: Full AdaTTT
        try:
            logits, routing_info = self.router.predict(
                images, input_ids, attention_mask,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Check if we exceeded the timeout — if so, note it but still return
            if elapsed_ms > timeout and routing_info["adapt_count"] > 0:
                self._log_fallback(
                    FallbackLevel.FULL_ADATTT,
                    f"completed but exceeded timeout ({elapsed_ms:.0f}ms > {timeout}ms)",
                    elapsed_ms,
                )

            answer_idx = logits.argmax(dim=-1)[0].item()
            return FallbackResult(
                answer_idx=answer_idx,
                logits=logits.detach(),
                level=FallbackLevel.FULL_ADATTT,
                reason="success",
                latency_ms=elapsed_ms,
            )

        except (RuntimeError, AssertionError) as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                self._log_fallback(FallbackLevel.BASE_ONLY, f"CUDA OOM: {e}", elapsed_ms)
                torch.cuda.empty_cache()
            else:
                self._log_fallback(FallbackLevel.BASE_ONLY, f"{type(e).__name__}: {e}", elapsed_ms)

        # Level 1: Base only (no TTT)
        try:
            t1 = time.perf_counter()
            with torch.no_grad():
                visual_tokens, text_tokens = self.model.encode(
                    images, input_ids, attention_mask,
                )
                logits, z = self.model.fuse_and_predict(
                    visual_tokens, text_tokens, attention_mask,
                )
            elapsed_ms = (time.perf_counter() - t1) * 1000
            answer_idx = logits.argmax(dim=-1)[0].item()
            return FallbackResult(
                answer_idx=answer_idx,
                logits=logits.detach(),
                level=FallbackLevel.BASE_ONLY,
                reason="TTT skipped due to error; base prediction used",
                latency_ms=elapsed_ms,
            )

        except (RuntimeError, AssertionError) as e:
            elapsed_ms = (time.perf_counter() - t1) * 1000
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                self._log_fallback(FallbackLevel.REDUCED_RESOLUTION, f"OOM at base: {e}", elapsed_ms)
                torch.cuda.empty_cache()
            else:
                self._log_fallback(FallbackLevel.REDUCED_RESOLUTION, str(e), elapsed_ms)

        # Level 2: Reduced resolution
        try:
            t2 = time.perf_counter()
            reduced_size = self.reduced_resolution
            reduced_images = torch.nn.functional.interpolate(
                images, size=(reduced_size, reduced_size), mode="bilinear", align_corners=False,
            )
            # Resize back to 224 for ViT compatibility
            reduced_images = torch.nn.functional.interpolate(
                reduced_images, size=(224, 224), mode="bilinear", align_corners=False,
            )
            with torch.no_grad():
                visual_tokens, text_tokens = self.model.encode(
                    reduced_images, input_ids, attention_mask,
                )
                logits, z = self.model.fuse_and_predict(
                    visual_tokens, text_tokens, attention_mask,
                )
            elapsed_ms = (time.perf_counter() - t2) * 1000
            answer_idx = logits.argmax(dim=-1)[0].item()
            return FallbackResult(
                answer_idx=answer_idx,
                logits=logits.detach(),
                level=FallbackLevel.REDUCED_RESOLUTION,
                reason=f"reduced resolution ({reduced_size}px) due to OOM",
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t2) * 1000
            self._log_fallback(FallbackLevel.ERROR, str(e), elapsed_ms)

        # Level 3: Error
        total_ms = (time.perf_counter() - t0) * 1000
        return FallbackResult(
            answer_idx=-1,
            logits=None,
            level=FallbackLevel.ERROR,
            reason=f"all fallback levels exhausted",
            latency_ms=total_ms,
        )
