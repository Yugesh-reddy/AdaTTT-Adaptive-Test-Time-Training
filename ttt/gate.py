"""
Adaptive routing logic for the Efficient TTT system.

Routes test samples through the base path (SKIP) or TTT adaptation (ADAPT)
based on the confidence gate's prediction.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AdaptiveRouter:
    """Routes samples through base model or TTT based on gate confidence.

    Usage:
        router = AdaptiveRouter(model, ttt_adapter, threshold=0.8)
        predictions, routing_info = router.predict(images, input_ids, attention_mask)

    Logic:
        1. Encode all samples (frozen ViT + BERT)
        2. Fuse → z for each
        3. Gate: confidence = gate(z)
        4. Split: high_conf → SKIP TTT, low_conf → ADAPT with TTT
        5. Recombine predictions in original batch order
    """

    # FLOPs estimates (approximate, for ViT-B/16 + BERT-base)
    ENCODE_FLOPS = 40.1e9    # ViT (~17.6G) + BERT (~22.5G)
    FUSION_FLOPS = 0.8e9     # Fusion forward pass
    PRED_FLOPS = 0.006e9     # Prediction head forward
    TTT_STEP_FLOPS = 3.2e9   # Fusion fwd + bwd + pred fwd + bwd per step
    SKIP_FLOPS = ENCODE_FLOPS + FUSION_FLOPS + PRED_FLOPS  # ~40.9 GFLOPs

    def __init__(
        self,
        model: nn.Module,
        ttt_adapter: Any,
        threshold: float = 0.8,
    ):
        """
        Args:
            model: FullVQAModel instance (with encoders loaded).
            ttt_adapter: TTTAdapter instance.
            threshold: Gate threshold τ. High confidence > τ → SKIP.
        """
        self.model = model
        self.ttt_adapter = ttt_adapter
        self.threshold = threshold

    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run adaptive prediction on a batch.

        Args:
            images: (B, 3, 224, 224)
            input_ids: (B, L) — BERT token IDs
            attention_mask: (B, L) — BERT attention mask

        Returns:
            predictions: (B, num_answers) — logits
            routing_info: dict with skip/adapt counts, confidences, mask
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode all samples
        visual_tokens, text_tokens = self.model.encode(images, input_ids, attention_mask)

        # 2. Fuse → z for ALL samples
        z = self.model.fusion(visual_tokens, text_tokens, attention_mask)

        # 3. Gate confidence
        confidence = self.model.gate(z)  # (B, 1)
        conf_values = confidence.squeeze(-1)  # (B,)
        skip_mask = conf_values > self.threshold  # True = SKIP

        skip_count = skip_mask.sum().item()
        adapt_count = B - skip_count

        # Initialize output
        all_logits = torch.zeros(B, self.model.prediction_head.classifier[-1].out_features, device=device)

        # 4a. Process SKIP samples (base prediction, no TTT)
        if skip_count > 0:
            skip_idx = skip_mask.nonzero(as_tuple=True)[0]
            skip_z = z[skip_idx]
            skip_logits = self.model.prediction_head(skip_z)
            all_logits[skip_idx] = skip_logits

        # 4b. Process ADAPT samples (TTT adaptation)
        if adapt_count > 0:
            adapt_idx = (~skip_mask).nonzero(as_tuple=True)[0]
            adapt_images = images[adapt_idx]
            adapt_visual = visual_tokens[adapt_idx]
            adapt_text = text_tokens[adapt_idx]
            adapt_mask = attention_mask[adapt_idx] if attention_mask is not None else None

            adapt_logits, ttt_loss = self.ttt_adapter.adapt_and_predict(
                adapt_images, adapt_visual, adapt_text, adapt_mask
            )
            all_logits[adapt_idx] = adapt_logits

        # 5. Build routing info
        routing_info = {
            "skip_count": int(skip_count),
            "adapt_count": int(adapt_count),
            "confidences": conf_values.detach().cpu(),
            "skip_mask": skip_mask.detach().cpu(),
        }

        return all_logits, routing_info

    def compute_flops(self, routing_info: Dict[str, Any], k_steps: int) -> float:
        """Compute average FLOPs per sample for this batch.

        Args:
            routing_info: Dict from predict() with skip/adapt counts.
            k_steps: Number of TTT steps used.

        Returns:
            Average GFLOPs per sample.
        """
        n_skip = routing_info["skip_count"]
        n_adapt = routing_info["adapt_count"]
        total = n_skip + n_adapt

        if total == 0:
            return 0.0

        adapt_flops = self.SKIP_FLOPS + k_steps * self.TTT_STEP_FLOPS
        total_flops = n_skip * self.SKIP_FLOPS + n_adapt * adapt_flops
        avg_flops = total_flops / total

        return avg_flops / 1e9  # Convert to GFLOPs
