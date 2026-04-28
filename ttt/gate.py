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

    # FLOPs estimates (for ViT-B/16 + BERT-base, validated via scripts/05_measure_flops.py)
    # Split by tower: augmentation re-encodes the image only, so the two halves
    # have to be priced separately. Swapping BERT for CLIP's text tower drops
    # TEXT_ENCODE_FLOPS from ~22.5G to ~1G at 20-token questions — re-measure
    # both before reporting any v2 number.
    IMAGE_ENCODE_FLOPS = 17.6e9
    TEXT_ENCODE_FLOPS = 22.5e9
    ENCODE_FLOPS = IMAGE_ENCODE_FLOPS + TEXT_ENCODE_FLOPS  # 40.1G
    FUSION_FLOPS = 6.2e9     # 2-layer bidirectional cross-attention + FFN (measured)
    PRED_FLOPS = 0.008e9     # Prediction head forward (measured)
    TTT_STEP_FLOPS = 18.6e9  # Fusion fwd+bwd + pred fwd+bwd per step (~3x fwd)
    SKIP_FLOPS = ENCODE_FLOPS + FUSION_FLOPS + PRED_FLOPS  # ~46.3 GFLOPs

    # Consistency regularization overhead
    # Effective cost with consistency = SKIP_FLOPS + CONSISTENCY_OVERHEAD_FLOPS
    #   + k * (TTT_STEP_FLOPS + CONSISTENCY_PER_STEP_FLOPS)
    CONSISTENCY_OVERHEAD_FLOPS = 2 * 17.6e9   # 2x ViT fwd (one-time precompute)
    CONSISTENCY_PER_STEP_FLOPS = 2 * 6.2e9    # 2x extra fusion fwd per step (measured)

    def __init__(
        self,
        model: nn.Module,
        ttt_adapter: Any,
        threshold: float = 0.8,
        use_amp: bool = False,
        gate_type: str = "learned",
    ):
        """
        Args:
            model: FullVQAModel instance (with encoders loaded).
            ttt_adapter: TTTAdapter instance.
            threshold: Gate threshold τ. High confidence > τ → SKIP.
            use_amp: Enable mixed precision for encoding.
            gate_type: "learned" (supervised ConfidenceGate) or "entropy"
                (training-free EntropyGate). Entropy gate uses prediction
                distribution entropy — no gate training required.
        """
        self.model = model
        self.ttt_adapter = ttt_adapter
        self.threshold = threshold
        self.use_amp = use_amp
        self.gate_type = gate_type

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
        # 1. Encode all samples (with optional AMP)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            visual_tokens, text_tokens = self.model.encode(images, input_ids, attention_mask)

        return self._route_and_predict(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            images=images,
        )

    def predict_cached(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run adaptive prediction using precomputed encoder features.

        Use this when features were produced by gpu/precompute_features.py to
        skip redundant ViT+BERT forwards.

        Args:
            visual_tokens: (B, 197, 768)
            text_tokens: (B, L, 768)
            attention_mask: (B, L)
            images: (B, 3, 224, 224) or None. Required when the TTT objective
                re-encodes images (e.g., rotation); not needed for masked_patch.
        """
        return self._route_and_predict(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            images=images,
        )

    def _route_and_predict(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = visual_tokens.shape[0]
        device = visual_tokens.device

        # 2. Fuse → z for ALL samples
        z = self.model.fusion(visual_tokens.float(), text_tokens.float(), attention_mask)

        # 3. Gate confidence — route via learned gate or entropy gate
        if self.gate_type == "entropy":
            # Entropy gate: needs logits first, then decides
            logits = self.model.prediction_head(z)
            confidence = self.model.entropy_gate(logits)  # (B, 1)
        else:
            # Learned gate: uses fused representation directly
            confidence = self.model.gate(z)  # (B, 1)
            logits = self.model.prediction_head(z)

        conf_values = confidence.squeeze(-1)  # (B,)
        skip_mask = conf_values > self.threshold  # True = SKIP

        skip_count = skip_mask.sum().item()
        adapt_count = B - skip_count

        # Initialize output
        all_logits = torch.zeros(B, self.model.prediction_head.classifier[-1].out_features, device=device)

        # 4a. Process SKIP samples (base prediction, no TTT)
        if skip_count > 0:
            skip_idx = skip_mask.nonzero(as_tuple=True)[0]
            if self.gate_type == "entropy":
                # Logits already computed above
                all_logits[skip_idx] = logits[skip_idx]
            else:
                skip_z = z[skip_idx]
                skip_logits = self.model.prediction_head(skip_z)
                all_logits[skip_idx] = skip_logits

        # 4b. Process ADAPT samples (TTT adaptation)
        if adapt_count > 0:
            adapt_idx = (~skip_mask).nonzero(as_tuple=True)[0]
            # Run TTT independently per sample to preserve per-sample adaptation semantics.
            for idx in adapt_idx.tolist():
                sample_images = images[idx:idx + 1] if images is not None else None
                sample_visual = visual_tokens[idx:idx + 1]
                sample_text = text_tokens[idx:idx + 1]
                sample_mask = (
                    attention_mask[idx:idx + 1] if attention_mask is not None else None
                )
                sample_logits, _ = self.ttt_adapter.adapt_and_predict(
                    sample_images, sample_visual, sample_text, sample_mask
                )
                all_logits[idx] = sample_logits.squeeze(0)

        # 5. Build routing info
        routing_info = {
            "skip_count": int(skip_count),
            "adapt_count": int(adapt_count),
            "confidences": conf_values.detach().cpu(),
            "skip_mask": skip_mask.detach().cpu(),
        }

        return all_logits, routing_info

    # Per-backend encoder costs. CLIP's text tower is 12 layers at width 512
    # over <=77 tokens, so it costs roughly 1 GFLOP at 20-token questions
    # against BERT-base's ~22.5. That halves the base forward and therefore
    # RAISES the adapted:base ratio, because augmentation cost is unchanged.
    # These are estimates — re-derive with scripts/05_measure_flops.py before
    # reporting any number.
    ENCODER_COSTS = {
        "vit_bert": (17.6e9, 22.5e9),
        "clip": (17.6e9, 1.0e9),
    }

    @classmethod
    def configure_for_backend(cls, backend: str) -> None:
        """Repoint the FLOPs constants at an encoder backend.

        Raises:
            ValueError: If `backend` is unknown.
        """
        if backend not in cls.ENCODER_COSTS:
            valid = ", ".join(sorted(cls.ENCODER_COSTS))
            raise ValueError(f"Unknown encoder backend '{backend}'. Valid: {valid}")

        cls.IMAGE_ENCODE_FLOPS, cls.TEXT_ENCODE_FLOPS = cls.ENCODER_COSTS[backend]
        cls.ENCODE_FLOPS = cls.IMAGE_ENCODE_FLOPS + cls.TEXT_ENCODE_FLOPS
        cls.SKIP_FLOPS = cls.ENCODE_FLOPS + cls.FUSION_FLOPS + cls.PRED_FLOPS

    @staticmethod
    def sample_flops(
        adapted: bool,
        n_aug: int = 0,
        k_steps: int = 1,
        layernorm_only: bool = False,
    ) -> float:
        """FLOPs for ONE request on the deployment path.

        Priced from what a served request actually executes, never from what a
        benchmark happens to read out of a feature cache. In v1 the cascade cost
        model charged escalated requests only the expensive tier and omitted the
        cheap-tier encode every request had already run, which made the savings
        look real at any escalation rate. The v2 shape of that mistake is
        treating augmented views as free because they were precomputed offline.

        Every request pays the base forward. An adapted request additionally
        pays, for each augmented view, an image-tower encode (once; the tower is
        frozen) plus, on every step, a fusion and head forward AND backward
        through that view — the loss H(mean over views) depends on all of them.
        The prediction is then re-run on the original with the updated weights.
        The text tower is encoded once and reused, since AugMix perturbs pixels
        only.

        Args:
            adapted: Whether the gate fired for this sample.
            n_aug: Augmented views used by the adaptation objective (MEMO uses 4).
            k_steps: TTT gradient steps.
            layernorm_only: Accepted and deliberately ignored in the arithmetic.
                Restricting updates to LayerNorm affines shrinks optimizer state
                and the fitted hypothesis space; gradients still traverse the
                whole stack, so the backward costs the same. The 2,886x parameter
                ratio is not a compute saving and must not be reported as one.

        Returns:
            FLOPs for this request.
        """
        cost = AdaptiveRouter.SKIP_FLOPS
        if not adapted:
            return cost

        if n_aug == 0:
            # Entropy on the original view (TENT-style). Step 1 reuses the base
            # forward; each step adds a backward plus the next forward, the last
            # of which is the post-update prediction.
            return cost + k_steps * AdaptiveRouter.TTT_STEP_FLOPS

        # MEMO-style. An earlier version charged one backward per step whatever
        # n_aug was: exact for one view, but MEMO4 at K=1 was billed 138.6G of
        # the 175.8G it runs (three view backwards missing).
        cost += n_aug * AdaptiveRouter.IMAGE_ENCODE_FLOPS
        cost += k_steps * n_aug * AdaptiveRouter.TTT_STEP_FLOPS
        cost += AdaptiveRouter.FUSION_FLOPS + AdaptiveRouter.PRED_FLOPS
        return cost

    def compute_flops(
        self,
        routing_info: Dict[str, Any],
        k_steps: int,
        use_consistency: bool = False,
    ) -> float:
        """Compute average FLOPs per sample for this batch.

        Args:
            routing_info: Dict from predict() with skip/adapt counts.
            k_steps: Number of TTT steps used.
            use_consistency: Whether consistency regularization was used
                (adds ViT and fusion overhead to adapt cost).

        Returns:
            Average GFLOPs per sample.
        """
        n_skip = routing_info["skip_count"]
        n_adapt = routing_info["adapt_count"]
        total = n_skip + n_adapt

        if total == 0:
            return 0.0

        adapt_flops = self.SKIP_FLOPS + k_steps * self.TTT_STEP_FLOPS
        if use_consistency:
            adapt_flops += self.CONSISTENCY_OVERHEAD_FLOPS
            adapt_flops += k_steps * self.CONSISTENCY_PER_STEP_FLOPS
        total_flops = n_skip * self.SKIP_FLOPS + n_adapt * adapt_flops

        return (total_flops / total) / 1e9  # Convert to GFLOPs
