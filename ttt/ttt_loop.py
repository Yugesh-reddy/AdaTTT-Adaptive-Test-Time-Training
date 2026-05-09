"""
Test-Time Training (TTT) adaptation logic.

The core of the Efficient TTT system. Adapts model parameters per-sample
at test time using self-supervised objectives, then restores original params.

Supported objectives:
    - Masked patch prediction (MSE on masked visual tokens)
    - Rotation prediction (4-class classification)
    
Stabilization techniques:
    - Consistency regularization (symmetric KL on augmented views)
    - Mixup anchoring (interpolation toward anchor representation)
"""

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision imported lazily inside methods that need it


class TTTAdapter:
    """Performs test-time training on a single sample or mini-batch.

    At test time:
        1. Save a COPY of current θ_f and θ_d (the "anchor" state)
        2. For k in range(K):
            a. Compute self-supervised loss (masked patch OR rotation)
            b. Optionally add consistency loss
            c. Optionally apply mixup anchoring
            d. Backprop through θ_f and θ_d ONLY
        3. Run forward pass with adapted params → get prediction
        4. RESTORE θ_f and θ_d to anchor state

    The restore step is essential — TTT adapts per-sample, then resets.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        objective: Optional[str] = None,
        k_steps: Optional[int] = None,
        use_consistency: bool = False,
        use_mixup: bool = False,
    ):
        """
        Args:
            model: FullVQAModel instance.
            config: Configuration dictionary.
            objective: "masked_patch" or "rotation". Defaults to first in config.
            k_steps: Number of TTT gradient steps. Defaults to first in config.
            use_consistency: Enable consistency regularization.
            use_mixup: Enable mixup anchoring.
        """
        self.model = model
        self.config = config

        self.objective = objective or config.get("ttt_objectives", ["masked_patch"])[0]
        self.k_steps = k_steps if k_steps is not None else config.get("ttt_k_steps_sweep", [1])[0]
        self.lr = config.get("ttt_lr", 1e-4)
        self.mask_ratio = config.get("ttt_mask_ratio", 0.25)
        self.adapt_modules = list(config.get("ttt_adapt_modules", ["fusion", "prediction_head"]))

        self.use_consistency = use_consistency
        self.consistency_weight = config.get("consistency_weight", 0.1)

        self.use_mixup = use_mixup
        self.mixup_alpha_range = tuple(config.get("mixup_alpha_range", [0.7, 1.0]))

        self.grad_clip = config.get("ttt_grad_clip", 1.0)

        # Augmentation pipeline for consistency regularization
        self._consistency_transforms = None

    def _ttt_module_names(self) -> List[str]:
        """Resolve module names for TTT adaptation.

        Includes config-selected trainable modules and objective-specific heads.
        """
        names = list(self.adapt_modules)
        if self.objective == "masked_patch":
            names.append("mask_proj")
        elif self.objective == "rotation":
            names.append("rotation_head")

        # De-duplicate while preserving order.
        deduped = []
        seen = set()
        for name in names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def adapt_and_predict(
        self,
        images: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run TTT adaptation on a batch, then predict.

        CRITICAL: Saves and restores parameters around adaptation.

        Args:
            images: (B, 3, 224, 224) — original images (needed for rotation).
            visual_tokens: (B, 197, 768) — pre-encoded from frozen ViT.
            text_tokens: (B, L, 768) — pre-encoded from frozen BERT.
            text_mask: (B, L) — attention mask for text.

        Returns:
            logits: (B, num_answers) — predictions AFTER TTT.
            ttt_loss: float — final self-supervised loss value (for logging).
        """
        if self.k_steps == 0:
            # No adaptation — just predict
            logits, z = self.model.fuse_and_predict(visual_tokens, text_tokens, text_mask)
            return logits, 0.0

        # 1. Resolve and SAVE original parameters
        ttt_modules = self._ttt_module_names()
        named_ttt_params = self.model.get_ttt_params_named(
            adapt_modules=ttt_modules,
            include_auxiliary=True,
        )
        if not named_ttt_params:
            raise ValueError(
                "No parameters selected for TTT adaptation. "
                "Check config['ttt_adapt_modules']."
            )

        anchor_state = {name: param.data.clone() for name, param in named_ttt_params}

        # Save anchor representation for mixup
        if self.use_mixup:
            with torch.no_grad():
                z_anchor = self.model.fusion(visual_tokens, text_tokens, text_mask).detach()

        # Precompute consistency views BEFORE the K-loop (avoids redundant ViT forwards)
        consistency_views = None
        if self.use_consistency:
            consistency_views = self._precompute_consistency_views(images, text_tokens, text_mask)

        # 2. Create a SEPARATE optimizer for TTT
        ttt_optimizer = torch.optim.Adam([p for _, p in named_ttt_params], lr=self.lr)

        # 3-5. Adapt, predict, and ALWAYS restore original params
        final_loss = 0.0
        try:
            for step in range(self.k_steps):
                ttt_optimizer.zero_grad()

                # Inner-loop LR warmup (only active when K>1; K=1 is bit-identical)
                if self.k_steps > 1:
                    warmup_steps = min(self.k_steps, 2)
                    warmup_factor = min(1.0, (step + 1) / warmup_steps)
                    for pg in ttt_optimizer.param_groups:
                        pg['lr'] = self.lr * warmup_factor

                # Compute self-supervised loss
                if self.objective == "masked_patch":
                    loss = self.masked_patch_loss(visual_tokens, text_tokens, text_mask)
                elif self.objective == "rotation":
                    loss = self.rotation_loss(images, text_tokens, text_mask)
                elif self.objective == "contrastive":
                    loss = self.contrastive_loss(visual_tokens, text_tokens, text_mask)
                else:
                    raise ValueError(f"Unknown TTT objective: {self.objective}")

                # Optionally add consistency regularization
                if self.use_consistency and consistency_views is not None:
                    v1, v2 = consistency_views
                    cons_loss = self.consistency_loss(v1, v2, text_tokens, text_mask)
                    loss = loss + self.consistency_weight * cons_loss

                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in named_ttt_params], max_norm=self.grad_clip
                    )
                ttt_optimizer.step()
                final_loss = loss.item()

            # 4. Predict with adapted parameters
            z_adapted = self.model.fusion(visual_tokens, text_tokens, text_mask)

            # Optionally apply mixup anchoring
            if self.use_mixup:
                z_adapted = self.mixup_anchor(z_adapted, z_anchor)

            logits = self.model.prediction_head(z_adapted)
            return logits.detach(), final_loss
        finally:
            # Critical safety net: keep TTT strictly per-sample even on errors.
            for name, param in named_ttt_params:
                param.data.copy_(anchor_state[name])

    # ------------------------------------------------------------------
    # Self-supervised objectives
    # ------------------------------------------------------------------

    def masked_patch_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-token reconstruction at masked positions.

        1. Mask 25% of visual tokens (skip CLS at index 0)
        2. Run fusion with return_sequence=True → full visual sequence
        3. Extract outputs at masked positions
        4. Predict each masked token via mask_proj (Linear(768, 768))
        5. Loss = MSE(predicted_tokens, actual_masked_tokens)

        Args:
            visual_tokens: (B, 197, 768) — original visual tokens.
            text_tokens: (B, L, 768)
            text_mask: (B, L)

        Returns:
            Scalar MSE loss.
        """
        B, N, D = visual_tokens.shape
        num_patches = N - 1  # Exclude CLS token at index 0
        num_mask = max(1, int(num_patches * self.mask_ratio))

        # Generate random mask indices (same for all samples in batch for simplicity)
        mask_indices = torch.randperm(num_patches, device=visual_tokens.device)[:num_mask] + 1

        # Save targets: actual values at masked positions
        targets = visual_tokens[:, mask_indices, :].detach()  # (B, num_mask, D)

        # Create masked visual tokens
        masked_visual = visual_tokens.clone()
        masked_visual[:, mask_indices, :] = 0.0

        # Forward through fusion with full sequence output
        visual_seq = self.model.fusion(
            masked_visual, text_tokens, text_mask, return_sequence=True
        )  # (B, N, D)

        # Extract outputs at masked positions and predict
        masked_outputs = visual_seq[:, mask_indices, :]  # (B, num_mask, D)
        predicted = self.model.mask_proj(masked_outputs)  # (B, num_mask, D)

        return F.mse_loss(predicted, targets)

    def rotation_loss(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotation Prediction objective.

        1. Pick a random rotation: r ∈ {0, 90, 180, 270}
        2. Apply rotation to images
        3. Re-encode rotated image through ViT (frozen, no_grad)
        4. Run fusion on rotated visual tokens + text tokens
        5. Predict rotation class via rotation_head
        6. Loss = CrossEntropy(predicted_rotation, true_rotation)

        Args:
            images: (B, 3, 224, 224) — original images.
            text_tokens: (B, L, 768)
            text_mask: (B, L)

        Returns:
            Scalar cross-entropy loss.
        """
        B = images.shape[0]
        rotations = [0, 90, 180, 270]
        rotation_idx = random.randint(0, 3)
        angle = rotations[rotation_idx]

        # Apply rotation
        import torchvision.transforms.functional as TF
        rotated_images = TF.rotate(images, angle)

        # Re-encode through frozen ViT (no grad through ViT)
        with torch.no_grad():
            rotated_visual = self.model.vit(pixel_values=rotated_images).last_hidden_state

        # Fuse rotated visual + text
        z_rotated = self.model.fusion(rotated_visual, text_tokens, text_mask)

        # Predict rotation
        rotation_logits = self.model.rotation_head(z_rotated)  # (B, 4)

        # Target: same rotation for all samples in batch
        target = torch.full(
            (B,), rotation_idx, dtype=torch.long, device=images.device
        )

        return F.cross_entropy(rotation_logits, target)

    # ------------------------------------------------------------------
    # Stabilization techniques
    # ------------------------------------------------------------------

    def _precompute_consistency_views(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute augmented visual tokens for consistency loss.

        Called ONCE before the K-loop to avoid redundant ViT forwards
        (~35 GFLOPs per call) inside the loop.

        Args:
            images: (B, 3, 224, 224)
            text_tokens: unused (kept for signature consistency)
            text_mask: unused

        Returns:
            v1, v2: (B, 197, 768) — augmented visual tokens from frozen ViT.
        """
        import torchvision.transforms as T

        aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        aug1 = torch.stack([aug(img) for img in images])
        aug2 = torch.stack([aug(img) for img in images])

        with torch.no_grad():
            v1 = self.model.vit(pixel_values=aug1).last_hidden_state
            v2 = self.model.vit(pixel_values=aug2).last_hidden_state

        return v1, v2

    def consistency_loss(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consistency regularization using precomputed augmented views.

        Fuses both augmented visual token sets and computes symmetric KL
        divergence on prediction distributions.

        Args:
            v1: (B, 197, 768) — first augmented view visual tokens.
            v2: (B, 197, 768) — second augmented view visual tokens.
            text_tokens: (B, L, 768)
            text_mask: (B, L)

        Returns:
            Scalar symmetric KL loss.
        """
        z1 = self.model.fusion(v1, text_tokens, text_mask)
        z2 = self.model.fusion(v2, text_tokens, text_mask)

        p1 = F.log_softmax(self.model.prediction_head(z1), dim=-1)
        p2 = F.log_softmax(self.model.prediction_head(z2), dim=-1)

        # Symmetric KL
        kl_12 = F.kl_div(p1, p2.exp(), reduction="batchmean")
        kl_21 = F.kl_div(p2, p1.exp(), reduction="batchmean")

        return (kl_12 + kl_21) / 2.0

    def mixup_anchor(
        self,
        z_current: torch.Tensor,
        z_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Mixup anchoring to prevent parameter drift.

        Interpolates adapted representation toward the anchor (pre-TTT).

        Args:
            z_current: (B, 768) — representation after TTT adaptation.
            z_anchor: (B, 768) — representation before TTT (detached).

        Returns:
            z_mixed: (B, 768)
        """
        alpha_lo, alpha_hi = self.mixup_alpha_range
        alpha = random.uniform(alpha_lo, alpha_hi)
        return alpha * z_current + (1 - alpha) * z_anchor.detach()

    def contrastive_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-modal contrastive objective for TTT.

        Creates negative pairs by rolling text tokens within the batch,
        then trains the model to distinguish correct image-question pairings
        from incorrect ones. This is a stronger TTT signal than masked patch
        prediction because it directly operates on the question-visual
        alignment that VQA accuracy depends on.

        For batch size 1 (typical in TTT), we create a negative by zeroing
        out the text tokens to simulate a missing-question condition.

        Args:
            visual_tokens: (B, 197, 768) — pre-encoded from frozen ViT.
            text_tokens: (B, L, 768) — pre-encoded from frozen BERT.
            text_mask: (B, L) — attention mask for text.

        Returns:
            Scalar contrastive loss.
        """
        B = visual_tokens.shape[0]

        # Positive pair: correct image-question fusion
        z_pos = self.model.fusion(visual_tokens, text_tokens, text_mask)

        # Negative pair: mismatched text
        if B > 1:
            # Roll text tokens to create mismatched pairs
            text_neg = torch.roll(text_tokens, 1, dims=0)
            mask_neg = torch.roll(text_mask, 1, dims=0) if text_mask is not None else None
        else:
            # Single sample: use zeroed text as negative
            text_neg = torch.zeros_like(text_tokens)
            mask_neg = torch.zeros_like(text_mask) if text_mask is not None else None

        z_neg = self.model.fusion(visual_tokens, text_neg, mask_neg)

        # Use the gate as a binary discriminator: correct pairing should
        # get high confidence, incorrect should get low confidence
        conf_pos = self.model.gate(z_pos)  # (B, 1)
        conf_neg = self.model.gate(z_neg)  # (B, 1)

        logits = torch.cat([conf_pos, conf_neg], dim=0).squeeze(-1)
        labels = torch.cat([
            torch.ones(B, device=visual_tokens.device),
            torch.zeros(B, device=visual_tokens.device),
        ])

        return F.binary_cross_entropy(logits, labels)
