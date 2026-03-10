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

        self.use_consistency = use_consistency
        self.consistency_weight = config.get("consistency_weight", 0.1)

        self.use_mixup = use_mixup
        self.mixup_alpha_range = tuple(config.get("mixup_alpha_range", [0.7, 1.0]))

        # Augmentation pipeline for consistency regularization
        self._consistency_transforms = None

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

        # 1. SAVE original parameters
        anchor_state = {
            name: param.data.clone()
            for name, param in self.model.get_ttt_params_named()
        }

        # Save anchor representation for mixup
        if self.use_mixup:
            with torch.no_grad():
                z_anchor = self.model.fusion(visual_tokens, text_tokens, text_mask).detach()

        # 2. Create a SEPARATE optimizer for TTT
        ttt_optimizer = torch.optim.Adam(self.model.get_ttt_params(), lr=self.lr)

        # 3. TTT gradient steps
        final_loss = 0.0
        for step in range(self.k_steps):
            ttt_optimizer.zero_grad()

            # Compute self-supervised loss
            if self.objective == "masked_patch":
                loss = self.masked_patch_loss(visual_tokens, text_tokens, text_mask)
            elif self.objective == "rotation":
                loss = self.rotation_loss(images, text_tokens, text_mask)
            else:
                raise ValueError(f"Unknown TTT objective: {self.objective}")

            # Optionally add consistency regularization
            if self.use_consistency:
                cons_loss = self.consistency_loss(
                    images, visual_tokens, text_tokens, text_mask
                )
                loss = loss + self.consistency_weight * cons_loss

            loss.backward()
            ttt_optimizer.step()
            final_loss = loss.item()

        # 4. Predict with adapted parameters
        z_adapted = self.model.fusion(visual_tokens, text_tokens, text_mask)

        # Optionally apply mixup anchoring
        if self.use_mixup:
            z_adapted = self.mixup_anchor(z_adapted, z_anchor)

        logits = self.model.prediction_head(z_adapted)

        # 5. RESTORE original parameters (CRITICAL)
        for name, param in self.model.get_ttt_params_named():
            param.data.copy_(anchor_state[name])

        return logits.detach(), final_loss

    # ------------------------------------------------------------------
    # Self-supervised objectives
    # ------------------------------------------------------------------

    def masked_patch_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked Patch Prediction objective.

        Simplified approach (v1):
            1. Mask 25% of visual tokens (skip CLS at index 0)
            2. Run fusion → pooled z
            3. From z, predict the mean of masked tokens via Linear(768, 768)
            4. Loss = MSE(predicted_mean, actual_mean_of_masked_tokens)

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

        # Save target: mean of masked tokens
        target = visual_tokens[:, mask_indices, :].mean(dim=1).detach()  # (B, D)

        # Create masked visual tokens
        masked_visual = visual_tokens.clone()
        masked_visual[:, mask_indices, :] = 0.0

        # Forward through fusion
        z = self.model.fusion(masked_visual, text_tokens, text_mask)  # (B, D)

        # Predict mean of masked tokens
        predicted = self.model.mask_proj(z)  # (B, D)

        return F.mse_loss(predicted, target)

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

    def consistency_loss(
        self,
        images: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consistency regularization within the TTT loop.

        1. Generate two augmented views of the image
        2. Encode both through frozen ViT
        3. Fuse both and get prediction distributions
        4. Loss = symmetric KL divergence

        Args:
            images: (B, 3, 224, 224)
            visual_tokens: (B, 197, 768) — unused, images re-encoded after augmentation.
            text_tokens: (B, L, 768)
            text_mask: (B, L)

        Returns:
            Scalar symmetric KL loss.
        """
        import torchvision.transforms as T

        # Create augmented views
        aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        aug1 = torch.stack([aug(img) for img in images])
        aug2 = torch.stack([aug(img) for img in images])

        # Encode through frozen ViT
        with torch.no_grad():
            v1 = self.model.vit(pixel_values=aug1).last_hidden_state
            v2 = self.model.vit(pixel_values=aug2).last_hidden_state

        # Fuse and predict
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
