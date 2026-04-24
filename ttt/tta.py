"""
Phase 2 test-time adapters: TENT, MEMO, EATA, SAR.

Main method: fusion-LayerNorm-only + MEMO marginal entropy H(mean(p_aug4)),
K=1, SAR reliability filter, per-sample restore.

CLIP-LN (vision-tower LayerNorms) is skipped: it invalidates the 5-view
vision cache and needs on-the-fly encodes at ~4× cost.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def shannon_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entropy of a probability tensor along `dim`."""
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=dim)


def tent_loss(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits). TENT's objective."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=-1)
    return shannon_entropy(probs, dim=-1).mean()


def memo_loss(view_logits: torch.Tensor) -> torch.Tensor:
    """Marginal entropy H(mean_v softmax(logits_v)). MEMO's objective.

    Args:
        view_logits: (n_views, C) or (B, n_views, C)
    """
    if view_logits.dim() == 2:
        view_logits = view_logits.unsqueeze(0)
    probs = F.softmax(view_logits, dim=-1)
    p_bar = probs.mean(dim=-2)
    return shannon_entropy(p_bar, dim=-1).mean()


def sar_should_adapt(entropy: float, e0: float) -> bool:
    """SAR reliability filter: skip samples whose entropy is already unreliable."""
    return float(entropy) < float(e0)


def eata_should_adapt(
    entropy: float,
    e0: float,
    probs: Optional[torch.Tensor] = None,
    prototype: Optional[torch.Tensor] = None,
    cosine_threshold: float = 0.9,
) -> bool:
    """EATA sample filter: drop high-entropy (unreliable) and redundant samples.

    Fisher anti-forgetting is omitted: Phase 2 restores parameters per sample,
    so there is no running test-time state to forget.
    """
    if float(entropy) >= float(e0):
        return False
    if prototype is not None and probs is not None:
        p = probs.reshape(1, -1).float()
        q = prototype.reshape(1, -1).float()
        cos = F.cosine_similarity(p, q).item()
        if cos > cosine_threshold:
            return False
    return True


def clip_ln_not_supported() -> None:
    raise NotImplementedError(
        "CLIP-LN adapts vision-tower LayerNorms and invalidates the vision cache "
        "(~4× on-the-fly image encodes). Skip it, or give it a separate budget line."
    )


_MEMO_METHODS = {"memo", "memo_sar"}
_SAR_METHODS = {"sar", "memo_sar"}


class TTAAdapter:
    """Per-sample test-time adapter with guaranteed restore.

    Args:
        method: tent | memo | eata | sar | memo_sar
        n_aug: AugMix views in the MEMO loss. Default 4 for memo*, 0 otherwise.
        layernorm_only: Fusion LayerNorm affines only (the main method).
        adapt_vision_ln: Must stay False; CLIP-LN is a priced skip.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        method: str = "memo_sar",
        k_steps: int = 1,
        n_aug: Optional[int] = None,
        layernorm_only: bool = True,
        adapt_vision_ln: bool = False,
        sar_e0_ratio: float = 0.4,
        eata_e0_ratio: float = 0.4,
        eata_cosine_threshold: float = 0.9,
        lr: Optional[float] = None,
    ):
        if adapt_vision_ln:
            clip_ln_not_supported()
        if method not in {"tent", "memo", "eata", "sar", "memo_sar"}:
            raise ValueError(f"Unknown TTA method '{method}'")
        self.model = model
        self.config = config
        self.method = method
        self.k_steps = k_steps
        self.layernorm_only = layernorm_only
        self.n_aug = int(n_aug if n_aug is not None else (4 if method in _MEMO_METHODS else 0))
        self.sar_e0_ratio = sar_e0_ratio
        self.eata_e0_ratio = eata_e0_ratio
        self.eata_cosine_threshold = eata_cosine_threshold
        self.lr = float(lr if lr is not None else config.get("ttt_lr", 1e-4))
        self.grad_clip = float(config.get("ttt_grad_clip", 1.0))
        self.adapt_modules = list(config.get("ttt_adapt_modules", ["fusion"]))
        self._eata_prototype: Optional[torch.Tensor] = None
        self.memo_loss = memo_loss
        self.tent_loss = tent_loss

    def _named_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return self.model.get_ttt_params_named(
            adapt_modules=self.adapt_modules,
            include_auxiliary=False,
            layernorm_only=self.layernorm_only,
        )

    def _optimizer_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        params: List[Tuple[str, torch.nn.Parameter]],
    ) -> None:
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in params], max_norm=self.grad_clip
            )
        optimizer.step()

    def _forward_logits(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        z = self.model.fusion(visual, text, mask)
        return self.model.prediction_head(z)

    def adapt_and_predict(
        self,
        visual_original: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_views: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Adapt on one sample, predict, restore. Always restores, even on error."""
        vis = visual_original if visual_original.dim() == 3 else visual_original.unsqueeze(0)
        txt = text_tokens if text_tokens.dim() == 3 else text_tokens.unsqueeze(0)
        mask = text_mask
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        named = self._named_params()
        if not named:
            raise ValueError("No TTA parameters selected (check layernorm_only / adapt_modules).")
        anchor = {name: param.data.clone() for name, param in named}

        try:
            with torch.no_grad():
                base_logits = self._forward_logits(vis, txt, mask)
                probs0 = F.softmax(base_logits, dim=-1)
                entropy0 = float(shannon_entropy(probs0, dim=-1).mean().item())
                log_c = math.log(base_logits.size(-1))

            filtered = None
            should = True
            if self.method in _SAR_METHODS:
                e0 = self.sar_e0_ratio * log_c
                if not sar_should_adapt(entropy0, e0):
                    should = False
                    filtered = "sar"
            elif self.method == "eata":
                e0 = self.eata_e0_ratio * log_c
                if not eata_should_adapt(
                    entropy0,
                    e0,
                    probs=probs0,
                    prototype=self._eata_prototype,
                    cosine_threshold=self.eata_cosine_threshold,
                ):
                    should = False
                    filtered = "eata"

            if (not should) or self.k_steps == 0:
                return base_logits.detach(), {
                    "adapted": False,
                    "n_aug": 0,
                    "filtered": filtered,
                    "method": self.method,
                    "loss": 0.0,
                    "entropy": entropy0,
                }

            optimizer = torch.optim.Adam([p for _, p in named], lr=self.lr)
            final_loss = 0.0
            for _ in range(self.k_steps):
                optimizer.zero_grad()
                if self.method in _MEMO_METHODS:
                    if visual_views is None:
                        raise ValueError("MEMO requires visual_views (4 AugMix rows).")
                    views = visual_views[0] if visual_views.dim() == 4 else visual_views
                    view_logits = []
                    for i in range(self.n_aug):
                        view_logits.append(
                            self._forward_logits(views[i : i + 1], txt, mask)
                        )
                    loss = self.memo_loss(torch.cat(view_logits, dim=0))
                else:
                    loss = self.tent_loss(self._forward_logits(vis, txt, mask))
                self._optimizer_step(loss, optimizer, named)
                final_loss = float(loss.item())

            logits = self._forward_logits(vis, txt, mask)
            if self.method == "eata":
                self._eata_prototype = F.softmax(logits.detach(), dim=-1)
            return logits.detach(), {
                "adapted": True,
                "n_aug": self.n_aug,
                "filtered": None,
                "method": self.method,
                "loss": final_loss,
                "entropy": entropy0,
            }
        finally:
            for name, param in named:
                param.data.copy_(anchor[name])
