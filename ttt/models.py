"""
Model components for the Efficient TTT VQA system.

Contains:
- FusionModule: Bidirectional cross-modal attention (~4.7M params)
- ConfidenceGate: Lightweight confidence MLP (~50K params)
- PredictionHead: Answer classifier MLP (~2.3M params)
- FullVQAModel: Complete model with frozen ViT + BERT + trainable components
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q attends to K, V from another modality."""

    def __init__(self, dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, D)
            key:   (B, Lk, D)
            value: (B, Lk, D)
            key_mask: (B, Lk) — True where valid, False where padded.

        Returns:
            (B, Lq, D)
        """
        B, Lq, D = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lk)

        if key_mask is not None:
            # key_mask: (B, Lk) → (B, 1, 1, Lk)
            attn = attn.masked_fill(~key_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Fusion Layer (one layer of the cross-modal transformer)
# ---------------------------------------------------------------------------


class FusionLayer(nn.Module):
    """One layer of bidirectional cross-modal fusion.

    Structure:
        1. CrossAttention(Q=visual, K=text, V=text) + residual + LayerNorm
        2. CrossAttention(Q=text, K=visual, V=visual) + residual + LayerNorm
        3. FFN on visual + residual + LayerNorm
        4. FFN on text + residual + LayerNorm
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        # Visual attends to text
        self.cross_attn_v2t = CrossAttention(dim, num_heads, dropout)
        self.norm_v2t = nn.LayerNorm(dim)

        # Text attends to visual
        self.cross_attn_t2v = CrossAttention(dim, num_heads, dropout)
        self.norm_t2v = nn.LayerNorm(dim)

        # FFN for visual stream
        self.ffn_v = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_v = nn.LayerNorm(dim)

        # FFN for text stream
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_t = nn.LayerNorm(dim)

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual: (B, Lv, D) — visual tokens
            text:   (B, Lt, D) — text tokens
            text_mask: (B, Lt) — attention mask for text

        Returns:
            updated_visual: (B, Lv, D)
            updated_text:   (B, Lt, D)
        """
        # Visual attends to text
        visual = self.norm_v2t(visual + self.cross_attn_v2t(visual, text, text, text_mask))
        # Text attends to visual
        text = self.norm_t2v(text + self.cross_attn_t2v(text, visual, visual))
        # FFNs
        visual = self.norm_ffn_v(visual + self.ffn_v(visual))
        text = self.norm_ffn_t(text + self.ffn_t(text))
        return visual, text


# ---------------------------------------------------------------------------
# FusionModule
# ---------------------------------------------------------------------------


class FusionModule(nn.Module):
    """Cross-modal fusion via bidirectional cross-attention.

    Stacks `num_layers` FusionLayers, then pools the visual stream
    to produce a single fused vector z ∈ R^(768).

    Parameters: θ_f (~4.7M params for 2 layers)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [FusionLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.pool_norm = nn.LayerNorm(dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, 197, 768) from ViT
            text_tokens:   (B, L, 768) from BERT
            text_mask:     (B, L) attention mask
            return_sequence: If True, return full visual sequence instead of pooled.

        Returns:
            If return_sequence=False: z (B, 768) — pooled fused representation
            If return_sequence=True:  (B, 197, 768) — full visual sequence
        """
        v, t = visual_tokens, text_tokens
        for layer in self.layers:
            v, t = layer(v, t, text_mask)

        if return_sequence:
            return v

        # Pool: use CLS token (index 0)
        z = self.pool_norm(v[:, 0, :])
        return z


# ---------------------------------------------------------------------------
# ConfidenceGate
# ---------------------------------------------------------------------------


class ConfidenceGate(nn.Module):
    """Lightweight MLP that predicts whether TTT will help this sample.

    Architecture: Linear(768, hidden) → ReLU → Dropout → Linear(hidden, 1) → Sigmoid
    Parameters: θ_g (~50K params)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 768) — fused representation.

        Returns:
            confidence: (B, 1) — values in [0, 1].
            High confidence → SKIP TTT. Low confidence → APPLY TTT.
        """
        return self.mlp(z)

    def route(self, z: torch.Tensor, threshold: float) -> torch.Tensor:
        """Determine which samples skip TTT.

        Args:
            z: (B, 768)
            threshold: Gate threshold τ.

        Returns:
            mask: (B,) — True = SKIP TTT, False = APPLY TTT.
        """
        confidence = self.forward(z)
        return confidence.squeeze(-1) > threshold


# ---------------------------------------------------------------------------
# PredictionHead
# ---------------------------------------------------------------------------


class PredictionHead(nn.Module):
    """MLP that maps fused representation to answer logits.

    Architecture: Linear(768, hidden) → ReLU → Dropout → LayerNorm → Linear(hidden, num_answers)
    Parameters: θ_d (~2.3M params for num_answers=3129)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 768)

        Returns:
            logits: (B, num_answers)
        """
        return self.classifier(z)


# ---------------------------------------------------------------------------
# Frozen Encoder Loaders
# ---------------------------------------------------------------------------


def load_frozen_vit(model_name: str = "google/vit-base-patch16-224") -> nn.Module:
    """Load ViT and freeze ALL parameters.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Frozen ViT model in eval mode.
    """
    from transformers import ViTModel

    vit = ViTModel.from_pretrained(model_name)
    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()
    return vit


def load_frozen_bert(model_name: str = "bert-base-uncased") -> nn.Module:
    """Load BERT and freeze ALL parameters.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Frozen BERT model in eval mode.
    """
    from transformers import BertModel

    bert = BertModel.from_pretrained(model_name)
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()
    return bert


# ---------------------------------------------------------------------------
# Auxiliary projection heads (used only during TTT)
# ---------------------------------------------------------------------------


class MaskedPatchProjection(nn.Module):
    """Projects pooled z to predict mean of masked visual tokens."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class RotationHead(nn.Module):
    """Predicts rotation class (0°, 90°, 180°, 270°) from pooled z."""

    def __init__(self, dim: int = 768, num_rotations: int = 4):
        super().__init__()
        self.head = nn.Linear(dim, num_rotations)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ---------------------------------------------------------------------------
# FullVQAModel
# ---------------------------------------------------------------------------


class FullVQAModel(nn.Module):
    """Complete VQA model combining all components.

    Components:
        - self.vit: frozen ViT-B/16 (google/vit-base-patch16-224)
        - self.bert: frozen BERT-base (bert-base-uncased)
        - self.fusion: FusionModule (θ_f — trainable)
        - self.gate: ConfidenceGate (θ_g — trainable)
        - self.prediction_head: PredictionHead (θ_d — trainable)
        - self.mask_proj: MaskedPatchProjection (for TTT masked patch objective)
        - self.rotation_head: RotationHead (for TTT rotation objective)

    IMPORTANT:
        ViT and BERT are FROZEN. Only fusion, gate, and prediction_head are trainable.
    """

    def __init__(self, config: Dict):
        super().__init__()
        dim = config.get("fusion_dim", 768)
        num_heads = config.get("fusion_heads", 12)
        num_layers = config.get("fusion_layers", 2)
        dropout = config.get("fusion_dropout", 0.1)
        pred_hidden = config.get("prediction_hidden", 1024)
        num_answers = config.get("num_answers", 3129)
        gate_hidden = config.get("gate_hidden", 256)

        # Frozen encoders (loaded lazily — see load_encoders())
        self.vit = None
        self.bert = None

        # Trainable components
        self.fusion = FusionModule(dim, num_heads, num_layers, dropout)
        self.gate = ConfidenceGate(dim, gate_hidden, dropout)
        self.prediction_head = PredictionHead(dim, pred_hidden, num_answers)

        # TTT auxiliary heads
        self.mask_proj = MaskedPatchProjection(dim)
        self.rotation_head = RotationHead(dim)

        # Modules that are allowed to adapt during TTT.
        self.ttt_adapt_modules = config.get("ttt_adapt_modules", ["fusion", "prediction_head"])

    def load_encoders(self, config: Dict) -> None:
        """Load frozen ViT and BERT encoders.

        Call this separately so tests can skip the expensive download.
        """
        self.vit = load_frozen_vit(config.get("vision_encoder", "google/vit-base-patch16-224"))
        self.bert = load_frozen_bert(config.get("text_encoder", "bert-base-uncased"))

    def train(self, mode: bool = True):
        """Override train() to keep frozen encoders in eval mode.

        This prevents dropout/stochastic behavior in ViT/BERT when the top-level
        model is switched to train mode for fusion/head optimization.
        """
        super().train(mode)
        if self.vit is not None:
            self.vit.eval()
        if self.bert is not None:
            self.bert.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass (training mode).

        Args:
            images: (B, 3, 224, 224)
            input_ids: (B, L) — BERT token IDs
            attention_mask: (B, L) — BERT attention mask

        Returns:
            logits: (B, num_answers)
            confidence: (B, 1)
            z: (B, 768) — fused representation
        """
        visual_tokens, text_tokens = self.encode(images, input_ids, attention_mask)
        z = self.fusion(visual_tokens, text_tokens, attention_mask)
        confidence = self.gate(z)
        logits = self.prediction_head(z)
        return logits, confidence, z

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run frozen encoders only.

        Args:
            images: (B, 3, 224, 224)
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            visual_tokens: (B, 197, 768) from ViT
            text_tokens: (B, L, 768) from BERT
        """
        assert self.vit is not None and self.bert is not None, (
            "Call model.load_encoders(config) before forward pass."
        )
        # Keep frozen encoders deterministic even if parent module is in train mode.
        self.vit.eval()
        self.bert.eval()
        visual_tokens = self.vit(pixel_values=images).last_hidden_state
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_output.last_hidden_state
        return visual_tokens, text_tokens

    def fuse_and_predict(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run fusion + prediction on pre-encoded tokens.

        Used by TTT loop for efficient adaptation (avoids re-encoding).

        Args:
            visual_tokens: (B, 197, 768)
            text_tokens: (B, L, 768)
            text_mask: (B, L)

        Returns:
            logits: (B, num_answers)
            z: (B, 768)
        """
        z = self.fusion(visual_tokens, text_tokens, text_mask)
        logits = self.prediction_head(z)
        return logits, z

    def _resolve_ttt_modules(
        self,
        adapt_modules: Optional[List[str]] = None,
        include_auxiliary: bool = False,
    ) -> List[Tuple[str, nn.Module]]:
        """Resolve named modules used by TTT.

        Args:
            adapt_modules: Module names to adapt. If None, uses self.ttt_adapt_modules.
            include_auxiliary: Include objective-specific auxiliary heads in valid names.

        Returns:
            List of (module_name, module) tuples in the requested order.
        """
        requested = adapt_modules if adapt_modules is not None else self.ttt_adapt_modules
        if requested is None:
            requested = []

        registry: Dict[str, nn.Module] = {
            "fusion": self.fusion,
            "prediction_head": self.prediction_head,
            "gate": self.gate,
        }
        if include_auxiliary:
            registry["mask_proj"] = self.mask_proj
            registry["rotation_head"] = self.rotation_head

        modules: List[Tuple[str, nn.Module]] = []
        seen = set()
        for name in requested:
            if name in seen:
                continue
            if name not in registry:
                valid = ", ".join(sorted(registry.keys()))
                raise ValueError(f"Unknown TTT module '{name}'. Valid modules: {valid}")
            modules.append((name, registry[name]))
            seen.add(name)
        return modules

    def get_ttt_params(
        self,
        adapt_modules: Optional[List[str]] = None,
        include_auxiliary: bool = False,
    ) -> List[torch.nn.Parameter]:
        """Get parameters that TTT can update.

        Args:
            adapt_modules: Optional explicit module names to adapt.
            include_auxiliary: Whether auxiliary TTT heads are valid module names.
        """
        params: List[torch.nn.Parameter] = []
        for _, module in self._resolve_ttt_modules(adapt_modules, include_auxiliary):
            params.extend(list(module.parameters()))
        return params

    def get_ttt_params_named(
        self,
        adapt_modules: Optional[List[str]] = None,
        include_auxiliary: bool = False,
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        """Get named parameters for TTT (for save/restore)."""
        params: List[Tuple[str, torch.nn.Parameter]] = []
        for module_name, module in self._resolve_ttt_modules(adapt_modules, include_auxiliary):
            for name, param in module.named_parameters():
                params.append((f"{module_name}.{name}", param))
        return params
