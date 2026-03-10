"""
Tests for TTT adaptation loop.

Validates parameter save/restore, loss computation, and prediction changes.
Uses random tensors — no model downloads required.
"""

import pytest
import torch

from ttt.models import FullVQAModel
from ttt.ttt_loop import TTTAdapter


@pytest.fixture
def config():
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.1,
        "prediction_hidden": 1024,
        "num_answers": 3129,
        "gate_hidden": 256,
        "ttt_objectives": ["masked_patch"],
        "ttt_k_steps_sweep": [1],
        "ttt_lr": 1e-3,
        "ttt_mask_ratio": 0.25,
        "consistency_weight": 0.1,
        "mixup_alpha_range": [0.7, 1.0],
    }


@pytest.fixture
def model_and_tensors(config):
    """Create model and random input tensors."""
    model = FullVQAModel(config)
    B = 2
    visual = torch.randn(B, 197, 768)
    text = torch.randn(B, 20, 768)
    text_mask = torch.ones(B, 20, dtype=torch.bool)
    images = torch.randn(B, 3, 224, 224)
    return model, images, visual, text, text_mask


class TestTTTParamRestore:
    def test_params_restored_after_ttt(self, config, model_and_tensors):
        """After TTT adaptation, model params return to original values."""
        model, images, visual, text, text_mask = model_and_tensors

        # Save original params
        original_params = {
            name: param.data.clone()
            for name, param in model.get_ttt_params_named()
        }

        # Run TTT
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=3)
        adapter.adapt_and_predict(images, visual, text, text_mask)

        # Verify params are restored
        for name, param in model.get_ttt_params_named():
            assert torch.allclose(param.data, original_params[name], atol=1e-6), (
                f"Parameter {name} was not restored after TTT!"
            )

    def test_k0_no_change(self, config, model_and_tensors):
        """K=0 produces no parameter changes at all."""
        model, images, visual, text, text_mask = model_and_tensors

        original_params = {
            name: param.data.clone()
            for name, param in model.get_ttt_params_named()
        }

        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=0)
        logits, loss = adapter.adapt_and_predict(images, visual, text, text_mask)

        assert loss == 0.0

        for name, param in model.get_ttt_params_named():
            assert torch.equal(param.data, original_params[name])


class TestMaskedPatchLoss:
    def test_masked_patch_loss_is_positive(self, config, model_and_tensors):
        """Masked patch loss is a positive scalar."""
        model, images, visual, text, text_mask = model_and_tensors
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)
        loss = adapter.masked_patch_loss(visual, text, text_mask)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive


class TestRotationLoss:
    def test_rotation_loss_is_positive(self, config, model_and_tensors):
        """Rotation prediction loss is a positive scalar.

        NOTE: This test requires model.vit to be available for re-encoding rotated images.
        We mock the ViT with a simple module for testing.
        """
        model, images, visual, text, text_mask = model_and_tensors

        # Mock ViT for testing (avoids HuggingFace download)
        class MockViTOutput:
            def __init__(self, hidden):
                self.last_hidden_state = hidden

        class MockViT(torch.nn.Module):
            def forward(self, pixel_values):
                B = pixel_values.shape[0]
                return MockViTOutput(torch.randn(B, 197, 768))

        model.vit = MockViT()

        adapter = TTTAdapter(model, config, objective="rotation", k_steps=1)
        loss = adapter.rotation_loss(images, text, text_mask)

        assert loss.dim() == 0
        assert loss.item() > 0


class TestTTTChangesPrediction:
    def test_ttt_changes_logits(self, config, model_and_tensors):
        """TTT with K>0 produces different logits than K=0."""
        model, images, visual, text, text_mask = model_and_tensors

        # K=0 prediction
        adapter_k0 = TTTAdapter(model, config, objective="masked_patch", k_steps=0)
        logits_k0, _ = adapter_k0.adapt_and_predict(images, visual, text, text_mask)

        # K=3 prediction (more steps = more likely to differ)
        adapter_k3 = TTTAdapter(model, config, objective="masked_patch", k_steps=3)
        logits_k3, _ = adapter_k3.adapt_and_predict(images, visual, text, text_mask)

        # They should be different (with very high probability for random inputs)
        assert not torch.allclose(logits_k0, logits_k3, atol=1e-4), (
            "TTT with K=3 should produce different logits than K=0"
        )
