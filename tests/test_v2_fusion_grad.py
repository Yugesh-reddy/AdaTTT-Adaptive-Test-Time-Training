"""
Every fusion parameter must receive a gradient from the VQA loss.

FusionModule pools z from the visual stream only, so the last FusionLayer's
text update (t2v cross-attention, text FFN and their norms — 7.1M params at
dim 768) never reaches the loss. Phase 1 checkpoints carry those parameters at
their initial values with no AdamW state. Nothing raises; the model is just 23%
larger than what actually trains.

The default keeps them so existing checkpoints still load and resume;
fusion_skip_last_text_update drops them.
"""

import pytest
import torch

from ttt.models import FullVQAModel

DEAD_WHEN_KEPT = (
    "layers.1.cross_attn_t2v.",
    "layers.1.norm_t2v.",
    "layers.1.ffn_t.",
    "layers.1.norm_ffn_t.",
)


def _phase1_config(**overrides):
    # config/config.yaml's Phase 1 architecture: CLIP widths, 2 layers, 1 query.
    config = {
        "fusion_dim": 768,
        "text_dim": 512,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.1,
        "prediction_hidden": 1024,
        "num_answers": 100,
        "gate_hidden": 256,
        "num_query_tokens": 1,
    }
    config.update(overrides)
    return config


def _random_batch(B=4, Lt=20):
    torch.manual_seed(0)
    visual = torch.randn(B, 197, 768)
    text = torch.randn(B, Lt, 512)
    mask = torch.ones(B, Lt, dtype=torch.long)
    mask[:, 15:] = 0  # padded tail, as the tokenizer produces
    return visual, text, mask


@pytest.mark.parametrize(
    "skip_last_text_update",
    [
        pytest.param(
            False,
            id="default",
            marks=pytest.mark.xfail(
                reason="default keeps the last layer's text update so existing "
                       "checkpoints load; it never receives a gradient. An XPASS "
                       "means the default changed and old checkpoints may break",
                strict=True,
            ),
        ),
        pytest.param(True, id="skip_last_text_update"),
    ],
)
def test_every_fusion_parameter_gets_a_gradient(skip_last_text_update):
    model = FullVQAModel(
        _phase1_config(fusion_skip_last_text_update=skip_last_text_update)
    ).train()
    visual, text, mask = _random_batch()

    logits, _ = model.fuse_and_predict(visual, text, mask)
    logits.sum().backward()

    no_grad = [n for n, p in model.fusion.named_parameters() if p.grad is None]
    assert no_grad == [], f"fusion parameters with no gradient: {no_grad}"


def test_skipping_the_dead_update_leaves_the_output_unchanged():
    """A default checkpoint minus the dead parameters is the same model."""
    torch.manual_seed(0)
    default = FullVQAModel(_phase1_config()).eval()
    lean = FullVQAModel(_phase1_config(fusion_skip_last_text_update=True)).eval()

    # Strict load: fails if the lean layout drops anything beyond the dead set,
    # or keeps any of it.
    lean.fusion.load_state_dict(
        {k: v for k, v in default.fusion.state_dict().items()
         if not k.startswith(DEAD_WHEN_KEPT)}
    )
    lean.prediction_head.load_state_dict(default.prediction_head.state_dict())

    visual, text, mask = _random_batch()
    with torch.no_grad():
        torch.testing.assert_close(
            lean.fuse_and_predict(visual, text, mask)[0],
            default.fuse_and_predict(visual, text, mask)[0],
        )
