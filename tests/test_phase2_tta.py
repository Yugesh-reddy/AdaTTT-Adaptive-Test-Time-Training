"""
Phase 2 TTA adapters: TENT / MEMO / EATA / SAR, fusion-LN only, per-sample restore.

CLIP-LN is skipped: adapting vision LayerNorms invalidates the 5-view cache.
"""

import pytest
import torch
import torch.nn.functional as F

from ttt.models import FullVQAModel
from ttt.tta import (
    TTAAdapter,
    clip_ln_not_supported,
    eata_should_adapt,
    memo_loss,
    sar_should_adapt,
    shannon_entropy,
    tent_loss,
)


def _config():
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.0,
        "prediction_hidden": 64,
        "num_answers": 10,
        "gate_hidden": 16,
        "num_query_tokens": 1,
        "text_dim": 512,
        "ttt_lr": 5e-2,
        "ttt_grad_clip": 1.0,
        "ttt_adapt_modules": ["fusion"],
        "ttt_k_steps_sweep": [1],
    }


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = FullVQAModel(_config())
    m.eval()
    return m


def _batch(n_aug=4, answers=10):
    visual = torch.randn(1, 197, 768)
    views = torch.randn(n_aug, 197, 768)
    text = torch.randn(1, 20, 512)
    mask = torch.ones(1, 20, dtype=torch.long)
    return visual, views, text, mask


class TestEntropyObjectives:
    def test_memo_is_entropy_of_mean_not_mean_of_entropy(self):
        # Two peaked distributions on different classes: each H(p)≈0, H(mean p)≈log 2.
        logits = torch.zeros(2, 4)
        logits[0, 0] = 20.0
        logits[1, 1] = 20.0
        memo = memo_loss(logits).item()
        tent_mean = torch.stack([tent_loss(logits[i]) for i in range(2)]).mean().item()
        assert tent_mean < 0.05
        assert memo > 0.5
        assert memo == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=0.1)

    def test_tent_is_zero_on_a_one_hot(self):
        logits = torch.zeros(1, 5)
        logits[0, 3] = 30.0
        assert tent_loss(logits).item() < 1e-5

    def test_shannon_entropy_uniform(self):
        p = torch.full((1, 8), 1 / 8)
        assert shannon_entropy(p).item() == pytest.approx(torch.log(torch.tensor(8.0)).item())


class TestFilters:
    def test_sar_skips_unreliable_high_entropy(self):
        e0 = 0.4 * torch.log(torch.tensor(10.0)).item()
        assert sar_should_adapt(entropy=0.1, e0=e0) is True
        assert sar_should_adapt(entropy=e0 + 1.0, e0=e0) is False

    def test_eata_skips_unreliable_and_redundant(self):
        e0 = 1.0
        p = F.softmax(torch.tensor([[4.0, 0.0, 0.0]]), dim=-1)
        assert eata_should_adapt(entropy=0.2, e0=e0, probs=p, prototype=None) is True
        assert eata_should_adapt(entropy=2.0, e0=e0, probs=p, prototype=None) is False
        assert eata_should_adapt(entropy=0.2, e0=e0, probs=p, prototype=p, cosine_threshold=0.9) is False


class TestClipLnSkipped:
    def test_clip_ln_raises(self):
        with pytest.raises(NotImplementedError, match="vision cache"):
            clip_ln_not_supported()

    def test_adapter_rejects_vision_ln(self, model):
        with pytest.raises(NotImplementedError, match="vision cache"):
            TTAAdapter(model, _config(), method="memo", adapt_vision_ln=True)


class TestLayernormOnlyRestore:
    def test_only_fusion_layernorms_are_selected(self, model):
        adapter = TTAAdapter(model, _config(), method="tent", k_steps=1)
        names = {n for n, _ in adapter._named_params()}
        assert names
        assert all("fusion." in n for n in names)
        assert all(
            n.endswith(".weight") or n.endswith(".bias") for n in names
        )
        assert not any("text_proj" in n for n in names)
        assert not any("prediction_head" in n for n in names)

    def test_params_restored_after_memo(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(model, _config(), method="memo", k_steps=1, n_aug=4)
        before = {n: p.detach().clone() for n, p in adapter._named_params()}
        logits, info = adapter.adapt_and_predict(visual, text, mask, visual_views=views)
        assert logits.shape == (1, 10)
        assert info["adapted"] is True
        assert info["n_aug"] == 4
        for n, p in adapter._named_params():
            assert torch.equal(p.detach(), before[n]), n

    def test_params_changed_during_step_then_restored(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(model, _config(), method="tent", k_steps=1, n_aug=0)
        seen = {}

        orig_step = adapter._optimizer_step

        def wrapped(loss, optimizer, params):
            orig_step(loss, optimizer, params)
            seen["after"] = {n: p.detach().clone() for n, p in params}

        adapter._optimizer_step = wrapped
        before = {n: p.detach().clone() for n, p in adapter._named_params()}
        adapter.adapt_and_predict(visual, text, mask)
        assert seen["after"]
        assert any(not torch.equal(seen["after"][n], before[n]) for n in before)
        for n, p in adapter._named_params():
            assert torch.equal(p.detach(), before[n])

    def test_restore_on_exception(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(model, _config(), method="memo", k_steps=1, n_aug=4)
        before = {n: p.detach().clone() for n, p in adapter._named_params()}
        adapter.memo_loss = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            adapter.adapt_and_predict(visual, text, mask, visual_views=views)
        for n, p in adapter._named_params():
            assert torch.equal(p.detach(), before[n])

    def test_sar_filter_skips_without_touching_params(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(
            model, _config(), method="memo_sar", k_steps=1, n_aug=4, sar_e0_ratio=0.0
        )
        before = {n: p.detach().clone() for n, p in adapter._named_params()}
        logits, info = adapter.adapt_and_predict(visual, text, mask, visual_views=views)
        assert info["adapted"] is False
        assert info["filtered"] == "sar"
        for n, p in adapter._named_params():
            assert torch.equal(p.detach(), before[n])
        assert logits.shape == (1, 10)

    def test_tent_charges_zero_aug_views(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(model, _config(), method="tent", k_steps=1)
        _, info = adapter.adapt_and_predict(visual, text, mask)
        assert info["n_aug"] == 0

    def test_eata_runs_and_restores(self, model):
        visual, views, text, mask = _batch()
        adapter = TTAAdapter(model, _config(), method="eata", k_steps=1)
        before = {n: p.detach().clone() for n, p in adapter._named_params()}
        _, info = adapter.adapt_and_predict(visual, text, mask)
        assert info["method"] == "eata"
        for n, p in adapter._named_params():
            assert torch.equal(p.detach(), before[n])
