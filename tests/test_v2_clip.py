"""
Phase 1 — CLIP backbone swap.

Covers the three things that fail silently rather than loudly:
  - width mismatch between CLIP's towers (vision 768, text 512)
  - CLIP fed ImageNet normalization instead of its own
  - CLIP fed BERT WordPiece ids instead of its BPE vocabulary

None of those raise. They just make the numbers worse, which is exactly the
class of bug that produced v1's results.
"""

import importlib
import os
import sys

import pytest
import torch

from ttt.data import NORMALIZATION, build_tokenizer, get_image_transform
from ttt.gate import AdaptiveRouter
from ttt.models import CLIP_TEXT_DIM, CLIP_VISUAL_DIM, FullVQAModel, FusionModule

SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)


@pytest.fixture
def clip_config():
    return {
        "encoder_backend": "clip",
        "vision_encoder": "openai/clip-vit-base-patch16",
        "text_encoder": "openai/clip-vit-base-patch16",
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


class TestTextProjector:
    """CLIP's towers disagree on width; the fusion has to reconcile them."""

    def test_projector_created_when_widths_differ(self):
        fusion = FusionModule(dim=768, num_layers=1, text_dim=512)
        assert fusion.text_proj is not None
        assert fusion.text_proj.in_features == CLIP_TEXT_DIM
        assert fusion.text_proj.out_features == CLIP_VISUAL_DIM
        assert sum(p.numel() for p in fusion.text_proj.parameters()) == 512 * 768 + 768

    def test_no_projector_when_widths_match(self):
        assert FusionModule(dim=768, num_layers=1, text_dim=768).text_proj is None
        assert FusionModule(dim=768, num_layers=1).text_proj is None

    def test_fusion_accepts_clip_shaped_inputs(self, clip_config):
        model = FullVQAModel(clip_config)
        logits, z = model.fuse_and_predict(
            torch.randn(2, 197, 768),
            torch.randn(2, 20, 512),
            torch.ones(2, 20, dtype=torch.long),
        )
        assert logits.shape == (2, clip_config["num_answers"])
        assert z.shape == (2, 768)

    def test_projector_is_trainable(self, clip_config):
        """It must stay out of the feature cache, so it has to keep learning."""
        model = FullVQAModel(clip_config)
        assert all(p.requires_grad for p in model.fusion.text_proj.parameters())

    def test_return_sequence_path_still_works(self, clip_config):
        """masked_patch reads the visual sequence; projection touches text only."""
        model = FullVQAModel(clip_config)
        seq = model.fusion(
            torch.randn(2, 197, 768),
            torch.randn(2, 20, 512),
            torch.ones(2, 20, dtype=torch.long),
            return_sequence=True,
        )
        assert seq.shape == (2, 197, 768)


class TestImageNormalization:
    """Wrong statistics degrade CLIP silently — no exception, worse numbers."""

    def test_clip_and_imagenet_differ(self):
        assert NORMALIZATION["clip"] != NORMALIZATION["vit_bert"]

    def test_clip_uses_its_own_statistics(self):
        mean, std = NORMALIZATION["clip"]
        assert mean == pytest.approx([0.48145466, 0.4578275, 0.40821073])
        assert std == pytest.approx([0.26862954, 0.26130258, 0.27577711])

    def test_transform_applies_the_backend_statistics(self):
        ones = torch.ones(3, 8, 8)
        for backend in ("clip", "vit_bert"):
            mean, std = NORMALIZATION[backend]
            out = get_image_transform(8, backend=backend).transforms[-1](ones.clone())
            assert out[0, 0, 0] == pytest.approx((1.0 - mean[0]) / std[0], rel=1e-5)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder backend"):
            get_image_transform(224, backend="siglip")


class TestBackendDispatch:
    def test_unknown_backend_raises_on_load(self, clip_config):
        model = FullVQAModel(clip_config)
        with pytest.raises(ValueError, match="Unknown encoder_backend"):
            model.load_encoders({**clip_config, "encoder_backend": "siglip"})

    def test_tokenizer_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown encoder backend"):
            build_tokenizer({"encoder_backend": "siglip"})


class TestBackendFlops:
    """Swapping BERT for CLIP's text tower halves the base forward."""

    def teardown_method(self):
        AdaptiveRouter.configure_for_backend("vit_bert")

    def test_clip_base_is_cheaper_than_vit_bert(self):
        AdaptiveRouter.configure_for_backend("vit_bert")
        bert_base = AdaptiveRouter.sample_flops(adapted=False)
        AdaptiveRouter.configure_for_backend("clip")
        clip_base = AdaptiveRouter.sample_flops(adapted=False)
        assert clip_base < bert_base / 1.5

    def test_clip_worsens_the_adapted_to_base_ratio(self):
        """The base gets cheaper; four augmented encodes do not. Ratio rises."""
        ratios = {}
        for backend in ("vit_bert", "clip"):
            AdaptiveRouter.configure_for_backend(backend)
            ratios[backend] = (
                AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1)
                / AdaptiveRouter.sample_flops(adapted=False)
            )
        assert ratios["clip"] > ratios["vit_bert"], (
            f"CLIP ratio {ratios['clip']:.2f}x should exceed vit_bert's "
            f"{ratios['vit_bert']:.2f}x — cheaper base, same augmentation cost."
        )

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder backend"):
            AdaptiveRouter.configure_for_backend("siglip")


def _load_ceiling_module():
    sys.path.insert(0, SCRIPTS)
    try:
        return importlib.import_module("06_ceiling_check")
    finally:
        sys.path.remove(SCRIPTS)


class TestCeilingCheck:
    """The Phase 1 kill criterion, including the case that must not kill."""

    def setup_method(self):
        self.mod = _load_ceiling_module()

    def _history(self, ep5, ep8, final=None):
        history = {e: 40.0 for e in range(1, 9)}
        history[5], history[8] = ep5, ep8
        if final is not None:
            history[8] = final
        return history

    def test_flat_and_no_better_kills(self):
        r = self.mod.evaluate(self._history(49.11, 49.08), 0.5, 2.0, 49.56)
        assert r["verdict"] == "KILL"

    def test_flat_but_much_better_continues(self):
        """Saturating early at a higher ceiling is the success case."""
        r = self.mod.evaluate(self._history(62.0, 62.1), 0.5, 2.0, 54.30)
        assert r["is_flat"] and not r["is_near_v1"]
        assert r["verdict"] == "CONTINUE"

    def test_still_climbing_continues(self):
        r = self.mod.evaluate(self._history(50.0, 53.0), 0.5, 2.0, 54.30)
        assert not r["is_flat"]
        assert r["verdict"] == "CONTINUE"

    def test_missing_epochs_raise(self):
        with pytest.raises(ValueError, match="Criterion reads epochs"):
            self.mod.evaluate({1: 40.0, 2: 42.0}, 0.5, 2.0, 54.30)

    def test_multi_run_log_is_split_not_merged(self, tmp_path):
        """logs/train.log holds several appended runs; merging reads the wrong one."""
        log = tmp_path / "train.log"
        log.write_text(
            "\n".join(
                [f"Epoch {e} | Val accuracy: {40 + e}%" for e in range(1, 9)]
                + [f"Epoch {e} | Val accuracy: {70 + e}%" for e in range(1, 9)]
            )
        )
        runs = self.mod.parse_log_runs(str(log))
        assert len(runs) == 2
        assert self.mod.read_history(str(log), run_index=0)[8] == pytest.approx(48.0)
        assert self.mod.read_history(str(log), run_index=-1)[8] == pytest.approx(78.0)

    def test_flat_and_far_below_is_a_regression_not_a_moved_ceiling(self):
        """Two runs flat near 30% sit ~20pp under v1; 'ceiling moved' would be backwards."""
        r = self.mod.evaluate(self._history(30.34, 30.65), 0.5, 2.0, 49.56)
        assert r["is_flat"] and r["is_below"]
        assert r["verdict"] == "REGRESSION"

    def test_soft_lines_parse_separately_from_exact(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("\n".join(
            f"Epoch {e} | Val accuracy: {30 + e}%\nEpoch {e} | Official VQA soft: {47 + e}%"
            for e in range(1, 9)))
        exact = self.mod.read_history(str(log), metric="exact")
        soft = self.mod.read_history(str(log), metric="soft")
        assert exact[8] == pytest.approx(38.0) and soft[8] == pytest.approx(55.0)

    def test_resumed_log_with_replayed_epochs_is_one_run(self, tmp_path):
        """Preemption replay (1..6, then 5..8 after --resume) is one run.

        Splitting it would keep only epochs 5-8 of the tail and lose the
        pre-preemption history; merging by last-write keeps all eight.
        """
        lines = [f"Epoch {e} | Val accuracy: {40 + e}%" for e in range(1, 7)]
        lines += [f"Epoch {e} | Val accuracy: {50 + e}%" for e in range(5, 9)]
        log = tmp_path / "train.log"
        log.write_text("\n".join(lines))

        runs = self.mod.parse_log_runs(str(log))
        assert len(runs) == 1
        history = runs[0]
        assert sorted(history) == list(range(1, 9))
        assert history[4] == pytest.approx(44.0)  # pre-preemption epoch kept
        assert history[5] == pytest.approx(55.0)  # replay overwrites

    def test_v1_log_reproduces_the_known_kill(self):
        """Regression anchor: run 0 of the real log is v1's VQA run."""
        path = os.path.join(os.path.dirname(SCRIPTS), "logs", "train.log")
        if not os.path.exists(path):
            pytest.skip("logs/train.log not present")

        history = self.mod.read_history(path, run_index=0)
        r = self.mod.evaluate(history, 0.5, 2.0, 49.56)
        assert r["acc_ep5"] == pytest.approx(49.11)
        assert r["slope_5_to_8"] == pytest.approx(-0.03, abs=0.01)
        assert r["verdict"] == "KILL"
