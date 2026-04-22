"""
AdaTTT-v2.2 Phase 0 acceptance contracts.

These encode the hygiene requirements agreed for v2 before any backbone or
adapter work begins. Several are expected to FAIL against the v1 tree — that is
the point: each failure names a specific thing Phase 0 has to fix.

Contract                      Status against v1
-------------------------------------------------------------------------
gate label balance            FAIL — 47,612/48,099 labels are "skip" (99.0%)
restore after adapt           PASS — regression guard for ttt_loop.py:142,206
LayerNorm-only selection      FAIL — no layernorm_only mode exists
dataset routing               FAIL — no shared router; "vizwiz" silently
                                     loads VQA-v2 paths in all four runners
soft metric recoverable       FAIL — prediction files carry no answer_scores
deployment cost function      FAIL — no per-sample cost fn; v1's cascade
                                     undercounted the escalated path
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from ttt.gate import AdaptiveRouter
from ttt.models import FullVQAModel
from ttt.ttt_loop import TTTAdapter


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Minimum share of the minority class in any gate-label set. v1 collapsed to
# 1.01% because labels were generated on the memorised training split.
MIN_MINORITY_SHARE = 0.05


@pytest.fixture
def config():
    """Small but production-shaped: 2 fusion layers keeps the tests fast."""
    return {
        "fusion_dim": 768,
        "fusion_heads": 12,
        "fusion_layers": 2,
        "fusion_dropout": 0.1,
        "prediction_hidden": 1024,
        "num_answers": 100,
        "gate_hidden": 256,
        "num_query_tokens": 32,
        "ttt_objectives": ["masked_patch"],
        "ttt_k_steps_sweep": [1],
        "ttt_lr": 1e-3,
        "ttt_mask_ratio": 0.25,
        "ttt_adapt_modules": ["fusion", "prediction_head"],
        "consistency_weight": 0.1,
        "mixup_alpha_range": [0.7, 1.0],
    }


@pytest.fixture
def model(config):
    """FullVQAModel without encoders — fusion-level tests need no ViT/BERT."""
    return FullVQAModel(config)


def _gate_label_files():
    data_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.isdir(data_dir):
        return []
    return [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.startswith("gate_labels") and f.endswith(".json")
    ]


class TestGateLabelBalance:
    """Gate labels must come from a split the base model has not memorised.

    v1's data/gate_labels_train.json is derived from the checkpoint the
    Memotion2 run overwrote, so it cannot be regenerated now — xfail until
    Phase 4 relabels on data/gate_train_subset_8k.json. An XPASS here is the
    signal to delete this marker.
    """

    @pytest.mark.xfail(
        reason="v1 labels came from the memorised train split; Phase 4 relabels "
               "on the frozen held-out subset",
        strict=False,
    )
    @pytest.mark.parametrize("path", _gate_label_files())
    def test_labels_are_not_collapsed(self, path):
        labels = [item["gate_label"] for item in json.load(open(path))]
        assert labels, f"{os.path.basename(path)} is empty"

        share_adapt = sum(1 for v in labels if v == 0.0) / len(labels)
        share_skip = sum(1 for v in labels if v == 1.0) / len(labels)
        minority = min(share_adapt, share_skip)

        assert minority >= MIN_MINORITY_SHARE, (
            f"{os.path.basename(path)}: minority class is {minority:.2%} of "
            f"{len(labels):,} labels (skip={share_skip:.2%}, adapt={share_adapt:.2%}). "
            f"A gate trained on this collapses to a constant. Regenerate the "
            f"labels on held-out data where base accuracy is near 50%, not on "
            f"the training split where it is 95%."
        )


class TestRestoreAfterAdapt:
    """TTT is per-sample: parameters must return to the anchor state."""

    def test_parameters_restored_bitwise(self, config, model):
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)
        named = model.get_ttt_params_named(
            adapt_modules=["fusion", "prediction_head", "mask_proj"],
            include_auxiliary=True,
        )
        before = {n: p.detach().clone() for n, p in named}

        visual = torch.randn(2, 197, 768)
        text = torch.randn(2, 20, 768)
        mask = torch.ones(2, 20, dtype=torch.long)
        adapter.adapt_and_predict(torch.randn(2, 3, 224, 224), visual, text, mask)

        for name, param in named:
            assert torch.equal(param.detach(), before[name]), (
                f"{name} was not restored after adaptation — TTT is leaking "
                f"state across samples."
            )

    def test_parameters_restored_when_objective_raises(self, config, model):
        adapter = TTTAdapter(model, config, objective="masked_patch", k_steps=1)
        named = model.get_ttt_params_named(
            adapt_modules=["fusion", "prediction_head", "mask_proj"],
            include_auxiliary=True,
        )
        before = {n: p.detach().clone() for n, p in named}

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure mid-adaptation")

        adapter.masked_patch_loss = boom
        with pytest.raises(RuntimeError):
            adapter.adapt_and_predict(
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 197, 768),
                torch.randn(1, 20, 768),
                torch.ones(1, 20, dtype=torch.long),
            )

        for name, param in named:
            assert torch.equal(param.detach(), before[name]), (
                f"{name} not restored after a mid-adaptation exception."
            )


class TestLayerNormOnlySelection:
    """v2 adapts fusion LayerNorm affines only — needs a first-class mode."""

    def test_layernorm_only_returns_exactly_the_affines(self, model):
        expected = {
            f"fusion.{n}"
            for mod_name, mod in model.fusion.named_modules()
            if isinstance(mod, nn.LayerNorm)
            for n, _ in ((f"{mod_name}.{pn}", p) for pn, p in mod.named_parameters())
        }
        assert expected, "fusion has no LayerNorm modules — fixture is wrong"

        selected = model.get_ttt_params_named(
            adapt_modules=["fusion"], layernorm_only=True
        )
        assert {n for n, _ in selected} == expected

    def test_layernorm_only_is_a_tiny_fraction(self, model):
        selected = model.get_ttt_params_named(
            adapt_modules=["fusion"], layernorm_only=True
        )
        n_ln = sum(p.numel() for _, p in selected)
        n_all = sum(p.numel() for p in model.fusion.parameters())
        assert n_ln / n_all < 0.01, (
            f"LayerNorm-only selected {n_ln:,} of {n_all:,} fusion parameters"
        )


class TestDatasetRouting:
    """A dataset flag must route or fail loudly — never fall through to VQA-v2."""

    def test_router_exists(self):
        from ttt import data as ttt_data

        assert hasattr(ttt_data, "build_dataset"), (
            "No shared dataset router. Each runner branches inline on "
            "is_memotion2, so --dataset vizwiz silently loads VQA-v2 paths."
        )

    @pytest.mark.parametrize(
        "name,expected", [("vqa_v2", "VQADataset"), ("vizwiz", "VizWizDataset"),
                          ("memotion2", "Memotion2Dataset")]
    )
    def test_known_datasets_route_to_their_own_class(self, name, expected, config):
        from ttt.data import build_dataset

        cls = build_dataset(config, name, split="val", cls_only=True)
        assert cls.__name__ == expected

    def test_unknown_dataset_raises(self, config):
        from ttt.data import build_dataset

        with pytest.raises(ValueError):
            build_dataset(config, "not_a_dataset", split="val", cls_only=True)


class TestSoftMetricRecoverable:
    """The official metric must be computable from any prediction file.

    The contract sits on the writer, not on the artifacts: v1's results predate
    it and cannot be regenerated (the checkpoint that produced them was
    overwritten by the Memotion2 run), so they are marked xfail rather than
    deleted. Phase 1 replaces them.
    """

    def test_writer_carries_answer_scores(self):
        from ttt.metrics import prediction_record

        record = prediction_record(
            "262148000", 7, 7, "other", answer_scores=[0.0, 1.0, 1 / 3], ttt_loss=0.0
        )
        assert "answer_scores" in record
        assert record["answer_scores"][2] == pytest.approx(1 / 3)

    def test_writer_is_explicit_when_scores_are_unavailable(self):
        """Absent scores must be absent, not silently zero-filled."""
        from ttt.metrics import prediction_record

        assert "answer_scores" not in prediction_record("1", 0, 0, "yes/no")

    @pytest.mark.xfail(
        reason="v1 artifacts predate the contract and cannot be regenerated; "
               "Phase 1 replaces them",
        strict=False,
    )
    @pytest.mark.parametrize(
        "path",
        [
            "results/ttt_predictions/val/k0_baseline.json",
            "results/ttt_predictions/val/k1_masked_patch.json",
        ],
    )
    def test_legacy_predictions_carry_answer_scores(self, path):
        full = os.path.join(REPO_ROOT, path)
        if not os.path.exists(full):
            pytest.skip(f"{path} not present")

        with open(full) as fh:
            first = json.load(fh)[0]
        assert "answer_scores" in first


class TestFrozenSubsets:
    """Two disjoint held-out subsets, frozen once and reused everywhere."""

    @pytest.mark.parametrize(
        "name", ["eval_subset_8k.json", "gate_train_subset_8k.json"]
    )
    def test_subset_exists_and_is_stratified(self, name):
        path = os.path.join(REPO_ROOT, "data", name)
        assert os.path.exists(path), f"data/{name} has not been frozen"

        blob = json.load(open(path))
        assert blob["n"] == len(blob["sample_ids"]) == 8000
        assert blob["stratified_by"] == "answer_type"
        assert len(set(blob["sample_ids"])) == 8000, "duplicate ids in subset"

    def test_subsets_are_disjoint(self):
        """Training gate signal B on the reporting subset would leak."""
        ids = []
        for name in ("eval_subset_8k.json", "gate_train_subset_8k.json"):
            path = os.path.join(REPO_ROOT, "data", name)
            if not os.path.exists(path):
                pytest.skip(f"data/{name} not frozen yet")
            ids.append(set(json.load(open(path))["sample_ids"]))

        overlap = ids[0] & ids[1]
        assert not overlap, (
            f"{len(overlap)} ids appear in both the reporting subset and the "
            f"gate-training subset."
        )


class TestDeploymentCost:
    """Cost must be priced from the deployment path, not the cached path.

    v1's cascade charged escalated requests only the expensive tier, omitting
    the cheap-tier encode every request had already paid. The v2 equivalent is
    charging an adapted sample as if its augmented views were free because the
    experiment reads them from a feature cache.
    """

    def test_sample_flops_exists(self):
        assert hasattr(AdaptiveRouter, "sample_flops"), (
            "No per-sample deployment cost function."
        )

    def test_adapted_costs_more_than_skipped(self):
        skipped = AdaptiveRouter.sample_flops(adapted=False)
        adapted = AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1)
        assert adapted > skipped, (
            "An adapted sample must be charged the base forward it also ran."
        )

    def test_augmented_views_are_charged(self):
        """4 AugMix views cost 4 image-tower encodes; text is encoded once."""
        one = AdaptiveRouter.sample_flops(adapted=True, n_aug=1, k_steps=1)
        four = AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1)
        assert four > one, "Augmented views are not being charged at all."

        per_view = (four - one) / 3.0
        assert per_view > 5e9, (
            f"Each extra augmented view is charged {per_view/1e9:.1f} GFLOPs; a "
            f"ViT-B/16 image-tower forward is ~17.6 GFLOPs. The cache-read cost "
            f"is being reported instead of the deployment cost."
        )

    def test_layernorm_only_does_not_reduce_backward_flops(self):
        """Gradients still traverse the whole fusion stack to reach the affines.

        The 2,886x parameter reduction is optimizer state, not compute.
        """
        full = AdaptiveRouter.sample_flops(adapted=True, n_aug=4, k_steps=1)
        ln = AdaptiveRouter.sample_flops(
            adapted=True, n_aug=4, k_steps=1, layernorm_only=True
        )
        assert ln == pytest.approx(full, rel=0.05)
