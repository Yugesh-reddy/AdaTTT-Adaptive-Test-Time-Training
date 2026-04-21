"""
Tests for CachedFeaturesDataset and cached_vqa_collate_fn.

These cover the encoder-caching path used by gpu/run_ttt_sweep.py (and
future scripts) to skip redundant ViT/BERT forwards.
"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from ttt.data import CachedFeaturesDataset, cached_vqa_collate_fn


def _make_cache(tmp_path, n=4, max_q=20, sample_ids=None):
    """Build a minimal feature cache file matching the precompute script output."""
    sample_ids = sample_ids or [f"q{i}" for i in range(n)]
    cache = {
        "sample_ids": sample_ids,
        "visual_tokens": torch.randn(n, 197, 768, dtype=torch.float16),
        "text_tokens": torch.randn(n, max_q, 768, dtype=torch.float16),
        "attention_masks": torch.ones(n, max_q, dtype=torch.bool),
        "answer_idx": torch.arange(n, dtype=torch.long),
        "question_types": ["yes/no"] * n,
        "dataset": "vqa_v2",
        "split": "val",
        "num_samples": n,
        "max_question_length": max_q,
        "dtype": "torch.float16",
    }
    path = os.path.join(tmp_path, "features.pt")
    torch.save(cache, path)
    return path


class _FakeSourceDataset:
    """Minimal stand-in that mirrors VQADataset's `.samples` and __getitem__."""

    def __init__(self, sample_ids, image_size=224, max_q=20):
        self.samples = [{"sample_id": sid} for sid in sample_ids]
        self.image_size = image_size
        self.max_q = max_q

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "image": torch.zeros(3, self.image_size, self.image_size),
            "input_ids": torch.arange(self.max_q, dtype=torch.long),
            "attention_mask": torch.ones(self.max_q, dtype=torch.long),
            "answer_idx": idx,
            "question_type": "yes/no",
            "sample_id": self.samples[idx]["sample_id"],
        }


class TestCachedFeaturesDataset:
    def test_length_and_fields_no_images(self, tmp_path):
        path = _make_cache(str(tmp_path))
        ds = CachedFeaturesDataset(features_path=path, load_images=False)
        assert len(ds) == 4
        sample = ds[0]
        assert sample["visual_tokens"].shape == (197, 768)
        assert sample["text_tokens"].shape == (20, 768)
        assert sample["visual_tokens"].dtype == torch.float32
        assert sample["sample_id"] == "q0"

    def test_load_images_requires_source(self, tmp_path):
        path = _make_cache(str(tmp_path))
        with pytest.raises(ValueError, match="requires source_dataset"):
            CachedFeaturesDataset(features_path=path, load_images=True)

    def test_misaligned_source_raises(self, tmp_path):
        path = _make_cache(str(tmp_path), sample_ids=["a", "b", "c", "d"])
        src = _FakeSourceDataset(["a", "b", "x", "d"])
        with pytest.raises(ValueError, match="does not align"):
            CachedFeaturesDataset(features_path=path, source_dataset=src, load_images=True)

    def test_load_images_returns_source_image(self, tmp_path):
        ids = ["a", "b"]
        path = _make_cache(str(tmp_path), n=2, sample_ids=ids)
        src = _FakeSourceDataset(ids)
        ds = CachedFeaturesDataset(features_path=path, source_dataset=src, load_images=True)
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["input_ids"].numel() > 0

    def test_collate_stacks_features(self, tmp_path):
        path = _make_cache(str(tmp_path))
        ds = CachedFeaturesDataset(features_path=path, load_images=False)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=cached_vqa_collate_fn)
        batch = next(iter(loader))
        assert batch["visual_tokens"].shape == (2, 197, 768)
        assert batch["text_tokens"].shape == (2, 20, 768)
        assert batch["attention_mask"].shape == (2, 20)
        assert batch["sample_ids"] == ["q0", "q1"]
        # Without load_images, images+input_ids must NOT be present
        assert "images" not in batch
        assert "input_ids" not in batch

    def test_collate_includes_images_when_loaded(self, tmp_path):
        ids = ["a", "b"]
        path = _make_cache(str(tmp_path), n=2, sample_ids=ids)
        src = _FakeSourceDataset(ids)
        ds = CachedFeaturesDataset(features_path=path, source_dataset=src, load_images=True)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=cached_vqa_collate_fn)
        batch = next(iter(loader))
        assert batch["images"].shape == (2, 3, 224, 224)
        assert batch["input_ids"].shape == (2, 20)

    def test_missing_key_raises(self, tmp_path):
        bad = {"sample_ids": ["a"]}  # intentionally incomplete
        path = os.path.join(str(tmp_path), "bad.pt")
        torch.save(bad, path)
        with pytest.raises(ValueError, match="missing required key"):
            CachedFeaturesDataset(features_path=path, load_images=False)
