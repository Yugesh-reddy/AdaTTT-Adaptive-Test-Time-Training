"""
Phase 2 multi-view shift cache: 5 views/condition, no mmap, no Drive paths.

MEMO adapts on 4 AugMix views and predicts on the original (corrupted) image,
so the cache stores original + 4 views. Text is raw encoder output (CLIP 512-d),
never the trainable projector.
"""

import os

import pytest
import torch
from torch.utils.data import DataLoader

from ttt.shift_cache import (
    N_AUGMIX,
    N_VIEWS,
    ORIGINAL_VIEW,
    VIEW_LAYOUT,
    CachePathError,
    MultiViewCachedFeaturesDataset,
    assert_cache_path_allowed,
    is_drive_path,
    multiview_collate_fn,
    write_shift_cache,
)


def _cache_blob(n=3, n_views=N_VIEWS, vis=197, vis_dim=768, text_len=20, text_dim=512):
    return {
        "sample_ids": [f"q{i}" for i in range(n)],
        "visual_tokens": torch.randn(n, n_views, vis, vis_dim, dtype=torch.float16),
        "text_tokens": torch.randn(n, text_len, text_dim, dtype=torch.float16),
        "attention_masks": torch.ones(n, text_len, dtype=torch.bool),
        "answer_idx": torch.arange(n, dtype=torch.long),
        "answer_scores": torch.zeros(n, 8, dtype=torch.float32),
        "question_types": ["yes/no"] * n,
        "n_views": n_views,
        "view_layout": list(VIEW_LAYOUT),
        "text_projected": False,
        "corruption": "gaussian_blur",
        "severity": 3,
        "dtype": "float16",
    }


def _write(tmp_path, blob=None, name="cache.pt"):
    path = os.path.join(str(tmp_path), name)
    write_shift_cache(path, blob or _cache_blob())
    return path


class TestViewLayout:
    def test_five_views_original_plus_four_augmix(self):
        assert N_VIEWS == 5
        assert N_AUGMIX == 4
        assert ORIGINAL_VIEW == 0
        assert VIEW_LAYOUT[0] == "original"
        assert len(VIEW_LAYOUT) == 5
        assert VIEW_LAYOUT[1:] == ("augmix_0", "augmix_1", "augmix_2", "augmix_3")


class TestCachePathGuard:
    def test_drive_markers_are_detected(self):
        assert is_drive_path("/content/drive/MyDrive/AdaTTT/cache.pt")
        assert is_drive_path(
            "/Users/x/Library/CloudStorage/GoogleDrive-foo@gmail.com/My Drive/AdaTTT/x.pt"
        )
        assert not is_drive_path("/tmp/phase2/cache.pt")
        assert not is_drive_path("/content/phase2_cache/blur_s3.pt")

    def test_drive_path_is_rejected(self):
        with pytest.raises(CachePathError, match="Drive"):
            assert_cache_path_allowed("/content/drive/MyDrive/AdaTTT/cache.pt")

    def test_local_tmp_is_allowed(self, tmp_path):
        target = tmp_path / "ok.pt"
        assert os.path.abspath(assert_cache_path_allowed(str(target))) == os.path.abspath(
            str(target)
        )


class TestWriteAndLoad:
    def test_roundtrip_shapes_and_fp16_storage(self, tmp_path):
        path = _write(tmp_path)
        ds = MultiViewCachedFeaturesDataset(path)
        assert len(ds) == 3
        sample = ds[0]
        assert sample["visual_tokens"].shape == (5, 197, 768)
        assert sample["text_tokens"].shape == (20, 512)
        assert sample["visual_tokens"].dtype == torch.float32
        assert sample["text_tokens"].dtype == torch.float32
        assert sample["sample_id"] == "q0"
        assert ds.text_projected is False

        stored = torch.load(path, map_location="cpu", weights_only=False)
        assert stored["visual_tokens"].dtype == torch.float16
        assert stored["n_views"] == 5

    def test_manifest_records_seeds_and_sha256(self, tmp_path):
        path = _write(tmp_path)
        manifest_path = os.path.splitext(path)[0] + ".manifest.json"
        assert os.path.isfile(manifest_path)
        import json

        manifest = json.load(open(manifest_path))
        assert manifest["n_views"] == 5
        assert manifest["text_projected"] is False
        assert manifest["clip_ln_ablation"] == "skipped"
        assert "sha256" in manifest and len(manifest["sha256"]) == 64
        assert manifest["seed_fn"].startswith("crc32")
        assert manifest["legacy_ladder_gflops"]["tent_k1"] == 58
        assert manifest["legacy_ladder_note"]

    def test_wrong_view_count_raises(self, tmp_path):
        blob = _cache_blob(n_views=4)
        blob["visual_tokens"] = torch.randn(2, 4, 197, 768, dtype=torch.float16)
        blob["n_views"] = 4
        path = os.path.join(str(tmp_path), "bad.pt")
        with pytest.raises(ValueError, match="5 views"):
            write_shift_cache(path, blob)

    def test_missing_key_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.pt")
        torch.save({"sample_ids": ["a"]}, path)
        with pytest.raises(ValueError, match="missing required key"):
            MultiViewCachedFeaturesDataset(path)

    def test_mmap_flag_is_rejected(self, tmp_path):
        path = _write(tmp_path)
        with pytest.raises(ValueError, match="mmap"):
            MultiViewCachedFeaturesDataset(path, mmap=True)

    def test_source_never_calls_torch_load_with_mmap(self):
        import inspect

        import ttt.shift_cache as mod

        src = inspect.getsource(mod)
        assert "mmap=True" not in src

    def test_collate_stacks_views(self, tmp_path):
        path = _write(tmp_path)
        ds = MultiViewCachedFeaturesDataset(path)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=multiview_collate_fn)
        batch = next(iter(loader))
        assert batch["visual_tokens"].shape == (2, 5, 197, 768)
        assert batch["text_tokens"].shape == (2, 20, 512)
        assert batch["answer_scores"].shape[0] == 2
        assert batch["sample_ids"] == ["q0", "q1"]
        assert "images" not in batch
