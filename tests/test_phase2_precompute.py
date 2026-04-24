"""
Phase 2 precompute: VM-local 5-view caches, frozen 8k subset, CLIP-LN skipped.

Does not launch a VM or read the real CLIP checkpoint. Path guards, subset
hygiene, and the view-assembly helper are what this file can prove on CPU.
"""

import json
import os
import sys

import pytest
import torch
from PIL import Image

from ttt.shift_cache import N_VIEWS, CachePathError, VIEW_LAYOUT

GPU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu")
sys.path.insert(0, GPU)
import precompute_shift_features as pre  # noqa: E402


def test_clip_ln_flag_is_skipped_not_implemented():
    with pytest.raises(NotImplementedError, match="vision cache"):
        pre.reject_clip_ln_ablation(True)
    pre.reject_clip_ln_ablation(False)


def test_output_on_drive_is_rejected(tmp_path):
    with pytest.raises(CachePathError, match="Drive"):
        pre.resolve_output_path("/content/drive/MyDrive/AdaTTT/blur.pt")


def test_output_in_tmp_is_accepted(tmp_path):
    out = pre.resolve_output_path(str(tmp_path / "blur_s3.pt"))
    assert out.endswith("blur_s3.pt")


def test_subset_file_is_read_not_rewritten(tmp_path):
    subset = tmp_path / "eval_subset_8k.json"
    payload = {"n": 3, "stratified_by": "answer_type", "sample_ids": ["a", "b", "c"]}
    subset.write_text(json.dumps(payload))
    before = subset.read_text()
    mtime = subset.stat().st_mtime
    ids = pre.load_subset_ids(str(subset))
    assert ids == ["a", "b", "c"]
    assert subset.read_text() == before
    assert subset.stat().st_mtime == mtime


def test_filter_preserves_subset_order():
    class FakeDS:
        samples = [
            {"sample_id": "z"},
            {"sample_id": "a"},
            {"sample_id": "b"},
            {"sample_id": "a"},
        ]

    kept = pre.filter_to_subset(FakeDS.samples, ["a", "b"])
    assert [s["sample_id"] for s in kept] == ["a", "b"]


def test_assemble_five_views_original_first(monkeypatch):
    img = Image.new("RGB", (16, 16), color=(10, 20, 30))

    fake_views = [Image.new("RGB", (16, 16), color=(i, 0, 0)) for i in range(4)]
    monkeypatch.setattr(pre, "corrupt", lambda image, kind, severity, sample_id: img)
    monkeypatch.setattr(pre, "augmix_views", lambda image, n_views, sample_id: fake_views)

    def transform(im):
        # Distinct per colour so we can see order.
        val = float(im.getpixel((0, 0))[0])
        return torch.full((3, 4, 4), val)

    views = pre.assemble_views(
        img, sample_id="42", kind="gaussian_blur", severity=3, transform=transform
    )
    assert views.shape == (N_VIEWS, 3, 4, 4)
    assert views[0].unique().tolist() == [10.0]
    assert views[1].unique().tolist() == [0.0]
    assert views[4].unique().tolist() == [3.0]


def test_encode_batch_keeps_raw_text_and_five_visuals():
    class FakeModel:
        def encode(self, images, input_ids, attention_mask):
            b = images.shape[0]
            vis = torch.ones(b, 197, 768)
            vis[:, 0, 0] = images[:, 0, 0, 0]  # stamp the view identity
            text = torch.full((b, 20, 512), 0.5)
            return vis, text

    images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 3, 8, 8).clone()
    vis, text = pre.encode_views(
        FakeModel(),
        images,
        torch.zeros(1, 20, dtype=torch.long),
        torch.ones(1, 20, dtype=torch.long),
    )
    assert vis.shape == (5, 197, 768)
    assert text.shape == (20, 512)
    # Text is encoded once and reused; visual rows carry the five view stamps.
    assert vis[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert torch.equal(text, torch.full((20, 512), 0.5))


def test_view_layout_constant_matches_cache():
    assert tuple(pre.VIEW_LAYOUT) == VIEW_LAYOUT
