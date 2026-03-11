"""
Tests for Memotion2Dataset and utilities.

Uses temp files and mock data — no real data downloads required.
"""

import json
import os
import tempfile

import pytest
import torch
import numpy as np
from PIL import Image

from ttt.data import (
    Memotion2Dataset,
    build_memotion2_label_map,
    vqa_collate_fn,
)


@pytest.fixture(autouse=True)
def mock_image_transform(monkeypatch):
    """Mock get_image_transform to avoid torchvision dependency."""
    import ttt.data as data_module

    def _fake_transform(image_size=224):
        def _transform(img):
            return torch.randn(3, image_size, image_size)
        return _transform

    monkeypatch.setattr(data_module, "get_image_transform", _fake_transform)


@pytest.fixture
def mock_memotion2_dir():
    """Create a temp directory with mock Memotion2 annotations and images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image directory
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Create mock images (small 32×32 to keep tests fast)
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(os.path.join(img_dir, f"meme_{i:03d}.jpg"))

        # Create mock annotations
        annotations = [
            {
                "id": 0,
                "image": "meme_000.jpg",
                "text": "When you realize it's Monday again",
                "sentiment": "negative",
            },
            {
                "id": 1,
                "image": "meme_001.jpg",
                "text": "Finally Friday!",
                "sentiment": "positive",
            },
            {
                "id": 2,
                "image": "meme_002.jpg",
                "text": "Just another day",
                "sentiment": "neutral",
            },
            {
                "id": 3,
                "image": "meme_003.jpg",
                "text": "LOL this is great",
                "sentiment": "positive",
            },
            {
                "id": 4,
                "image": "meme_004.jpg",
                "text": "",  # empty OCR text
                "sentiment": "neutral",
            },
        ]
        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        yield {
            "tmpdir": tmpdir,
            "annotations_path": ann_path,
            "image_dir": img_dir,
            "num_samples": len(annotations),
        }


class TestBuildMemotion2LabelMap:
    def test_returns_3_classes(self):
        label_map = build_memotion2_label_map()
        assert len(label_map) == 3

    def test_expected_labels(self):
        label_map = build_memotion2_label_map()
        assert "positive" in label_map
        assert "negative" in label_map
        assert "neutral" in label_map

    def test_indices_are_unique(self):
        label_map = build_memotion2_label_map()
        indices = list(label_map.values())
        assert len(set(indices)) == len(indices)


class TestMemotion2Dataset:
    def test_construction(self, mock_memotion2_dir):
        """Dataset loads mock annotations correctly."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        assert len(ds) == mock_memotion2_dir["num_samples"]

    def test_getitem_keys(self, mock_memotion2_dir):
        """__getitem__ returns all required keys."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        sample = ds[0]
        expected_keys = {"image", "input_ids", "attention_mask",
                         "answer_idx", "question_type", "sample_id"}
        assert set(sample.keys()) == expected_keys

    def test_image_shape(self, mock_memotion2_dir):
        """Image tensor has correct shape."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, 32, 32)

    def test_question_type_is_sentiment(self, mock_memotion2_dir):
        """All samples have question_type = 'sentiment'."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        for i in range(len(ds)):
            assert ds[i]["question_type"] == "sentiment"

    def test_label_mapping(self, mock_memotion2_dir):
        """Sentiment labels are correctly mapped to indices."""
        label_map = build_memotion2_label_map()
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            label_map=label_map,
            image_size=32,
        )
        # First sample is "negative" → index 1
        assert ds[0]["answer_idx"] == label_map["negative"]
        # Second sample is "positive" → index 0
        assert ds[1]["answer_idx"] == label_map["positive"]
        # Third sample is "neutral" → index 2
        assert ds[2]["answer_idx"] == label_map["neutral"]

    def test_custom_label_map(self, mock_memotion2_dir):
        """Custom label map overrides defaults."""
        custom_map = {"positive": 10, "negative": 20, "neutral": 30}
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            label_map=custom_map,
            image_size=32,
        )
        assert ds[0]["answer_idx"] == 20  # negative

    def test_collate_fn_compatibility(self, mock_memotion2_dir):
        """Memotion2Dataset samples work with vqa_collate_fn."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        batch = [ds[i] for i in range(3)]
        collated = vqa_collate_fn(batch)

        assert collated["images"].shape == (3, 3, 32, 32)
        assert collated["answer_idx"].shape == (3,)
        assert len(collated["question_types"]) == 3
        assert len(collated["sample_ids"]) == 3

    def test_empty_text(self, mock_memotion2_dir):
        """Sample with empty OCR text doesn't crash."""
        ds = Memotion2Dataset(
            annotations_path=mock_memotion2_dir["annotations_path"],
            image_dir=mock_memotion2_dir["image_dir"],
            image_size=32,
        )
        # Fifth sample has empty text
        sample = ds[4]
        assert sample["input_ids"].shape[0] > 0  # Still has tokens (pad)
