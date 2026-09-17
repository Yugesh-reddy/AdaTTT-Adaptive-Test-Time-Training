"""
Multi-view cached features for Phase 2 shift evaluation.

MEMO predicts on the original (corrupted) image and adapts on 4 AugMix views,
so each condition stores 5 visual rows. Text is the frozen encoder output
(CLIP: width 512) and is *not* passed through the trainable projector — that
stays in the fusion module so it can keep learning, and so a CLIP-LN ablation
is not silently mixed into this cache.

Caches are VM-local. Enabling mmap on torch.load deadlocks on Google Drive
File Provider paths; this module never mmaps and refuses Drive destinations.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


N_VIEWS = 5
N_AUGMIX = 4
ORIGINAL_VIEW = 0
VIEW_LAYOUT = ("original", "augmix_0", "augmix_1", "augmix_2", "augmix_3")

REQUIRED_CACHE_KEYS = (
    "sample_ids",
    "visual_tokens",
    "text_tokens",
    "attention_masks",
    "answer_idx",
    "question_types",
)

# The 58/93/128 GFLOPs ladder from an earlier spec does not match
# AdaptiveRouter.sample_flops under CLIP. Stored on the manifest so reports
# can flag the gap instead of quietly using the old numbers.
LEGACY_LADDER_GFLOPS = {
    "base": 24.8,
    "tent_k1": 58.0,
    "memo2_k1": 93.0,
    "memo4_k1": 128.0,
}

_DRIVE_MARKERS = (
    "googledrive",
    "google drive",
    "/content/drive",
    "com.google.drivefs",
    "my drive",
)


class CachePathError(ValueError):
    """Raised when a Phase 2 cache path would hit Drive or another forbidden store."""


def is_drive_path(path: str) -> bool:
    resolved = os.path.abspath(os.path.expanduser(path)).lower()
    return any(marker in resolved for marker in _DRIVE_MARKERS)


def assert_cache_path_allowed(path: str) -> str:
    """Phase 2 caches stay on VM-local disk. Drive mmap deadlocks; Mac pulls materialize.

    Returns:
        The expanded path (not necessarily existing yet).
    """
    expanded = os.path.abspath(os.path.expanduser(path))
    if is_drive_path(expanded):
        raise CachePathError(
            f"Phase 2 feature caches must not live on Google Drive ({expanded}). "
            "Build them on VM-local disk, run the methods, write small results, "
            "and delete the cache. mmap on a Drive path deadlocks; every Drive "
            "byte read on the Mac is copied onto local disk."
        )
    return expanded


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(cache_path: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    """Sidecar describing one 5-view condition cache."""
    n = len(blob["sample_ids"])
    return {
        "n_views": N_VIEWS,
        "n_augmix": N_AUGMIX,
        "view_layout": list(VIEW_LAYOUT),
        "n_samples": n,
        "corruption": blob.get("corruption"),
        "severity": blob.get("severity"),
        "dtype": str(blob.get("dtype", blob["visual_tokens"].dtype)),
        "text_projected": False,
        "vision_ln_adapted": False,
        "clip_ln_ablation": "skipped",
        "seed_fn": "crc32(sample_id:view:salt)",
        "sha256": _sha256_file(cache_path),
        "flops_accounting": "AdaptiveRouter.sample_flops",
        "legacy_ladder_gflops": dict(LEGACY_LADDER_GFLOPS),
        "legacy_ladder_note": (
            "The 58/93/128 GFLOPs ladder in the original spec does not match "
            "AdaptiveRouter.sample_flops under CLIP. Report sample_flops."
        ),
    }


def write_shift_cache(path: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    """Write a 5-view fp16 cache and its manifest. Never mmaps. Never Drive."""
    path = assert_cache_path_allowed(path)
    visual = blob["visual_tokens"]
    n_views = int(blob.get("n_views", visual.shape[1]))
    if n_views != N_VIEWS or visual.shape[1] != N_VIEWS:
        raise ValueError(
            f"Phase 2 caches store {N_VIEWS} views/condition "
            f"(original + {N_AUGMIX} AugMix), got {n_views}."
        )
    payload = dict(blob)
    payload["n_views"] = N_VIEWS
    payload["view_layout"] = list(VIEW_LAYOUT)
    payload["text_projected"] = False
    payload["clip_ln_ablation"] = "skipped"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    manifest = build_manifest(path, payload)
    manifest_path = os.path.splitext(path)[0] + ".manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class MultiViewCachedFeaturesDataset(Dataset):
    """Eager (non-mmap) loader for a Phase 2 5-view condition cache."""

    def __init__(self, features_path: str, mmap: bool = False):
        if mmap:
            raise ValueError(
                "mmap is forbidden for Phase 2 caches (deadlocks on Drive). "
                "Load eagerly from VM-local disk."
            )
        path = assert_cache_path_allowed(features_path)
        blob = _torch_load(path)
        for key in REQUIRED_CACHE_KEYS:
            if key not in blob:
                raise ValueError(
                    f"Shift cache at {path} is missing required key '{key}'. "
                    "Regenerate with gpu/precompute_shift_features.py."
                )
        if blob["visual_tokens"].ndim != 4 or blob["visual_tokens"].shape[1] != N_VIEWS:
            raise ValueError(
                f"Expected visual_tokens (N, {N_VIEWS}, tokens, dim); "
                f"got {tuple(blob['visual_tokens'].shape)}."
            )
        if blob.get("text_projected"):
            raise ValueError(
                "This cache baked in the text projector. Phase 2 stores raw CLIP "
                "text (width 512) so the projector stays trainable."
            )
        self._blob = blob
        self.sample_ids: List[str] = list(blob["sample_ids"])
        self.question_types: List[str] = list(blob["question_types"])
        self.text_projected = False
        self.n_views = N_VIEWS

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        blob = self._blob
        sample: Dict[str, Any] = {
            "visual_tokens": blob["visual_tokens"][idx].float(),
            "text_tokens": blob["text_tokens"][idx].float(),
            "attention_mask": blob["attention_masks"][idx].to(torch.long),
            "answer_idx": int(blob["answer_idx"][idx]),
            "question_type": self.question_types[idx],
            "sample_id": self.sample_ids[idx],
        }
        if "answer_scores" in blob:
            sample["answer_scores"] = blob["answer_scores"][idx].float()
        return sample


def multiview_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a multi-view cache batch. Images are not in the cache."""
    result: Dict[str, Any] = {
        "visual_tokens": torch.stack([s["visual_tokens"] for s in batch]),
        "text_tokens": torch.stack([s["text_tokens"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "answer_idx": torch.tensor([s["answer_idx"] for s in batch], dtype=torch.long),
        "question_types": [s["question_type"] for s in batch],
        "sample_ids": [s["sample_id"] for s in batch],
    }
    if "answer_scores" in batch[0]:
        result["answer_scores"] = torch.stack([s["answer_scores"] for s in batch])
    return result
