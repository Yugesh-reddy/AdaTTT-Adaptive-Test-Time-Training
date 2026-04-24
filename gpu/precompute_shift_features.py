#!/usr/bin/env python3
"""
Precompute a 5-view CLIP feature cache for one Phase 2 shift condition.

Views per sample: original (already corrupted) + 4 AugMix. Text is raw CLIP
output (width 512); the fusion text projector is deliberately not baked in.
CLIP-LN is skipped — adapting vision LayerNorms invalidates this cache.

Run on the VM against local disk. Do not write to Drive. Do not pull the
~12 GB cache to the Mac. This script never allocates a Colab VM.

Usage (on the VM, after the first A100 session starts — not from this machine):
    python gpu/precompute_shift_features.py \\
        --corruption gaussian_blur --severity 3 \\
        --output /content/phase2_cache/blur_s3.pt \\
        --subset data/eval_subset_8k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Sequence

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.shift_cache import (  # noqa: E402
    N_AUGMIX,
    N_VIEWS,
    VIEW_LAYOUT,
    assert_cache_path_allowed,
    write_shift_cache,
)

# ttt.shifts.py is still uncommitted on this machine (item 1). Import lazily so
# the rest of the Phase 2 stack can be tested and reviewed without it.
corrupt = None  # type: ignore[assignment]
augmix_views = None  # type: ignore[assignment]


def _shift_fns():
    """Resolve corrupt/augmix, honouring test monkeypatches on this module."""
    global corrupt, augmix_views
    if corrupt is None or augmix_views is None:
        from ttt.shifts import augmix_views as _aug, corrupt as _corrupt
        if corrupt is None:
            corrupt = _corrupt
        if augmix_views is None:
            augmix_views = _aug
    return corrupt, augmix_views


def reject_clip_ln_ablation(enabled: bool) -> None:
    """CLIP-LN is a priced ablation or a skip — not the main cache path."""
    if enabled:
        raise NotImplementedError(
            "CLIP-LN adapts LayerNorms inside the vision tower, which changes "
            "the cached visual features and invalidates the vision cache. "
            "It needs on-the-fly encoding at ~4× cost. Skip it, or give it a "
            "separate budget line; do not write a CLIP-LN cache with this script."
        )


def resolve_output_path(path: str) -> str:
    return assert_cache_path_allowed(path)


def load_subset_ids(path: str) -> List[str]:
    """Read a frozen subset JSON. Never rewrite it."""
    with open(path, "r") as fh:
        blob = json.load(fh)
    ids = blob["sample_ids"]
    return [str(x) for x in ids]


def filter_to_subset(
    samples: Sequence[Dict[str, Any]], ids: Sequence[str]
) -> List[Dict[str, Any]]:
    """Keep subset ids in subset order (first occurrence in `samples`)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        sid = str(sample["sample_id"])
        by_id.setdefault(sid, sample)
    return [by_id[i] for i in ids if i in by_id]


def assemble_views(
    image: Image.Image,
    sample_id: str,
    kind: str,
    severity: int,
    transform: Callable[[Image.Image], torch.Tensor],
    n_aug: int = N_AUGMIX,
) -> torch.Tensor:
    """(5, 3, H, W): original corrupted image, then 4 deterministic AugMix views."""
    _corrupt, _augmix = _shift_fns()
    shifted = _corrupt(image, kind, severity, sample_id)
    extras = _augmix(shifted, n_views=n_aug, sample_id=sample_id)
    tensors = [transform(shifted)] + [transform(view) for view in extras]
    if len(tensors) != N_VIEWS:
        raise ValueError(f"Expected {N_VIEWS} views, assembled {len(tensors)}")
    return torch.stack(tensors, dim=0)


def encode_views(
    model: Any,
    view_images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple:
    """Encode 5 visual views; text is encoded once and reused.

    Args:
        view_images: (5, 3, H, W)
        input_ids / attention_mask: (1, L) or (L,)

    Returns:
        visual_tokens: (5, N_vis, D_vis)
        text_tokens: (L, D_text)  — raw encoder width, not projected
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    n = view_images.shape[0]
    ids = input_ids.expand(n, -1)
    mask = attention_mask.expand(n, -1)
    visual, text = model.encode(view_images, ids, mask)
    return visual, text[0]


def _pil_from_sample(sample: Dict[str, Any]) -> Image.Image:
    if "image_path" in sample:
        return Image.open(sample["image_path"]).convert("RGB")
    image = sample.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError("Sample has neither image_path nor a PIL image")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute 5-view shift features")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--corruption", type=str, required=True,
                        choices=["gaussian_blur", "gaussian_noise"])
    parser.add_argument("--severity", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=str, required=True,
                        help="VM-local .pt path (Drive paths are rejected)")
    parser.add_argument("--subset", type=str, default="data/eval_subset_8k.json",
                        help="Frozen id list; read-only")
    parser.add_argument("--dataset", type=str, default="vqa_v2")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Samples per encode; each sample still stores 5 views")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clip-ln", action="store_true",
                        help="Rejected: CLIP-LN invalidates this cache")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Optional JSON heartbeat (VM probe); never a cache path")
    args = parser.parse_args(argv)

    reject_clip_ln_ablation(args.clip_ln)
    out_path = resolve_output_path(args.output)

    from ttt.data import build_dataset, get_image_transform
    from ttt.models import FullVQAModel
    from ttt.utils import get_device, load_config, set_seed, setup_logging

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    logger = setup_logging("logs")
    device = get_device()
    cache_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    backend = config.get("encoder_backend", "clip")
    transform = get_image_transform(config.get("image_size", 224), backend=backend)

    subset_ids = load_subset_ids(args.subset)
    logger.info("Subset %s: %d ids (read-only)", args.subset, len(subset_ids))

    dataset = build_dataset(config, args.dataset, split=args.split)
    kept = filter_to_subset(dataset.samples, subset_ids)
    if args.max_samples is not None:
        kept = kept[: args.max_samples]
    logger.info("Encoding %d samples, %d views each, %s s%s → %s",
                len(kept), N_VIEWS, args.corruption, args.severity, out_path)

    model = FullVQAModel(config)
    model.load_encoders(config)
    model = model.to(device)
    model.eval()

    visual_rows = []
    text_rows = []
    mask_rows = []
    answer_rows = []
    score_rows = []
    sample_ids = []
    question_types = []

    def _tick(done: int, total: int) -> None:
        if not args.progress_file:
            return
        tmp = args.progress_file + ".tmp"
        os.makedirs(os.path.dirname(args.progress_file) or ".", exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump({
                "stage": "precompute",
                "step": f"precompute {done}/{total}",
                "n": done,
                "n_total": total,
                "done": False,
                "crash": False,
            }, fh)
        os.replace(tmp, args.progress_file)

    id_to_index = {str(s["sample_id"]): i for i, s in enumerate(dataset.samples)}
    total = len(kept)
    _tick(0, total)
    for i_keep, sample_meta in enumerate(kept):
        j = id_to_index[str(sample_meta["sample_id"])]
        live = dataset[j]
        pil = _pil_from_sample(dataset.samples[j])
        views = assemble_views(
            pil,
            sample_id=str(live["sample_id"]),
            kind=args.corruption,
            severity=args.severity,
            transform=transform,
        ).to(device)
        input_ids = live["input_ids"].unsqueeze(0).to(device)
        attention_mask = live["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            vis, text = encode_views(model, views, input_ids, attention_mask)
        visual_rows.append(vis.to(cache_dtype).cpu())
        text_rows.append(text.to(cache_dtype).cpu())
        mask_rows.append(live["attention_mask"].to(torch.bool).cpu())
        answer_rows.append(int(live["answer_idx"]))
        if "answer_scores" in live:
            score_rows.append(live["answer_scores"].cpu())
        sample_ids.append(str(live["sample_id"]))
        question_types.append(live["question_type"])
        if (i_keep + 1) % 50 == 0 or i_keep + 1 == total:
            _tick(i_keep + 1, total)
            logger.info("precompute %d/%d", i_keep + 1, total)

    blob: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "visual_tokens": torch.stack(visual_rows),
        "text_tokens": torch.stack(text_rows),
        "attention_masks": torch.stack(mask_rows),
        "answer_idx": torch.tensor(answer_rows, dtype=torch.long),
        "question_types": question_types,
        "n_views": N_VIEWS,
        "view_layout": list(VIEW_LAYOUT),
        "text_projected": False,
        "corruption": args.corruption,
        "severity": args.severity,
        "dtype": str(cache_dtype),
        "subset": args.subset,
        "dataset": args.dataset,
    }
    if score_rows:
        blob["answer_scores"] = torch.stack(score_rows)
    manifest = write_shift_cache(out_path, blob)
    logger.info("Wrote %s (%d samples, sha256=%s)", out_path, len(sample_ids), manifest["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
