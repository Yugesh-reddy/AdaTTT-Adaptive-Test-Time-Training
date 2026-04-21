#!/usr/bin/env python3
"""
Precompute frozen encoder features (ViT visual tokens + BERT text tokens).

Since ViT-B/16 and BERT-base are FROZEN, their outputs are deterministic.
Running them once and caching the outputs avoids re-encoding the same inputs
during every TTT sweep, adaptive inference run, and ablation experiment.

Expected H100 savings for the VQA-v2 pipeline:
    - TTT sweep (K={0,1,2,3,5} × 2 objectives, 10 runs): ~6 hrs
    - Adaptive inference (5 thresholds):                  ~1.5 hrs
    - Ablations (stabilization + component, 8 runs):      ~2 hrs

Storage (fp16):
    - VQA-v2 val:   ~13 GB (40k samples × 197×768 visual + 20×768 text)
    - VQA-v2 train: ~148 GB — usually skip unless Drive has headroom

Usage on Colab:
    !python gpu/precompute_features.py \\
        --split val --output data/features/val_features.pt

    !python gpu/precompute_features.py \\
        --split val --dataset memotion2 \\
        --output data/features/memotion2_val_features.pt
"""

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttt.models import FullVQAModel
from ttt.data import (
    VQADataset, Memotion2Dataset, vqa_collate_fn,
    load_answer_vocab,
)
from ttt.utils import (
    load_config,
    setup_logging,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute frozen encoder features")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: vqa_v2, vizwiz, or memotion2")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Storage dtype (float16 halves disk size with negligible accuracy impact)")
    parser.add_argument("--shard-size", type=int, default=50000,
                        help="Max samples per shard file. Large datasets are split into "
                             "multiple shards to avoid OOM on Colab (214K val @ fp16 "
                             "needs ~67 GB RAM without sharding).")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    data_dir = config.get("data_dir", "data/")
    dataset_name = args.dataset or config.get("dataset", "vqa_v2")
    is_memotion2 = dataset_name == "memotion2"
    strict_images = config.get("strict_images", True)

    if is_memotion2:
        config["num_answers"] = config.get("memotion2_num_classes", 3)

    cache_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    logger.info(f"Precompute features: dataset={dataset_name}, split={args.split}")
    logger.info(f"Device: {device}, AMP: {use_amp}, storage dtype: {args.dtype}")
    logger.info(f"Output: {args.output}")

    model = FullVQAModel(config)
    model.load_encoders(config)
    model = model.to(device)
    model.eval()

    if is_memotion2:
        memo_dir = config.get("memotion2_data_dir", os.path.join(data_dir, "memotion2"))
        dataset = Memotion2Dataset(
            annotations_path=os.path.join(memo_dir, f"{args.split}.json"),
            image_dir=os.path.join(memo_dir, "images"),
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            strict_images=strict_images,
        )
    else:
        answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
        split_year = "train2014" if args.split == "train" else "val2014"
        dataset = VQADataset(
            questions_path=os.path.join(data_dir, f"v2_OpenEnded_mscoco_{split_year}_questions.json"),
            annotations_path=os.path.join(data_dir, f"v2_mscoco_{split_year}_annotations.json"),
            image_dir=os.path.join(data_dir, split_year),
            answer_vocab=answer_vocab,
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            split="train" if args.split == "train" else "val",
            strict_images=strict_images,
        )

    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    n_samples = len(dataset)
    max_q_len = config.get("max_question_length", 20)
    shard_size = args.shard_size
    n_shards = (n_samples + shard_size - 1) // shard_size
    logger.info(f"Samples: {n_samples}, max_question_length: {max_q_len}")
    logger.info(f"Shard size: {shard_size}, num shards: {n_shards}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # --- Shard-aware encoding ---------------------------------------------------
    # Instead of pre-allocating one giant tensor for all samples (which OOMs on
    # Colab for 214K val samples @ ~67 GB), we accumulate into a shard-sized
    # buffer and flush to disk whenever it fills up.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stem, ext = os.path.splitext(args.output)

    cur_shard_visual = []
    cur_shard_text = []
    cur_shard_mask = []
    cur_shard_answer = []
    cur_shard_ids: List[str] = []
    cur_shard_qtypes: List[str] = []
    global_idx = 0
    shard_idx = 0
    shard_paths: List[str] = []

    def _flush_shard():
        nonlocal shard_idx
        shard_n = len(cur_shard_ids)
        if shard_n == 0:
            return
        if n_shards == 1:
            path = args.output  # single-file when data fits in one shard
        else:
            path = f"{stem}_shard{shard_idx}{ext}"
        torch.save({
            "sample_ids": cur_shard_ids[:],
            "visual_tokens": torch.cat(cur_shard_visual),
            "text_tokens": torch.cat(cur_shard_text),
            "attention_masks": torch.cat(cur_shard_mask),
            "answer_idx": torch.cat(cur_shard_answer),
            "question_types": cur_shard_qtypes[:],
            "dataset": dataset_name,
            "split": args.split,
            "num_samples": shard_n,
            "max_question_length": max_q_len,
            "dtype": str(cache_dtype),
        }, path)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"  Shard {shard_idx}: {shard_n} samples → {path} ({size_mb:.0f} MB)")
        shard_paths.append(path)
        # Free memory
        cur_shard_visual.clear()
        cur_shard_text.clear()
        cur_shard_mask.clear()
        cur_shard_answer.clear()
        cur_shard_ids.clear()
        cur_shard_qtypes.clear()
        shard_idx += 1

    t0 = time.time()
    for batch in tqdm(loader, desc="Encoding"):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        cur_shard_visual.append(visual_tokens.to(cache_dtype).cpu())
        cur_shard_text.append(text_tokens.to(cache_dtype).cpu())
        cur_shard_mask.append(attention_mask.to(torch.bool).cpu())
        cur_shard_answer.append(batch["answer_idx"].cpu())
        cur_shard_ids.extend(batch["sample_ids"])
        cur_shard_qtypes.extend(batch["question_types"])
        global_idx += visual_tokens.size(0)

        # Flush when shard is full
        if len(cur_shard_ids) >= shard_size:
            _flush_shard()

    # Flush remaining samples
    _flush_shard()

    elapsed = time.time() - t0
    logger.info(f"Encoded {n_samples} samples in {elapsed:.0f}s "
                f"({elapsed / max(n_samples, 1) * 1000:.1f}ms/sample)")
    logger.info(f"Saved {len(shard_paths)} shard(s): {shard_paths}")


if __name__ == "__main__":
    main()

