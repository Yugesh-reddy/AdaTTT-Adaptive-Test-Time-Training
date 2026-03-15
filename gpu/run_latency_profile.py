#!/usr/bin/env python3
"""
Profile per-stage latency of the AdaTTT pipeline.

Measures P50/P95 latency for: image preprocessing, ViT encoding,
BERT encoding, fusion+predict, and TTT adaptation.

Usage on Colab:
    !python gpu/run_latency_profile.py --checkpoint checkpoints/base/best.pt --k 1
    !python gpu/run_latency_profile.py --checkpoint checkpoints/base/best.pt --k 0

Saves: results/latency_profiles.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from ttt.models import FullVQAModel
from ttt.ttt_loop import TTTAdapter
from ttt.latency import LatencyProfiler
from ttt.data import VQADataset, get_image_transform, load_answer_vocab
from ttt.utils import (
    load_config,
    load_checkpoint,
    save_json,
    setup_logging,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Profile AdaTTT latency")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=1, help="TTT gradient steps")
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    logger = setup_logging("logs")

    num_samples = args.num_samples or config.get("latency_num_samples", 100)
    warmup_runs = config.get("latency_warmup_runs", 5)

    logger.info(f"Latency profile: K={args.k}, {args.objective}, n={num_samples}")
    logger.info(f"Device: {device}")

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Create TTT adapter
    ttt_adapter = TTTAdapter(
        model, config, objective=args.objective, k_steps=args.k,
    )

    # Create profiler
    image_transform = get_image_transform(config.get("image_size", 224))
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        config.get("text_encoder", "bert-base-uncased")
    )

    profiler = LatencyProfiler(model, ttt_adapter, image_transform, tokenizer, config)

    # Load dataset
    data_dir = config.get("data_dir", "data/")
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
        strict_images=config.get("strict_images", True),
    )

    # Create a wrapper that provides PIL images + questions
    class PILDatasetWrapper:
        def __init__(self, vqa_dataset):
            self.dataset = vqa_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample_info = self.dataset.samples[idx]
            image_pil = Image.open(sample_info["image_path"]).convert("RGB")
            return {
                "image_pil": image_pil,
                "question": sample_info["question"],
            }

    pil_dataset = PILDatasetWrapper(dataset)

    # Profile
    summary = profiler.profile_batch(
        pil_dataset,
        n_samples=num_samples,
        warmup_runs=warmup_runs,
        threshold=args.threshold,
        k_steps=args.k,
    )

    # Log results
    for stage in ["image_preprocess_ms", "vision_encode_ms", "text_encode_ms",
                  "fusion_predict_ms", "ttt_adaptation_ms", "total_ms"]:
        stats = summary[stage]
        logger.info(f"  {stage}: P50={stats['p50']:.1f}ms P95={stats['p95']:.1f}ms")
    logger.info(f"  TTT trigger rate: {summary['ttt_trigger_rate']*100:.1f}%")

    # Save
    results_dir = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "latency_profiles.json")

    # Merge with existing profiles if present
    existing = {}
    if os.path.exists(save_path):
        from ttt.utils import load_json
        existing = load_json(save_path)

    key = f"k{args.k}_{args.objective}"
    existing[key] = summary
    save_json(existing, save_path)
    logger.info(f"  Saved: {save_path} (key={key})")


if __name__ == "__main__":
    main()
