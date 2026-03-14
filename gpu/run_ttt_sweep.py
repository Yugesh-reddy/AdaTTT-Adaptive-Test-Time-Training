#!/usr/bin/env python3
"""
The main experiment: sweep K steps × TTT objective.

Usage on Colab:
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 0 --split train
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective rotation
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 3 --objective masked_patch
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --dataset memotion2

For each configuration:
    1. Load base model
    2. For each val sample: encode → TTT adapt (if K>0) → predict
    3. Save results to results/ttt_predictions/k{K}_{objective}.json

Threshold τ is applied AFTER, during analysis (scripts/02_analyze_results.py).

Optimizations:
    - Encoding is batched (--encode-batch-size, default 64) even though TTT is per-sample
    - Mixed precision (AMP) for encoding on CUDA devices
    - Incremental saving (--save-every, default 1000) for Colab crash resilience
    - Resume support (--resume) to skip already-processed samples
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ttt.models import FullVQAModel
from ttt.ttt_loop import TTTAdapter
from ttt.data import (
    VQADataset, Memotion2Dataset, vqa_collate_fn,
    load_answer_vocab, build_memotion2_label_map,
)
from ttt.utils import (
    load_config,
    load_checkpoint,
    load_json,
    save_json,
    setup_logging,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Run TTT sweep")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint")
    parser.add_argument("--k", type=int, required=True, help="Number of TTT steps")
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--encode-batch-size", type=int, default=64,
                        help="Batch size for encoding (ViT+BERT). TTT is always per-sample.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate. Use train for gate-label generation.",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: vqa_v2, vizwiz, or memotion2 (overrides config)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save predictions incrementally every N samples (0 to disable)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results file")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    k = args.k
    objective = args.objective
    split = args.split
    data_dir = config.get("data_dir", "data/")
    dataset_name = args.dataset or config.get("dataset", "vqa_v2")
    is_memotion2 = dataset_name == "memotion2"
    strict_images = config.get("strict_images", True)

    # Override num_answers for Memotion2
    if is_memotion2:
        config["num_answers"] = config.get("memotion2_num_classes", 3)

    logger.info(f"TTT Sweep: K={k}, objective={objective}, dataset={dataset_name}")
    logger.info(f"Device: {device}, AMP: {use_amp}")
    logger.info(f"Split: {split}, encode_batch_size: {args.encode_batch_size}")

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Create TTT adapter
    ttt_adapter = TTTAdapter(
        model, config, objective=objective, k_steps=k
    )

    # Load dataset
    if is_memotion2:
        memo_dir = config.get("memotion2_data_dir", os.path.join(data_dir, "memotion2"))
        val_dataset = Memotion2Dataset(
            annotations_path=os.path.join(memo_dir, f"{split}.json"),
            image_dir=os.path.join(memo_dir, "images"),
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            strict_images=strict_images,
        )
    else:
        answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
        split_year = "train2014" if split == "train" else "val2014"
        val_dataset = VQADataset(
            questions_path=os.path.join(data_dir, f"v2_OpenEnded_mscoco_{split_year}_questions.json"),
            annotations_path=os.path.join(data_dir, f"v2_mscoco_{split_year}_annotations.json"),
            image_dir=os.path.join(data_dir, split_year),
            answer_vocab=answer_vocab,
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            split="train" if split == "train" else "val",
            strict_images=strict_images,
        )

    if args.max_samples:
        val_dataset.samples = val_dataset.samples[:args.max_samples]

    logger.info(f"Dataset samples: {len(val_dataset)}")

    # Determine save path
    if is_memotion2:
        results_dir = os.path.join(config.get("results_dir", "results/"), "memotion2", split)
    else:
        results_dir = os.path.join(config.get("results_dir", "results/"), "ttt_predictions", split)
    os.makedirs(results_dir, exist_ok=True)

    filename = f"k{k}_{objective}.json" if k > 0 else "k0_baseline.json"
    save_path = os.path.join(results_dir, filename)

    # Resume support: load existing partial results
    all_predictions = []
    processed_ids = set()
    if args.resume and os.path.exists(save_path):
        all_predictions = load_json(save_path)
        processed_ids = {p["sample_id"] for p in all_predictions}
        logger.info(f"Resuming: {len(processed_ids)} samples already processed")

    # Use larger batch size for encoding; TTT is per-sample within each batch
    val_loader = DataLoader(
        val_dataset, batch_size=args.encode_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Run evaluation
    correct = sum(1 for p in all_predictions if p["prediction"] == p["ground_truth"])
    total = len(all_predictions)
    new_samples = 0
    t0 = time.time()

    for batch in tqdm(val_loader, desc=f"TTT K={k} ({objective})"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)
        B = images.shape[0]

        # Batch encode with AMP (fast, even for large batches)
        with torch.cuda.amp.autocast(enabled=use_amp):
            visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        if k == 0:
            # No TTT — batch predict (much faster than per-sample)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, z = model.fuse_and_predict(
                        visual_tokens, text_tokens, attention_mask
                    )
            preds = logits.argmax(dim=-1)

            for i in range(B):
                sid = batch["sample_ids"][i]
                if sid in processed_ids:
                    continue
                all_predictions.append({
                    "sample_id": sid,
                    "prediction": preds[i].item(),
                    "ground_truth": answers[i].item(),
                    "question_type": batch["question_types"][i],
                    "ttt_loss": 0.0,
                })
                correct += (preds[i] == answers[i]).item()
                total += 1
                new_samples += 1
        else:
            # TTT per-sample within the batch (encoding already done)
            for i in range(B):
                sid = batch["sample_ids"][i]
                if sid in processed_ids:
                    continue

                sample_logits, ttt_loss = ttt_adapter.adapt_and_predict(
                    images[i:i + 1],
                    visual_tokens[i:i + 1],
                    text_tokens[i:i + 1],
                    attention_mask[i:i + 1],
                )
                pred = sample_logits.argmax(dim=-1)
                all_predictions.append({
                    "sample_id": sid,
                    "prediction": pred.item(),
                    "ground_truth": answers[i].item(),
                    "question_type": batch["question_types"][i],
                    "ttt_loss": float(ttt_loss),
                })
                correct += (pred == answers[i]).item()
                total += 1
                new_samples += 1

        # Incremental save
        if args.save_every > 0 and new_samples > 0 and new_samples % args.save_every < B:
            save_json(all_predictions, save_path)
            logger.info(f"  [checkpoint] Saved {len(all_predictions)} predictions")

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults: K={k}, {objective}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Total samples: {total} ({new_samples} new)")
    if new_samples > 0:
        logger.info(f"  Time: {elapsed:.0f}s ({elapsed/new_samples:.2f}s/new sample)")

    # Final save
    save_json(all_predictions, save_path)
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
