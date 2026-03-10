#!/usr/bin/env python3
"""
The main experiment: sweep K steps × TTT objective.

Usage on Colab:
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 0
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective rotation
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 3 --objective masked_patch

For each configuration:
    1. Load base model
    2. For each val sample: encode → TTT adapt (if K>0) → predict
    3. Save results to results/ttt_predictions/k{K}_{objective}.json

Threshold τ is applied AFTER, during analysis (scripts/02_analyze_results.py).
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
from ttt.data import VQADataset, vqa_collate_fn, load_answer_vocab
from ttt.utils import (
    load_config,
    load_checkpoint,
    save_json,
    setup_logging,
    get_device,
)


def main():
    parser = argparse.ArgumentParser(description="Run TTT sweep")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint")
    parser.add_argument("--k", type=int, required=True, help="Number of TTT steps")
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for TTT (typically 1 for per-sample adaptation)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logging("logs")

    k = args.k
    objective = args.objective
    data_dir = config.get("data_dir", "data/")

    logger.info(f"TTT Sweep: K={k}, objective={objective}")
    logger.info(f"Device: {device}")

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

    # Load vocabulary and dataset
    answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
    val_dataset = VQADataset(
        questions_path=os.path.join(data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        annotations_path=os.path.join(data_dir, "v2_mscoco_val2014_annotations.json"),
        image_dir=os.path.join(data_dir, "val2014"),
        answer_vocab=answer_vocab,
        max_question_length=config.get("max_question_length", 20),
        image_size=config.get("image_size", 224),
        split="val",
    )

    if args.max_samples:
        val_dataset.samples = val_dataset.samples[:args.max_samples]

    logger.info(f"Val samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Run evaluation
    all_predictions = []
    correct = 0
    total = 0
    t0 = time.time()

    for batch in tqdm(val_loader, desc=f"TTT K={k} ({objective})"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)

        # Encode (frozen)
        visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        if k == 0:
            # No TTT — just predict
            with torch.no_grad():
                logits, z = model.fuse_and_predict(visual_tokens, text_tokens, attention_mask)
            ttt_loss = 0.0
        else:
            # TTT adaptation
            logits, ttt_loss = ttt_adapter.adapt_and_predict(
                images, visual_tokens, text_tokens, attention_mask
            )

        preds = logits.argmax(dim=-1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)

        # Record predictions
        for i in range(answers.size(0)):
            all_predictions.append({
                "sample_id": batch["sample_ids"][i],
                "prediction": preds[i].item(),
                "ground_truth": answers[i].item(),
                "question_type": batch["question_types"][i],
                "ttt_loss": float(ttt_loss),
            })

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults: K={k}, {objective}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/total:.2f}s/sample)")

    # Save results
    results_dir = os.path.join(config.get("results_dir", "results/"), "ttt_predictions")
    os.makedirs(results_dir, exist_ok=True)

    filename = f"k{k}_{objective}.json" if k > 0 else "k0_baseline.json"
    save_path = os.path.join(results_dir, filename)
    save_json(all_predictions, save_path)
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
