#!/usr/bin/env python3
"""
Ablation study for TTT stabilization techniques.

Fixed: K=1 (or K=2), masked_patch objective.
Vary:
    (a) TTT alone (no stabilization)             — ttt_only
    (b) TTT + consistency regularization          — ttt_consistency
    (c) TTT + mixup anchoring                     — ttt_mixup
    (d) TTT + both                                — ttt_both

Usage on Colab:
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_only
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_consistency
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_mixup
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_both

Saves: results/ablation/{mode}_k{K}.json
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


MODES = {
    "ttt_only": {"use_consistency": False, "use_mixup": False},
    "ttt_consistency": {"use_consistency": True, "use_mixup": False},
    "ttt_mixup": {"use_consistency": False, "use_mixup": True},
    "ttt_both": {"use_consistency": True, "use_mixup": True},
}


def main():
    parser = argparse.ArgumentParser(description="TTT stabilization ablation")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--mode", type=str, required=True, choices=list(MODES.keys()))
    parser.add_argument("--objective", type=str, default="masked_patch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logging("logs")

    mode_config = MODES[args.mode]
    strict_images = config.get("strict_images", True)
    logger.info(f"Ablation: mode={args.mode}, K={args.k}, {args.objective}")
    logger.info(f"  consistency={mode_config['use_consistency']}, mixup={mode_config['use_mixup']}")
    logger.info(f"Device: {device}")
    logger.info(f"Split: {args.split}")

    if args.batch_size != 1:
        raise ValueError(
            "Ablation requires --batch-size 1 for per-sample adaptation semantics."
        )

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Create TTT adapter with the specified stabilization mode
    ttt_adapter = TTTAdapter(
        model, config,
        objective=args.objective,
        k_steps=args.k,
        use_consistency=mode_config["use_consistency"],
        use_mixup=mode_config["use_mixup"],
    )

    # Load dataset
    data_dir = config.get("data_dir", "data/")
    answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
    split_year = "train2014" if args.split == "train" else "val2014"
    val_dataset = VQADataset(
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
    total_ttt_loss = 0.0
    t0 = time.time()

    for batch in tqdm(val_loader, desc=f"Ablation {args.mode} K={args.k}"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)

        # Encode
        visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        # TTT adaptation
        logits, ttt_loss = ttt_adapter.adapt_and_predict(
            images, visual_tokens, text_tokens, attention_mask
        )

        preds = logits.argmax(dim=-1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)
        total_ttt_loss += ttt_loss

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
    avg_ttt_loss = total_ttt_loss / total if total > 0 else 0.0

    logger.info(f"\nResults: {args.mode}, K={args.k}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Avg TTT loss: {avg_ttt_loss:.4f}")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/total:.2f}s/sample)")

    # Save
    ablation_dir = os.path.join(config.get("results_dir", "results/"), "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    filename = f"{args.mode}_{args.split}_k{args.k}.json"
    save_path = os.path.join(ablation_dir, filename)
    save_json(all_predictions, save_path)
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
