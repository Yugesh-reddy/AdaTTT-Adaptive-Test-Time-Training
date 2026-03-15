#!/usr/bin/env python3
"""
Efficient single-pass gate threshold sweep.

Instead of running run_inference.py N times (once per threshold),
this script runs TTT on ALL samples once, saves both base and TTT logits,
then applies each threshold post-hoc.

Usage on Colab:
    !python gpu/run_gate_sweep.py --checkpoint checkpoints/base/best.pt --k 1

Saves: results/gate_sweep.json
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
from ttt.gate import AdaptiveRouter
from ttt.data import VQADataset, vqa_collate_fn, load_answer_vocab
from ttt.utils import (
    load_config,
    load_checkpoint,
    save_json,
    setup_logging,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Single-pass gate threshold sweep")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gate-checkpoint", type=str, default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--thresholds", type=str,
                        default="0.1,0.3,0.5,0.7,0.8,0.9,0.95,1.0",
                        help="Comma-separated thresholds to sweep")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    logger.info(f"Gate sweep: K={args.k}, {args.objective}, thresholds={thresholds}")
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.checkpoint)

    if args.gate_checkpoint:
        gate_ckpt = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)
        model.gate.load_state_dict(gate_ckpt["gate"])
        logger.info(f"Loaded refined gate: {args.gate_checkpoint}")

    model = model.to(device)
    model.eval()

    # Create TTT adapter
    ttt_adapter = TTTAdapter(model, config, objective=args.objective, k_steps=args.k)

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
        strict_images=config.get("strict_images", True),
    )

    if args.max_samples:
        val_dataset.samples = val_dataset.samples[:args.max_samples]

    logger.info(f"Dataset samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=args.encode_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Phase 1: Collect base logits, TTT logits, and confidences for all samples
    all_samples = []
    t0 = time.time()

    for batch in tqdm(val_loader, desc=f"Gate sweep K={args.k}"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)
        B = images.shape[0]

        with torch.amp.autocast("cuda", enabled=use_amp):
            visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        # Base predictions (batch)
        with torch.no_grad():
            z = model.fusion(visual_tokens.float(), text_tokens.float(), attention_mask)
            confidences = model.gate(z).squeeze(-1)
            base_logits = model.prediction_head(z)
        base_preds = base_logits.argmax(dim=-1)

        # TTT predictions (per-sample)
        for i in range(B):
            ttt_logits_i, ttt_loss = ttt_adapter.adapt_and_predict(
                images[i:i + 1],
                visual_tokens[i:i + 1],
                text_tokens[i:i + 1],
                attention_mask[i:i + 1],
            )
            ttt_pred = ttt_logits_i.argmax(dim=-1).item()

            all_samples.append({
                "sample_id": batch["sample_ids"][i],
                "ground_truth": answers[i].item(),
                "question_type": batch["question_types"][i],
                "confidence": confidences[i].item(),
                "base_pred": base_preds[i].item(),
                "ttt_pred": ttt_pred,
            })

    elapsed = time.time() - t0
    logger.info(f"Phase 1 done: {len(all_samples)} samples in {elapsed:.0f}s")

    # Phase 2: Apply thresholds post-hoc
    results = {}
    skip_flops = AdaptiveRouter.SKIP_FLOPS
    adapt_flops = skip_flops + args.k * AdaptiveRouter.TTT_STEP_FLOPS

    for tau in thresholds:
        correct = 0
        skip_count = 0
        adapt_count = 0

        for s in all_samples:
            if s["confidence"] > tau:
                # SKIP: use base prediction
                pred = s["base_pred"]
                skip_count += 1
            else:
                # ADAPT: use TTT prediction
                pred = s["ttt_pred"]
                adapt_count += 1

            if pred == s["ground_truth"]:
                correct += 1

        n = len(all_samples)
        accuracy = correct / n if n > 0 else 0.0
        skip_rate = skip_count / n if n > 0 else 0.0
        total_flops = skip_count * skip_flops + adapt_count * adapt_flops
        avg_flops = (total_flops / n / 1e9) if n > 0 else 0.0

        results[str(tau)] = {
            "threshold": tau,
            "accuracy": accuracy,
            "skip_rate": skip_rate,
            "skip_count": skip_count,
            "adapt_count": adapt_count,
            "avg_gflops": avg_flops,
        }
        logger.info(
            f"  τ={tau:.2f}: acc={accuracy*100:.2f}%, "
            f"skip={skip_rate*100:.1f}%, GFLOPs={avg_flops:.1f}"
        )

    # Save
    results_dir = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "gate_sweep.json")
    save_json({
        "k": args.k,
        "objective": args.objective,
        "num_samples": len(all_samples),
        "thresholds": results,
        "per_sample": all_samples,
    }, save_path)
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
