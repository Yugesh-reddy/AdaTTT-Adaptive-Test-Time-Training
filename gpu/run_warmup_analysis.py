#!/usr/bin/env python3
"""
TTT warm-up cost analysis: cumulative vs per-sample restore.

Tests whether accumulating TTT adaptations across samples (no restore)
improves or degrades accuracy compared to the standard per-sample restore.

Modes:
    - "cumulative": Process N samples sequentially WITHOUT restoring params
    - "restore" (baseline): Normal per-sample restore (standard behavior)

Tracks per-sample: accuracy, ttt_loss, param drift (L2 from anchor).

Usage on Colab:
    !python gpu/run_warmup_analysis.py --checkpoint checkpoints/base/best.pt --k 1 --mode cumulative
    !python gpu/run_warmup_analysis.py --checkpoint checkpoints/base/best.pt --k 1 --mode restore

Saves: results/warmup_analysis.json
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
    set_seed,
)


def compute_param_drift(model, anchor_state, adapt_modules):
    """Compute L2 distance between current params and anchor."""
    named_params = model.get_ttt_params_named(
        adapt_modules=adapt_modules, include_auxiliary=True,
    )
    total_drift = 0.0
    for name, param in named_params:
        if name in anchor_state:
            diff = param.data - anchor_state[name]
            total_drift += diff.norm().item() ** 2
    return total_drift ** 0.5


def main():
    parser = argparse.ArgumentParser(description="TTT warmup analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["cumulative", "restore"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    logger.info(f"Warmup analysis: mode={args.mode}, K={args.k}, max_samples={args.max_samples}")
    logger.info(f"Device: {device}")

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Save initial anchor state
    adapt_modules = list(config.get("ttt_adapt_modules", ["fusion", "prediction_head"]))
    obj_head = "mask_proj" if args.objective == "masked_patch" else "rotation_head"
    all_modules = adapt_modules + [obj_head]

    named_params = model.get_ttt_params_named(
        adapt_modules=all_modules, include_auxiliary=True,
    )
    anchor_state = {name: param.data.clone() for name, param in named_params}

    # Create TTT adapter
    ttt_adapter = TTTAdapter(
        model, config, objective=args.objective, k_steps=args.k,
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
        strict_images=config.get("strict_images", True),
    )

    if args.max_samples:
        val_dataset.samples = val_dataset.samples[:args.max_samples]

    logger.info(f"Dataset samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Process samples
    per_sample_results = []
    running_correct = 0
    t0 = time.time()

    for idx, batch in enumerate(tqdm(val_loader, desc=f"Warmup {args.mode}")):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            visual_tokens, text_tokens = model.encode(images, input_ids, attention_mask)

        if args.mode == "cumulative":
            # Adapt WITHOUT restoring — params drift over time
            # Manual TTT: save state, adapt, predict, conditionally restore
            ttt_modules = ttt_adapter._ttt_module_names()
            ttt_named = model.get_ttt_params_named(
                adapt_modules=ttt_modules, include_auxiliary=True,
            )

            # Create optimizer
            ttt_optimizer = torch.optim.Adam(
                [p for _, p in ttt_named], lr=ttt_adapter.lr,
            )

            # K gradient steps
            final_loss = 0.0
            for step in range(args.k):
                ttt_optimizer.zero_grad()
                if args.objective == "masked_patch":
                    loss = ttt_adapter.masked_patch_loss(
                        visual_tokens, text_tokens, attention_mask,
                    )
                else:
                    loss = ttt_adapter.rotation_loss(
                        images, text_tokens, attention_mask,
                    )
                loss.backward()
                if ttt_adapter.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in ttt_named], max_norm=ttt_adapter.grad_clip,
                    )
                ttt_optimizer.step()
                final_loss = loss.item()

            # Predict with adapted (drifted) params
            with torch.no_grad():
                logits, z = model.fuse_and_predict(
                    visual_tokens, text_tokens, attention_mask,
                )
            pred = logits.argmax(dim=-1)
            ttt_loss = final_loss
            # NO restore in cumulative mode

        else:  # restore mode (standard)
            logits, ttt_loss = ttt_adapter.adapt_and_predict(
                images, visual_tokens, text_tokens, attention_mask,
            )
            pred = logits.argmax(dim=-1)

        is_correct = (pred == answers).item()
        running_correct += is_correct

        # Compute drift from initial anchor
        drift = compute_param_drift(model, anchor_state, all_modules)

        per_sample_results.append({
            "sample_idx": idx,
            "sample_id": batch["sample_ids"][0],
            "prediction": pred.item(),
            "ground_truth": answers[0].item(),
            "correct": is_correct,
            "running_accuracy": running_correct / (idx + 1),
            "ttt_loss": float(ttt_loss),
            "param_drift_l2": drift,
        })

    elapsed = time.time() - t0
    final_accuracy = running_correct / len(per_sample_results) if per_sample_results else 0.0
    logger.info(f"\nResults: mode={args.mode}, K={args.k}")
    logger.info(f"  Final accuracy: {final_accuracy*100:.2f}%")
    logger.info(f"  Time: {elapsed:.0f}s")

    # Save
    results_dir = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "warmup_analysis.json")

    # Merge with existing
    existing = {}
    if os.path.exists(save_path):
        from ttt.utils import load_json
        existing = load_json(save_path)

    existing[args.mode] = {
        "k": args.k,
        "objective": args.objective,
        "num_samples": len(per_sample_results),
        "final_accuracy": final_accuracy,
        "per_sample": per_sample_results,
    }
    save_json(existing, save_path)
    logger.info(f"  Saved: {save_path} (key={args.mode})")


if __name__ == "__main__":
    main()
