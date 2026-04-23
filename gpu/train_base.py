#!/usr/bin/env python3
"""
Train the base VQA model (fusion + gate + prediction head) on VQA-v2, VizWiz, or Memotion2.

Standard supervised training with cross-entropy loss. No TTT during training.
For Memotion2 cross-task evaluation, the same frozen encoders and fusion module
are reused — only the prediction head (θ_d) changes (num_answers → num_classes).

Usage on Colab:
    !python gpu/train_base.py --config config/config.yaml --epochs 15
    !python gpu/train_base.py --config config/config.yaml --dataset memotion2

What this trains:
    - θ_f (FusionModule): cross-modal attention
    - θ_g (ConfidenceGate): auxiliary confidence predictor
    - θ_d (PredictionHead): answer/sentiment classifier

Loss: L = L_vqa + 0.1 * L_gate
    L_vqa  = CrossEntropy(logits, answer_idx)
    L_gate = BCE(gate(z), 1[base_prediction == ground_truth])
"""

import argparse
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ttt.losses import vqa_loss
from ttt.models import FullVQAModel
from ttt.data import (
    build_dataset, build_tokenizer,
    VQADataset, Memotion2Dataset, vqa_collate_fn,
    load_answer_vocab, build_memotion2_label_map,
)
from ttt.utils import (
    load_config,
    save_checkpoint,
    save_json,
    setup_logging,
    get_device,
    count_parameters,
    set_seed,
)


def evaluate(model, val_loader, device, use_amp=False, return_soft=False):
    """Evaluate model on validation set.

    Returns (exact-match accuracy, predictions), plus the official VQA soft
    accuracy as a third element when return_soft=True.
    """
    model.eval()
    correct = 0
    total = 0
    soft_total = 0.0
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer_idx"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, confidence, z = model(images, input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == answers).sum().item()
            total += answers.size(0)

            # Official VQA score of each prediction — min(#humans/3, 1) for the
            # predicted answer — kept as one scalar per sample. The full
            # 3129-wide vector used to be stored per sample instead: ~21 GB of
            # Python floats per epoch on full val, written twice as ~3.5 GB of
            # JSON on every new best. The scalar is all the official metric
            # needs to score this run's own predictions.
            scores = None
            if "answer_scores" in batch:
                scores = batch["answer_scores"].gather(1, preds.cpu().unsqueeze(1)).squeeze(1)
                soft_total += scores.sum().item()

            # Save predictions
            for i in range(answers.size(0)):
                pred_entry = {
                    "sample_id": batch["sample_ids"][i],
                    "prediction": preds[i].item(),
                    "ground_truth": answers[i].item(),
                    "question_type": batch["question_types"][i],
                    "confidence": confidence[i].item(),
                }
                if scores is not None:
                    pred_entry["soft_score"] = scores[i].item()
                all_predictions.append(pred_entry)

    accuracy = correct / total if total > 0 else 0.0
    model.train()
    if return_soft:
        return accuracy, all_predictions, (soft_total / total if total > 0 else 0.0)
    return accuracy, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Train base VQA model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: vqa_v2, vizwiz, or memotion2 (overrides config)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    # Determine dataset
    dataset_name = args.dataset or config.get("dataset", "vqa_v2")
    is_memotion2 = dataset_name == "memotion2"

    # Override num_answers for Memotion2
    if is_memotion2:
        config["num_answers"] = config.get("memotion2_num_classes", 3)

    # Override config with CLI args
    epochs = args.epochs or config.get("train_epochs", 15)
    batch_size = config.get("train_batch_size", 64)
    lr = config.get("train_lr", 1e-4)
    weight_decay = config.get("train_weight_decay", 0.01)
    warmup_ratio = config.get("train_warmup_ratio", 0.1)
    data_dir = config.get("data_dir", "data/")
    strict_images = config.get("strict_images", True)

    logger.info(f"Device: {device}, AMP: {use_amp}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Create model
    model = FullVQAModel(config)
    model.load_encoders(config)
    model = model.to(device)

    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Enable gradient checkpointing if configured
    if config.get("gradient_checkpointing", False):
        if hasattr(model.vit, "gradient_checkpointing_enable"):
            model.vit.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Create datasets through the shared router, which picks the tokenizer and
    # the image normalization from encoder_backend. Building VQADataset inline
    # here fell back to BERT WordPiece ids and ImageNet statistics — a CLIP run
    # would have trained on both without a single error.
    tokenizer = build_tokenizer(config)
    answer_vocab = None
    if not is_memotion2:
        answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
        logger.info(f"Answer vocab size: {len(answer_vocab)}")
    train_dataset = build_dataset(config, dataset_name, split="train",
                                  answer_vocab=answer_vocab, tokenizer=tokenizer)
    val_dataset = build_dataset(config, dataset_name, split="val",
                                answer_vocab=answer_vocab, tokenizer=tokenizer)
    norm_mean = [round(float(m), 4) for m in val_dataset.transform.transforms[-1].mean]
    logger.info(f"Preprocessing: {type(tokenizer).__name__} | image mean {norm_mean} "
                f"| backend {config.get('encoder_backend', 'vit_bert')}")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=vqa_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    grad_accum_steps = max(1, args.grad_accum_steps)

    # Optimizer — only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler should track optimizer steps, not raw micro-batches.
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Resume from checkpoint. Restores the full training state, not only the
    # weights: rebuilding LambdaLR from step 0 would re-run warmup and restart
    # the cosine mid-run, and resetting best_val_acc would let the first
    # post-resume epoch overwrite best.pt with a worse model. Either corrupts
    # the epoch-5-to-8 slope that scripts/06_ceiling_check.py reads.
    start_epoch = 0
    best_val_acc = -1.0  # below any real accuracy, so epoch 1 always writes best.pt
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.resume:
        from ttt.utils import load_checkpoint
        ckpt = load_checkpoint(model, args.resume, load_optimizer=True, optimizer=optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Pre-fix checkpoints carry no scheduler state; fast-forward instead.
            logger.warning("Checkpoint has no scheduler state; fast-forwarding the LR schedule")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch * steps_per_epoch):
                    scheduler.step()
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        logger.info(
            f"Resumed from epoch {start_epoch} | lr {scheduler.get_last_lr()[0]:.3e} "
            f"| best val {best_val_acc*100:.2f}%"
        )

    # Training loop
    checkpoint_dir = os.path.join(config.get("checkpoint_dir", "checkpoints/"), "base")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_vqa_loss = 0.0
        epoch_gate_loss = 0.0
        num_batches = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer_idx"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, confidence, z = model(images, input_ids, attention_mask)

            # Loss: soft-label BCE (official VQA training loss) or hard cross-entropy.
            # vqa_loss sums the BCE over answers; see ttt/losses.py for why.
            use_soft = config.get("train_loss", "soft_bce") == "soft_bce" and "answer_scores" in batch
            answer_scores = batch["answer_scores"].to(device) if use_soft else None
            loss_vqa = vqa_loss(logits, answers, answer_scores)

            # Gate auxiliary loss (fp32 for BCE numerical stability)
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == answers).float()
            loss_gate = F.binary_cross_entropy(confidence.float().squeeze(-1), correct)

            # Combined loss, scaled for gradient accumulation
            loss = (loss_vqa + 0.1 * loss_gate) / grad_accum_steps

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Step only at accumulation boundaries (or last batch)
            is_step_boundary = ((batch_idx + 1) % grad_accum_steps == 0
                                or (batch_idx + 1) == len(train_loader))
            if is_step_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * grad_accum_steps
            epoch_vqa_loss += loss_vqa.item()
            epoch_gate_loss += loss_gate.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f} (VQA: {epoch_vqa_loss/num_batches:.4f}, "
                    f"Gate: {epoch_gate_loss/num_batches:.4f})"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch+1}/{epochs} done in {elapsed:.0f}s | "
            f"Loss: {avg_loss:.4f}"
        )

        # Validate
        val_acc, val_predictions, val_soft = evaluate(
            model, val_loader, device, use_amp=use_amp, return_soft=True
        )
        logger.info(f"Epoch {epoch+1} | Val accuracy: {val_acc*100:.2f}%")
        # Separate line, deliberately not matching the "| Val accuracy:" pattern
        # scripts/06_ceiling_check.py parses — that stays exact-match (49.56 scale).
        logger.info(f"Epoch {epoch+1} | Official VQA soft: {val_soft*100:.2f}%")

        # Every checkpoint carries the full training state so --resume can
        # continue the run exactly instead of restarting its LR schedule.
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        train_state = {
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_acc": best_val_acc,
        }

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch,
                        os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"), extra=train_state)

        # Save best
        if is_best:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(checkpoint_dir, "best.pt"), extra=train_state)
            logger.info(f"  New best! Val accuracy: {val_acc*100:.2f}%")

            # Save val predictions for gate label generation
            results_dir = config.get("results_dir", "results/")
            os.makedirs(results_dir, exist_ok=True)
            save_json(val_predictions, os.path.join(results_dir, "base_predictions.json"))
            save_json(val_predictions, os.path.join(results_dir, "base_predictions_val.json"))

    logger.info(f"\nTraining complete! Best val accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
