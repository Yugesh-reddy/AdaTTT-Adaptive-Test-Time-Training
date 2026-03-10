#!/usr/bin/env python3
"""
Train the base VQA model (fusion + gate + prediction head) on VQA-v2 or VizWiz.

Standard supervised training with cross-entropy loss. No TTT during training.

Usage on Colab:
    !python gpu/train_base.py --config config/config.yaml --epochs 15

What this trains:
    - θ_f (FusionModule): cross-modal attention
    - θ_g (ConfidenceGate): auxiliary confidence predictor
    - θ_d (PredictionHead): answer classifier

Loss: L = L_vqa + 0.1 * L_gate
    L_vqa  = CrossEntropy(logits, answer_idx)
    L_gate = BCE(gate(z), 1[base_prediction == ground_truth])
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ttt.models import FullVQAModel
from ttt.data import VQADataset, vqa_collate_fn, load_answer_vocab
from ttt.utils import (
    load_config,
    save_checkpoint,
    save_json,
    setup_logging,
    get_device,
    count_parameters,
)


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer_idx"].to(device)

            logits, confidence, z = model(images, input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == answers).sum().item()
            total += answers.size(0)

            # Save predictions
            for i in range(answers.size(0)):
                all_predictions.append({
                    "sample_id": batch["sample_ids"][i],
                    "prediction": preds[i].item(),
                    "ground_truth": answers[i].item(),
                    "question_type": batch["question_types"][i],
                    "confidence": confidence[i].item(),
                })

    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return accuracy, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Train base VQA model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logging("logs")

    # Override config with CLI args
    epochs = args.epochs or config.get("train_epochs", 15)
    batch_size = config.get("train_batch_size", 64)
    lr = config.get("train_lr", 1e-4)
    weight_decay = config.get("train_weight_decay", 0.01)
    warmup_ratio = config.get("train_warmup_ratio", 0.1)
    data_dir = config.get("data_dir", "data/")

    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Load answer vocabulary
    vocab_path = os.path.join(data_dir, "answer_vocab.json")
    answer_vocab = load_answer_vocab(vocab_path)
    logger.info(f"Answer vocab size: {len(answer_vocab)}")

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

    # Create datasets
    train_dataset = VQADataset(
        questions_path=os.path.join(data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
        annotations_path=os.path.join(data_dir, "v2_mscoco_train2014_annotations.json"),
        image_dir=os.path.join(data_dir, "train2014"),
        answer_vocab=answer_vocab,
        max_question_length=config.get("max_question_length", 20),
        image_size=config.get("image_size", 224),
        split="train",
    )
    val_dataset = VQADataset(
        questions_path=os.path.join(data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        annotations_path=os.path.join(data_dir, "v2_mscoco_val2014_annotations.json"),
        image_dir=os.path.join(data_dir, "val2014"),
        answer_vocab=answer_vocab,
        max_question_length=config.get("max_question_length", 20),
        image_size=config.get("image_size", 224),
        split="val",
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=vqa_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Optimizer — only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        from ttt.utils import load_checkpoint
        ckpt = load_checkpoint(model, args.resume, load_optimizer=True, optimizer=optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = os.path.join(config.get("checkpoint_dir", "checkpoints/"), "base")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_vqa_loss = 0.0
        epoch_gate_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer_idx"].to(device)

            # Forward pass
            logits, confidence, z = model(images, input_ids, attention_mask)

            # VQA loss
            loss_vqa = F.cross_entropy(logits, answers)

            # Gate auxiliary loss: predict whether base model gets it right
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == answers).float()
            loss_gate = F.binary_cross_entropy(confidence.squeeze(-1), correct)

            # Combined loss
            loss = loss_vqa + 0.1 * loss_gate

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
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
        val_acc, val_predictions = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Val accuracy: {val_acc*100:.2f}%")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, "best.pt"))
            logger.info(f"  New best! Val accuracy: {val_acc*100:.2f}%")

            # Save val predictions for gate label generation
            results_dir = config.get("results_dir", "results/")
            os.makedirs(results_dir, exist_ok=True)
            save_json(val_predictions, os.path.join(results_dir, "base_predictions.json"))

    logger.info(f"\nTraining complete! Best val accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
