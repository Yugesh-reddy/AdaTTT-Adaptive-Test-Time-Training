#!/usr/bin/env python3
"""
Final evaluation with the trained adaptive gate.

Runs the FULL adaptive pipeline:
    1. Encode sample (frozen ViT + BERT)
    2. Fuse → z
    3. Gate decides: SKIP or ADAPT
    4. If SKIP: predict from z directly
    5. If ADAPT: run K TTT steps, predict from z'
    6. Record: prediction, routing decision, confidence, FLOPs

Usage on Colab:
    !python gpu/run_inference.py \\
        --base-checkpoint checkpoints/base/best.pt \\
        --gate-checkpoint checkpoints/gate/best.pt \\
        --threshold 0.8 \\
        --k 1 \\
        --objective masked_patch

Run once per threshold to build the Pareto curve:
    for tau in 0.5 0.7 0.8 0.9 0.95; do
        !python gpu/run_inference.py --threshold $tau --k 1 ...
    done
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
from ttt.data import (
    VQADataset, Memotion2Dataset, vqa_collate_fn,
    load_answer_vocab, build_memotion2_label_map,
)
from ttt.metrics import vqa_accuracy, compute_gate_statistics
from ttt.utils import (
    load_config,
    load_checkpoint,
    save_json,
    setup_logging,
    get_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Adaptive inference with gate")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--gate-checkpoint", type=str, default=None,
                        help="Gate checkpoint. If None, uses gate from base checkpoint.")
    parser.add_argument("--threshold", type=float, required=True, help="Gate threshold τ")
    parser.add_argument("--k", type=int, default=1, help="TTT gradient steps")
    parser.add_argument("--objective", type=str, default="masked_patch",
                        choices=["masked_patch", "rotation"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset: vqa_v2, vizwiz, or memotion2 (overrides config)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    use_amp = device.type == "cuda"
    logger = setup_logging("logs")

    threshold = args.threshold
    k = args.k
    objective = args.objective
    data_dir = config.get("data_dir", "data/")
    dataset_name = args.dataset or config.get("dataset", "vqa_v2")
    is_memotion2 = dataset_name == "memotion2"
    strict_images = config.get("strict_images", True)

    # Override num_answers for Memotion2
    if is_memotion2:
        config["num_answers"] = config.get("memotion2_num_classes", 3)

    logger.info(f"Adaptive Inference: τ={threshold}, K={k}, {objective}, dataset={dataset_name}")
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # Load model
    model = FullVQAModel(config)
    model.load_encoders(config)
    load_checkpoint(model, args.base_checkpoint)

    # Optionally load refined gate
    if args.gate_checkpoint:
        gate_ckpt = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)
        model.gate.load_state_dict(gate_ckpt["gate"])
        logger.info(f"Loaded refined gate: {args.gate_checkpoint}")

    model = model.to(device)
    model.eval()

    # Create TTT adapter and router
    ttt_adapter = TTTAdapter(model, config, objective=objective, k_steps=k)
    router = AdaptiveRouter(model, ttt_adapter, threshold=threshold, use_amp=use_amp)

    # Load dataset
    if is_memotion2:
        memo_dir = config.get("memotion2_data_dir", os.path.join(data_dir, "memotion2"))
        val_dataset = Memotion2Dataset(
            annotations_path=os.path.join(memo_dir, "val.json"),
            image_dir=os.path.join(memo_dir, "images"),
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            strict_images=strict_images,
        )
    else:
        answer_vocab = load_answer_vocab(os.path.join(data_dir, "answer_vocab.json"))
        val_dataset = VQADataset(
            questions_path=os.path.join(data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
            annotations_path=os.path.join(data_dir, "v2_mscoco_val2014_annotations.json"),
            image_dir=os.path.join(data_dir, "val2014"),
            answer_vocab=answer_vocab,
            max_question_length=config.get("max_question_length", 20),
            image_size=config.get("image_size", 224),
            split="val",
            strict_images=strict_images,
        )

    if args.max_samples:
        val_dataset.samples = val_dataset.samples[:args.max_samples]

    logger.info(f"Val samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vqa_collate_fn,
    )

    # Run adaptive inference
    all_predictions = []
    all_routing_infos = []
    correct = 0
    total = 0
    total_flops = 0.0
    t0 = time.time()

    for batch in tqdm(val_loader, desc=f"Adaptive τ={threshold}"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answer_idx"].to(device)

        # Adaptive prediction
        logits, routing_info = router.predict(images, input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        correct += (preds == answers).sum().item()
        total += answers.size(0)

        batch_flops = router.compute_flops(routing_info, k)
        total_flops += batch_flops * answers.size(0)
        all_routing_infos.append(routing_info)

        # Record predictions
        skip_mask = routing_info["skip_mask"]
        confidences = routing_info["confidences"]

        for i in range(answers.size(0)):
            all_predictions.append({
                "sample_id": batch["sample_ids"][i],
                "prediction": preds[i].item(),
                "ground_truth": answers[i].item(),
                "question_type": batch["question_types"][i],
                "skipped": bool(skip_mask[i]),
                "confidence": float(confidences[i]),
            })

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0
    avg_flops = total_flops / total if total > 0 else 0.0

    # Gate statistics
    gate_stats = compute_gate_statistics(all_routing_infos)

    logger.info(f"\nResults: τ={threshold}, K={k}, {objective}")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Skip rate: {gate_stats['skip_rate']*100:.1f}%")
    logger.info(f"  Avg GFLOPs/sample: {avg_flops:.1f}")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/total:.2f}s/sample)")
    logger.info(f"  Samples: {gate_stats['total_skip']} skip, {gate_stats['total_adapt']} adapt")

    # Save results
    results_dir = config.get("results_dir", "results/")
    if is_memotion2:
        results_dir = os.path.join(results_dir, "memotion2")
    os.makedirs(results_dir, exist_ok=True)

    prefix = "memotion2_" if is_memotion2 else ""
    filename = f"{prefix}adaptive_t{threshold}_k{k}_{objective}.json"
    save_path = os.path.join(results_dir, filename)
    save_json(all_predictions, save_path)
    logger.info(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
