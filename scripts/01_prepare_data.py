#!/usr/bin/env python3
"""
Download and preprocess VQA-v2 / VizWiz / Memotion2 dataset.

Usage (LOCAL — no GPU needed):
    python scripts/01_prepare_data.py --config config/config.yaml
    python scripts/01_prepare_data.py --config config/config.yaml --dataset vizwiz
    python scripts/01_prepare_data.py --config config/config.yaml --dataset memotion2
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.utils import load_config
from ttt.data import download_vqa_v2, download_memotion2, build_answer_vocab


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess VQA data")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to download (vqa_v2, vizwiz, or memotion2). Defaults to config value.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image download (download questions/annotations only)",
    )
    parser.add_argument(
        "--build-vocab-only",
        action="store_true",
        help="Only build answer vocabulary (assumes data already downloaded)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.get("data_dir", "data/")
    dataset = args.dataset or config.get("dataset", "vqa_v2")
    num_answers = config.get("num_answers", 3129)

    print(f"Dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Answer vocab size: {num_answers}")
    print()

    if dataset == "vqa_v2":
        if not args.build_vocab_only:
            # Step 1: Download questions and annotations
            print("=" * 60)
            print("Step 1: Downloading VQA-v2 questions and annotations")
            print("=" * 60)
            download_vqa_v2(data_dir)

        # Step 2: Build answer vocabulary from training annotations
        train_ann_path = os.path.join(data_dir, "v2_mscoco_train2014_annotations.json")
        vocab_save_path = os.path.join(data_dir, "answer_vocab.json")

        if os.path.exists(train_ann_path):
            print("\n" + "=" * 60)
            print("Step 2: Building answer vocabulary")
            print("=" * 60)
            vocab = build_answer_vocab(
                train_ann_path, top_k=num_answers, save_path=vocab_save_path
            )
            print(f"  Built vocabulary with {len(vocab)} answers")
            print(f"  Saved to: {vocab_save_path}")

            # Print some stats
            print(f"\n  Top 10 answers:")
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            for ans, idx in sorted_vocab[:11]:
                print(f"    [{idx}] {ans}")
        else:
            print(f"\n  [WARNING] Training annotations not found at {train_ann_path}")
            print(f"  Download data first, then run with --build-vocab-only")

        # Step 3: Create directory structure
        print("\n" + "=" * 60)
        print("Step 3: Creating directory structure")
        print("=" * 60)
        for subdir in [
            "checkpoints/base",
            "checkpoints/gate",
            "results/ttt_predictions",
            "results/ablation",
            "figures",
        ]:
            os.makedirs(subdir, exist_ok=True)
            print(f"  Created: {subdir}/")

    elif dataset == "vizwiz":
        print("VizWiz dataset setup")
        print("Download VizWiz from https://vizwiz.org/tasks-and-datasets/vqa/")
        print(f"Extract images and annotations to: {data_dir}/vizwiz/")

        os.makedirs(os.path.join(data_dir, "vizwiz"), exist_ok=True)

    elif dataset == "memotion2":
        # Step 1: Download instructions
        print("=" * 60)
        print("Step 1: Memotion2 dataset setup")
        print("=" * 60)
        download_memotion2(data_dir)

        # Step 2: Create directory structure
        print("\n" + "=" * 60)
        print("Step 2: Creating directory structure")
        print("=" * 60)
        memo_dir = config.get("memotion2_data_dir", os.path.join(data_dir, "memotion2"))
        for subdir in [
            os.path.join(memo_dir, "images"),
            "checkpoints/base",
            "checkpoints/gate",
            "results/memotion2",
            "figures",
        ]:
            os.makedirs(subdir, exist_ok=True)
            print(f"  Created: {subdir}/")

    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
