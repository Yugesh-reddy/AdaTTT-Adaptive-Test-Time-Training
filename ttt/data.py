"""
Dataset classes for VQA-v2, VizWiz, and Memotion2.

Handles:
- Image preprocessing (resize 224×224, ImageNet normalization)
- Question/text tokenization (BERT WordPiece, pad/truncate to max_question_length)
- Answer vocabulary construction (top 3129 answers for VQA)
- Memotion2 sentiment label mapping (positive/negative/neutral)
- Data download utilities
"""

import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_image_transform(image_size: int = 224):
    """Get standard image preprocessing for ViT.

    Resize to image_size × image_size, normalize with ImageNet stats.
    """
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Answer vocabulary
# ---------------------------------------------------------------------------

def build_answer_vocab(
    annotations_path: str,
    top_k: int = 3129,
    save_path: Optional[str] = None,
) -> Dict[str, int]:
    """Count answer frequencies across all training annotations.

    Args:
        annotations_path: Path to VQA-v2 annotations JSON file.
        top_k: Number of most frequent answers to keep.
        save_path: If provided, save vocab to this path.

    Returns:
        Dict mapping answer_string → index (0 to top_k-1).
        Index 0 is reserved for <UNK>.
    """
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    answer_counts: Counter = Counter()
    for ann in annotations["annotations"]:
        # Each annotation has 10 answers from human annotators
        for ans in ann["answers"]:
            answer_counts[ans["answer"]] += 1

    # Get top-k most frequent answers (reserve index 0 for <UNK>)
    most_common = answer_counts.most_common(top_k - 1)
    vocab = {"<UNK>": 0}
    for idx, (answer, _count) in enumerate(most_common, start=1):
        vocab[answer] = idx

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab


def load_answer_vocab(path: str) -> Dict[str, int]:
    """Load precomputed answer vocabulary.

    Args:
        path: Path to answer_vocab.json.

    Returns:
        Dict mapping answer_string → index.
    """
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# VQA-v2 Dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    """VQA-v2 or VizWiz dataset.

    __getitem__ returns:
        image: torch.Tensor (3, 224, 224) — preprocessed for ViT
        input_ids: torch.Tensor (max_question_length,) — BERT tokenized question
        attention_mask: torch.Tensor (max_question_length,)
        answer_idx: int — index into answer vocabulary
        question_type: str — "yes/no", "number", "other"
        sample_id: str — unique identifier

    Image preprocessing:
        Resize to 224×224
        Normalize with ImageNet mean/std

    Question preprocessing:
        BERT WordPiece tokenizer, pad/truncate to max_question_length

    Answer preprocessing:
        Map answer string to index in answer vocabulary
        Answers not in vocab get mapped to <UNK> (index 0)
    """

    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        image_dir: str,
        answer_vocab: Dict[str, int],
        tokenizer: Any = None,
        max_question_length: int = 20,
        image_size: int = 224,
        split: str = "train",
    ):
        """
        Args:
            questions_path: Path to VQA-v2 questions JSON file.
            annotations_path: Path to VQA-v2 annotations JSON file.
            image_dir: Directory containing COCO images.
            answer_vocab: Dict mapping answer string → index.
            tokenizer: BERT tokenizer instance. If None, loaded from HuggingFace.
            max_question_length: Max tokens for question.
            image_size: Image resize dimension.
            split: "train" or "val".
        """
        self.image_dir = image_dir
        self.answer_vocab = answer_vocab
        self.max_question_length = max_question_length
        self.image_size = image_size
        self.split = split
        self.transform = get_image_transform(image_size)

        # Load tokenizer
        if tokenizer is None:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # Load questions
        with open(questions_path, "r") as f:
            questions_data = json.load(f)
        questions_map = {q["question_id"]: q for q in questions_data["questions"]}

        # Load annotations
        with open(annotations_path, "r") as f:
            annotations_data = json.load(f)

        # Build samples
        self.samples = []
        for ann in annotations_data["annotations"]:
            qid = ann["question_id"]
            question_info = questions_map.get(qid)
            if question_info is None:
                continue

            # Most frequent answer (mode answer)
            answer_counts: Counter = Counter()
            for ans in ann["answers"]:
                answer_counts[ans["answer"]] += 1
            mode_answer = answer_counts.most_common(1)[0][0]
            answer_idx = self.answer_vocab.get(mode_answer, 0)  # 0 = <UNK>

            # Question type
            question_type = ann.get("answer_type", "other")

            # Image path
            image_id = ann["image_id"]
            image_filename = f"COCO_{split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(image_dir, image_filename)

            self.samples.append({
                "image_path": image_path,
                "question": question_info["question"],
                "answer_idx": answer_idx,
                "question_type": question_type,
                "sample_id": str(qid),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load and preprocess image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError):
            # Return a blank image if file not found (graceful fallback)
            image = torch.zeros(3, self.image_size, self.image_size)

        # Tokenize question
        encoding = self.tokenizer(
            sample["question"],
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_idx": sample["answer_idx"],
            "question_type": sample["question_type"],
            "sample_id": sample["sample_id"],
        }


# ---------------------------------------------------------------------------
# VizWiz Dataset
# ---------------------------------------------------------------------------

class VizWizDataset(Dataset):
    """VizWiz VQA dataset.

    Similar to VQADataset but loads VizWiz format.
    VizWiz has an "unanswerable" class — used as a natural difficulty signal.
    """

    def __init__(
        self,
        annotations_path: str,
        image_dir: str,
        answer_vocab: Dict[str, int],
        tokenizer: Any = None,
        max_question_length: int = 20,
        image_size: int = 224,
    ):
        """
        Args:
            annotations_path: Path to VizWiz annotations JSON file.
            image_dir: Directory containing VizWiz images.
            answer_vocab: Dict mapping answer string → index.
            tokenizer: BERT tokenizer instance.
            max_question_length: Max tokens for question.
            image_size: Image resize dimension.
        """
        self.image_dir = image_dir
        self.answer_vocab = answer_vocab
        self.max_question_length = max_question_length
        self.image_size = image_size
        self.transform = get_image_transform(image_size)

        if tokenizer is None:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # Load annotations
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        self.samples = []
        for ann in annotations:
            # VizWiz format: each entry has image, question, answers list
            answer_counts: Counter = Counter()
            answerable = True
            for ans in ann.get("answers", []):
                ans_text = ans.get("answer", "unanswerable")
                if ans_text == "unanswerable":
                    answerable = False
                answer_counts[ans_text] += 1

            mode_answer = answer_counts.most_common(1)[0][0] if answer_counts else "unanswerable"
            answer_idx = self.answer_vocab.get(mode_answer, 0)

            question_type = "unanswerable" if not answerable else "other"

            image_path = os.path.join(image_dir, ann.get("image", ""))

            self.samples.append({
                "image_path": image_path,
                "question": ann.get("question", ""),
                "answer_idx": answer_idx,
                "question_type": question_type,
                "sample_id": str(ann.get("question_id", len(self.samples))),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError):
            image = torch.zeros(3, self.image_size, self.image_size)

        encoding = self.tokenizer(
            sample["question"],
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_idx": sample["answer_idx"],
            "question_type": sample["question_type"],
            "sample_id": sample["sample_id"],
        }


# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------

def vqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for VQA dataloader.

    Args:
        batch: List of sample dicts from VQADataset.__getitem__.

    Returns:
        Batched tensors dict.
    """
    images = torch.stack([s["image"] for s in batch])
    input_ids = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    answer_idx = torch.tensor([s["answer_idx"] for s in batch], dtype=torch.long)
    question_types = [s["question_type"] for s in batch]
    sample_ids = [s["sample_id"] for s in batch]

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_idx": answer_idx,
        "question_types": question_types,
        "sample_ids": sample_ids,
    }


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_vqa_v2(data_dir: str) -> None:
    """Download VQA-v2 data files.

    Files needed:
    - train2014 images (COCO): http://images.cocodataset.org/zips/train2014.zip (~13GB)
    - val2014 images (COCO): http://images.cocodataset.org/zips/val2014.zip (~6GB)
    - Questions (train/val)
    - Annotations (train/val)

    NOTE: Images are large. For Colab, consider using a subset or Google Drive.
    For initial development, use val split only (smaller) and split into train/test.

    Args:
        data_dir: Root directory for data storage.
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    urls = {
        # Questions
        "v2_Questions_Train_mscoco.zip": (
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
        ),
        "v2_Questions_Val_mscoco.zip": (
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
        ),
        # Annotations
        "v2_Annotations_Train_mscoco.zip": (
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
        ),
        "v2_Annotations_Val_mscoco.zip": (
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
        ),
    }

    for filename, url in urls.items():
        zip_path = os.path.join(data_dir, filename)
        if os.path.exists(zip_path):
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, zip_path)

        # Extract
        print(f"  Extracting {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    # Image downloads — print instructions (too large for auto-download)
    print("\n" + "=" * 60)
    print("IMAGE DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("VQA-v2 images are large. Download manually:")
    print(f"  train2014: http://images.cocodataset.org/zips/train2014.zip (~13GB)")
    print(f"  val2014:   http://images.cocodataset.org/zips/val2014.zip (~6GB)")
    print(f"  Extract to: {data_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Memotion2 label map
# ---------------------------------------------------------------------------

def build_memotion2_label_map() -> Dict[str, int]:
    """Build label mapping for Memotion2 sentiment classification.

    Returns:
        Dict mapping sentiment string → class index.
    """
    return {
        "positive": 0,
        "negative": 1,
        "neutral": 2,
    }


# ---------------------------------------------------------------------------
# Memotion2 Dataset
# ---------------------------------------------------------------------------

class Memotion2Dataset(Dataset):
    """Memotion2 meme sentiment dataset for cross-task evaluation.

    Loads meme images + OCR-extracted text with sentiment labels.
    Returns the same dict format as VQADataset for compatibility
    with existing collate functions and GPU scripts.

    __getitem__ returns:
        image: torch.Tensor (3, 224, 224) — meme image preprocessed for ViT
        input_ids: torch.Tensor (max_question_length,) — BERT tokenized OCR text
        attention_mask: torch.Tensor (max_question_length,)
        answer_idx: int — sentiment class index (positive=0, negative=1, neutral=2)
        question_type: str — always "sentiment" (for analysis compatibility)
        sample_id: str — unique identifier
    """

    def __init__(
        self,
        annotations_path: str,
        image_dir: str,
        label_map: Optional[Dict[str, int]] = None,
        tokenizer: Any = None,
        max_question_length: int = 20,
        image_size: int = 224,
    ):
        """
        Args:
            annotations_path: Path to Memotion2 annotations JSON file.
                Expected format: list of dicts with keys:
                    "image": filename, "text": OCR text, "sentiment": label string
            image_dir: Directory containing meme images.
            label_map: Dict mapping sentiment string → index.
                If None, uses default (positive=0, negative=1, neutral=2).
            tokenizer: BERT tokenizer instance. If None, loaded from HuggingFace.
            max_question_length: Max tokens for OCR text.
            image_size: Image resize dimension.
        """
        self.image_dir = image_dir
        self.label_map = label_map or build_memotion2_label_map()
        self.max_question_length = max_question_length
        self.image_size = image_size
        self.transform = get_image_transform(image_size)

        if tokenizer is None:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # Load annotations
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        self.samples = []
        for idx, ann in enumerate(annotations):
            # Get sentiment label
            sentiment = ann.get("sentiment", "neutral").lower().strip()
            label_idx = self.label_map.get(sentiment, self.label_map.get("neutral", 2))

            # Get OCR text (used as the "question" input to BERT)
            text = ann.get("text", ann.get("ocr_text", ""))

            # Image path
            image_filename = ann.get("image", ann.get("image_name", ""))
            image_path = os.path.join(image_dir, image_filename)

            self.samples.append({
                "image_path": image_path,
                "question": text,  # OCR text treated as "question" for BERT
                "answer_idx": label_idx,
                "question_type": "sentiment",
                "sample_id": str(ann.get("id", idx)),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load and preprocess meme image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError):
            image = torch.zeros(3, self.image_size, self.image_size)

        # Tokenize OCR text
        encoding = self.tokenizer(
            sample["question"],
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_idx": sample["answer_idx"],
            "question_type": sample["question_type"],
            "sample_id": sample["sample_id"],
        }


# ---------------------------------------------------------------------------
# Memotion2 download utility
# ---------------------------------------------------------------------------

def download_memotion2(data_dir: str) -> None:
    """Download Memotion2 data files.

    Memotion2 is from SemEval-2020 Task 8. Available via HuggingFace
    datasets or the official SemEval-2020 release.

    The dataset contains meme images with OCR-extracted text annotations
    and sentiment/humor/sarcasm labels. For AdaTTT we use sentiment
    classification (positive / negative / neutral) as the primary task.

    Args:
        data_dir: Root directory for data storage.
    """
    memotion_dir = os.path.join(data_dir, "memotion2")
    os.makedirs(memotion_dir, exist_ok=True)

    print("=" * 60)
    print("MEMOTION2 DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("Option 1 — HuggingFace:")
    print("  pip install datasets")
    print("  from datasets import load_dataset")
    print('  ds = load_dataset("mediaeval/memotion2")')
    print()
    print("Option 2 — SemEval-2020 official release:")
    print("  https://competitions.codalab.org/competitions/35688")
    print()
    print("After downloading, organize files as:")
    print(f"  {memotion_dir}/images/        — meme images")
    print(f"  {memotion_dir}/train.json     — training annotations")
    print(f"  {memotion_dir}/val.json       — validation annotations")
    print()
    print("Annotation JSON format (list of dicts):")
    print('  [{"image": "img_001.jpg", "text": "OCR text", "sentiment": "positive"}, ...]')
    print("=" * 60)
