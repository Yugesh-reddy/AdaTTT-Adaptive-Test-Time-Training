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
        strict_images: bool = True,
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
        self.strict_images = strict_images
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
            if self.strict_images:
                raise FileNotFoundError(
                    f"Image file missing or unreadable for sample_id={sample['sample_id']}: "
                    f"{sample['image_path']}"
                )
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
        strict_images: bool = True,
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
        self.strict_images = strict_images
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
            if self.strict_images:
                raise FileNotFoundError(
                    f"Image file missing or unreadable for sample_id={sample['sample_id']}: "
                    f"{sample['image_path']}"
                )
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

def download_vqa_v2(data_dir: str, include_image_instructions: bool = True) -> None:
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
    if include_image_instructions:
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
        strict_images: bool = True,
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
        self.strict_images = strict_images
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
            if self.strict_images:
                raise FileNotFoundError(
                    f"Image file missing or unreadable for sample_id={sample['sample_id']}: "
                    f"{sample['image_path']}"
                )
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
# Cached encoder-features dataset (skip frozen ViT/BERT re-encoding)
# ---------------------------------------------------------------------------

class CachedFeaturesDataset(Dataset):
    """Serves precomputed ViT visual tokens + BERT text tokens from disk.

    ViT-B/16 and BERT-base are frozen, so their outputs are deterministic.
    Producing them once with gpu/precompute_features.py and serving them here
    lets TTT sweeps, adaptive inference, and ablations skip the encoder
    forward pass (dominant cost on H100).

    Supports both single-file and sharded feature caches:
        - Single file: ``data/features/val.pt``
        - Sharded: ``data/features/val_shard0.pt``, ``val_shard1.pt``, ...
          (auto-discovered when the single file path doesn't exist)

    Shards are kept as separate tensors in memory — never concatenated —
    so peak RAM equals the size of the largest shard (~16 GB for 50K samples
    in fp16), not the entire dataset.

    load_images controls whether the source dataset is consulted for raw
    images and tokenized text:
        - False (default-safe for masked_patch without consistency reg):
              no image I/O, faster.
        - True: required when TTT objective re-encodes images (rotation)
              or when consistency regularization is active (needs augmented
              ViT forwards).
    """

    def __init__(
        self,
        features_path: str,
        source_dataset: Optional[Dataset] = None,
        load_images: bool = False,
    ):
        if load_images and source_dataset is None:
            raise ValueError(
                "load_images=True requires source_dataset for image loading. "
                "Pass the original VQADataset/Memotion2Dataset, or set "
                "load_images=False for cache-only TTT (e.g., masked_patch without consistency)."
            )

        shards = self._load_shards(features_path)

        # Validate every shard has the required keys
        required = ("sample_ids", "visual_tokens", "text_tokens",
                    "attention_masks", "answer_idx", "question_types")
        for i, shard in enumerate(shards):
            for key in required:
                if key not in shard:
                    raise ValueError(
                        f"Shard {i} from {features_path} is missing required "
                        f"key '{key}'. Regenerate with gpu/precompute_features.py."
                    )

        # Build flat lists for lightweight metadata, keep tensors per-shard
        self.sample_ids: List[str] = []
        self.question_types: List[str] = []
        self._shard_visual: List[torch.Tensor] = []
        self._shard_text: List[torch.Tensor] = []
        self._shard_masks: List[torch.Tensor] = []
        self._shard_answers: List[torch.Tensor] = []
        self._shard_offsets: List[int] = []  # cumulative start index per shard

        offset = 0
        for shard in shards:
            n = len(shard["sample_ids"])
            self.sample_ids.extend(shard["sample_ids"])
            self.question_types.extend(shard["question_types"])
            self._shard_visual.append(shard["visual_tokens"])
            self._shard_text.append(shard["text_tokens"])
            self._shard_masks.append(shard["attention_masks"])
            self._shard_answers.append(shard["answer_idx"])
            self._shard_offsets.append(offset)
            offset += n

        self._total = offset
        self.source_dataset = source_dataset
        self.load_images = load_images

        if source_dataset is not None:
            src_ids = [s["sample_id"] for s in source_dataset.samples]
            if src_ids != self.sample_ids:
                raise ValueError(
                    f"Feature cache at {features_path} does not align with the "
                    "source dataset. Regenerate with gpu/precompute_features.py."
                )

    # Backwards-compatible property so external code can still do dataset.answer_idx[i]
    @property
    def answer_idx(self) -> torch.Tensor:
        """Return answer indices. Concatenates lazily on first access."""
        if not hasattr(self, "_answer_idx_flat"):
            self._answer_idx_flat = torch.cat(self._shard_answers)
        return self._answer_idx_flat

    @property
    def attention_masks(self) -> torch.Tensor:
        """Return attention masks. Concatenates lazily on first access."""
        if not hasattr(self, "_masks_flat"):
            self._masks_flat = torch.cat(self._shard_masks)
        return self._masks_flat

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Map global index to (shard_index, local_index)."""
        # Binary search through shard offsets
        import bisect
        shard_i = bisect.bisect_right(self._shard_offsets, idx) - 1
        local_i = idx - self._shard_offsets[shard_i]
        return shard_i, local_i

    @staticmethod
    def _load_shards(features_path: str) -> list:
        """Load feature shard(s) from a single file or auto-discover shards.

        Returns a list of shard dicts (even for a single file, for uniformity).
        Tensors are memory-mapped (``mmap=True``) so the OS pages data from
        disk on demand — physical RAM usage stays low even for 60+ GB caches.
        """
        import glob as _glob

        # Try mmap first (PyTorch ≥ 2.1); fall back if unsupported.
        def _load(path: str) -> dict:
            try:
                return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except TypeError:
                # Older PyTorch without mmap support
                return torch.load(path, map_location="cpu", weights_only=False)

        if os.path.isfile(features_path):
            return [_load(features_path)]

        # Auto-discover shards: val.pt → val_shard*.pt
        stem, ext = os.path.splitext(features_path)
        pattern = f"{stem}_shard*{ext}"
        shard_files = sorted(_glob.glob(pattern))

        if not shard_files:
            raise FileNotFoundError(
                f"No feature cache found at {features_path} and no shards "
                f"matching {pattern}. Run gpu/precompute_features.py first."
            )

        shards = []
        for sf in shard_files:
            shards.append(_load(sf))
        return shards

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_i, local_i = self._locate(idx)

        if self.load_images:
            sample = self.source_dataset[idx]
        else:
            sample = {
                # Placeholder tensors — collate fn skips these when empty.
                "image": torch.empty(0),
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": self._shard_masks[shard_i][local_i].to(torch.long),
                "answer_idx": int(self._shard_answers[shard_i][local_i]),
                "question_type": self.question_types[idx],
                "sample_id": self.sample_ids[idx],
            }

        sample["visual_tokens"] = self._shard_visual[shard_i][local_i].float()
        sample["text_tokens"] = self._shard_text[shard_i][local_i].float()
        return sample


def cached_vqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate fn for CachedFeaturesDataset.

    Always stacks precomputed features. Includes raw images + input_ids only
    when the dataset was constructed with load_images=True.
    """
    visual_tokens = torch.stack([s["visual_tokens"] for s in batch])
    text_tokens = torch.stack([s["text_tokens"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    answer_idx = torch.tensor([s["answer_idx"] for s in batch], dtype=torch.long)
    question_types = [s["question_type"] for s in batch]
    sample_ids = [s["sample_id"] for s in batch]

    result: Dict[str, Any] = {
        "visual_tokens": visual_tokens,
        "text_tokens": text_tokens,
        "attention_mask": attention_mask,
        "answer_idx": answer_idx,
        "question_types": question_types,
        "sample_ids": sample_ids,
    }

    if batch[0]["image"].numel() > 0:
        result["images"] = torch.stack([s["image"] for s in batch])
        result["input_ids"] = torch.stack([s["input_ids"] for s in batch])

    return result


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
