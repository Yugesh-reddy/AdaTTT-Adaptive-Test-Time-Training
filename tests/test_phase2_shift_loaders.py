"""
Phase 2 shift-set loaders: VQA-CP's flat-list JSON and VizWiz routing.

VQA-CP ships questions and annotations as flat lists rather than the official
{"questions": [...]} / {"annotations": [...]} objects, and every entry names its
own COCO split because the test split mixes train2014 and val2014 images.
"""

import json
import os

import pytest

from ttt.data import DATASET_REGISTRY, VQADataset, VizWizDataset, build_dataset

VOCAB = {"<UNK>": 0, "yes": 1, "no": 2, "red": 3}


def _write(path, blob):
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return str(path)


def _cp_files(tmp_path, names=("q.json", "a.json")):
    """VQA-CP v2 shape: flat lists carrying coco_split per entry."""
    questions = _write(tmp_path / names[0], [
        {"question_id": 1, "image_id": 9, "question": "what colour?",
         "coco_split": "train2014"},
        {"question_id": 2, "image_id": 42, "question": "any people?",
         "coco_split": "val2014"},
    ])
    annotations = _write(tmp_path / names[1], [
        {"question_id": 1, "image_id": 9, "answer_type": "other", "coco_split": "train2014",
         "answers": [{"answer": "red"}] * 6 + [{"answer": "crimson"}] * 4},
        {"question_id": 2, "image_id": 42, "answer_type": "yes/no", "coco_split": "val2014",
         "answers": [{"answer": "no"}] * 10},
    ])
    return questions, annotations


def _dataset(questions, annotations, image_dir, split="test"):
    return VQADataset(questions, annotations, str(image_dir), VOCAB,
                      tokenizer=object(), split=split)


def test_flat_list_json_loads(tmp_path):
    ds = _dataset(*_cp_files(tmp_path), tmp_path)
    assert len(ds.samples) == 2
    assert ds.samples[0]["question"] == "what colour?"
    assert ds.samples[0]["answer_scores"][3].item() == pytest.approx(1.0)  # red, 6 votes
    assert ds.samples[0]["answer_scores"][0].item() == 0.0  # crimson is OOV: no <UNK> credit
    assert ds.samples[1]["question_type"] == "yes/no"


def test_per_entry_coco_split_drives_the_image_path(tmp_path):
    ds = _dataset(*_cp_files(tmp_path), tmp_path)
    assert ds.samples[0]["image_path"] == os.path.join(
        str(tmp_path), "train2014", "COCO_train2014_000000000009.jpg")
    assert ds.samples[1]["image_path"] == os.path.join(
        str(tmp_path), "val2014", "COCO_val2014_000000000042.jpg")


def test_official_wrapped_json_is_unchanged(tmp_path):
    questions = _write(tmp_path / "q.json", {"questions": [
        {"question_id": 7, "image_id": 5, "question": "what colour?"}]})
    annotations = _write(tmp_path / "a.json", {"annotations": [
        {"question_id": 7, "image_id": 5, "answer_type": "other",
         "answers": [{"answer": "red"}] * 10}]})
    ds = _dataset(questions, annotations, tmp_path / "val2014", split="val")
    assert ds.samples[0]["image_path"] == os.path.join(
        str(tmp_path / "val2014"), "COCO_val2014_000000000005.jpg")


def test_json_without_the_expected_key_still_fails_loudly(tmp_path):
    questions = _write(tmp_path / "q.json", {"data": []})
    annotations = _write(tmp_path / "a.json", {"annotations": []})
    with pytest.raises(ValueError, match="questions"):
        _dataset(questions, annotations, tmp_path)


def test_registry_routes_vqa_cp_and_vizwiz():
    assert "vqa_cp" in DATASET_REGISTRY
    assert build_dataset({}, "vqa_cp", "test", cls_only=True) is VQADataset
    assert build_dataset({}, "vizwiz", "val", cls_only=True) is VizWizDataset
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset({}, "vqacp", "test", cls_only=True)


def test_build_dataset_points_vqa_cp_at_both_coco_splits(tmp_path):
    cp_dir = tmp_path / "vqa_cp"
    cp_dir.mkdir()
    _cp_files(cp_dir, ("vqacp_v2_test_questions.json", "vqacp_v2_test_annotations.json"))
    ds = build_dataset({"data_dir": str(tmp_path), "strict_images": False},
                       "vqa_cp", "test", answer_vocab=VOCAB, tokenizer=object())
    assert ds.samples[0]["image_path"] == os.path.join(
        str(tmp_path), "train2014", "COCO_train2014_000000000009.jpg")
    assert ds.samples[1]["image_path"] == os.path.join(
        str(tmp_path), "val2014", "COCO_val2014_000000000042.jpg")


def test_vizwiz_routing_passes_preprocessing(tmp_path):
    vizwiz = tmp_path / "vizwiz"
    (vizwiz / "val").mkdir(parents=True)
    _write(vizwiz / "val.json", [
        {"question_id": 1, "image": "VizWiz_val_00000000.jpg", "question": "what is this?",
         "answers": [{"answer": "red"}] * 10},
        {"question_id": 2, "image": "VizWiz_val_00000001.jpg", "question": "and this?",
         "answers": [{"answer": "unanswerable"}] * 10},
    ])
    tokenizer = object()
    ds = build_dataset({"data_dir": str(tmp_path), "strict_images": False},
                       "vizwiz", "val", answer_vocab=VOCAB, tokenizer=tokenizer)
    assert ds.tokenizer is tokenizer and ds.transform is not None
    assert ds.samples[1]["question_type"] == "unanswerable"
