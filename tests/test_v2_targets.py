"""
Out-of-vocabulary votes must never become a positive <UNK> target.

ttt/data.py used to pool every out-of-vocabulary annotator answer into index 0.
<UNK> was then a positive soft target on ~31% of val questions, the Phase 1
control learned to answer "<UNK>" on 37% of a probe slice, and the logged soft
score credited those answers (17pp of a 48.8% score) although the official
metric never matches "<UNK>".
"""

import json

import pytest

from ttt.data import VQADataset

VOCAB = {"<UNK>": 0, "yes": 1, "no": 2, "red": 3}


@pytest.fixture
def make_dataset(tmp_path):
    def _make(answers_by_qid):
        questions = {"questions": [
            {"question_id": qid, "image_id": 1, "question": "what is it?"}
            for qid in answers_by_qid
        ]}
        annotations = {"annotations": [
            {"question_id": qid, "image_id": 1, "answer_type": "other",
             "answers": [{"answer": a} for a in answers]}
            for qid, answers in answers_by_qid.items()
        ]}
        q_path, a_path = tmp_path / "questions.json", tmp_path / "annotations.json"
        q_path.write_text(json.dumps(questions))
        a_path.write_text(json.dumps(annotations))
        # __init__ only stores the tokenizer; images are read in __getitem__.
        return VQADataset(str(q_path), str(a_path), str(tmp_path), VOCAB,
                          tokenizer=object(), split="val")
    return _make


def test_oov_votes_are_dropped_not_pooled_into_unk(make_dataset):
    sample = make_dataset({1: ["crimson"] * 4 + ["scarlet"] * 3 + ["red"] * 3}).samples[0]
    assert sample["answer_scores"][0].item() == 0.0  # pooled, 7 OOV votes gave 1.0
    assert sample["answer_scores"][3].item() == pytest.approx(1.0)


def test_oov_mode_keeps_the_unk_label_for_exact_match(make_dataset):
    """v1's exact-match convention: an out-of-vocab mode is labelled <UNK>."""
    sample = make_dataset({1: ["crimson"] * 6 + ["red"] * 4}).samples[0]
    assert sample["answer_idx"] == 0
    assert sample["answer_scores"][0].item() == 0.0
    assert sample["answer_scores"][3].item() == pytest.approx(1.0)


def test_in_vocab_scores_follow_the_official_formula(make_dataset):
    sample = make_dataset({1: ["yes"] * 2 + ["no"] * 8}).samples[0]
    assert sample["answer_scores"][1].item() == pytest.approx(2 / 3)
    assert sample["answer_scores"][2].item() == pytest.approx(1.0)
    assert sample["answer_scores"].sum().item() == pytest.approx(2 / 3 + 1.0)
