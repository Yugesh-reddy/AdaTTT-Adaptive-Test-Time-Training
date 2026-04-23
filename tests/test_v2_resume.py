"""
Resume after preemption must continue the run, not restart its schedule.

The Phase 1 runs execute on preemptible Colab VMs. train_base.py originally
restored only model and optimizer on --resume, so a resumed run rebuilt the
LambdaLR from step 0 — re-running warmup and restarting the cosine — and reset
best_val_acc, letting the first post-resume epoch overwrite best.pt with a
worse model. Both distort the epoch-5-to-8 slope the ceiling check reads.
"""

import math
import os

import pytest
import torch

from ttt.models import FullVQAModel
from ttt.utils import load_checkpoint, save_checkpoint

TRAIN_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu", "train_base.py"
)


def _cosine_with_warmup(total, warmup):
    """Same shape as train_base.py's lr_lambda."""
    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@pytest.fixture
def model():
    return FullVQAModel({
        "fusion_dim": 64, "fusion_heads": 4, "fusion_layers": 1,
        "prediction_hidden": 32, "num_answers": 10, "gate_hidden": 16,
        "num_query_tokens": 1,
    })


def _stepped(model, total, warmup, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine_with_warmup(total, warmup))
    for _ in range(steps):
        opt.step()
        sched.step()
    return opt, sched


def test_scheduler_state_round_trips_through_a_checkpoint(model, tmp_path):
    opt, sched = _stepped(model, 1000, 100, 437)
    path = tmp_path / "epoch_3.pt"
    save_checkpoint(model, opt, 3, str(path),
                    extra={"scheduler": sched.state_dict(), "best_val_acc": 0.4955})

    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, _cosine_with_warmup(1000, 100))
    ckpt = load_checkpoint(model, str(path), load_optimizer=True, optimizer=opt2)
    sched2.load_state_dict(ckpt["scheduler"])

    assert sched2.get_last_lr()[0] == pytest.approx(sched.get_last_lr()[0])
    assert sched2.last_epoch == 437
    assert ckpt["best_val_acc"] == pytest.approx(0.4955)


def test_a_rebuilt_scheduler_would_have_rewarmed(model):
    """The bug being guarded: a fresh LambdaLR sits back at warmup step 0."""
    _, sched = _stepped(model, 1000, 100, 437)
    _, fresh = _stepped(model, 1000, 100, 0)
    assert fresh.get_last_lr()[0] < sched.get_last_lr()[0] / 10


def test_train_base_saves_and_restores_full_training_state():
    """Source-level guard: every checkpoint carries the state; resume reads it."""
    src = open(TRAIN_BASE).read()
    assert 'scheduler.load_state_dict(ckpt["scheduler"])' in src
    assert 'scaler.load_state_dict(ckpt["scaler"])' in src
    assert 'best_val_acc = ckpt.get("best_val_acc"' in src
    assert src.count("extra=train_state") == 2


def test_checkpoint_write_leaves_no_temp_file(model, tmp_path):
    path = tmp_path / "epoch_0.pt"
    save_checkpoint(model, None, 0, str(path))
    assert path.exists()
    assert not (tmp_path / "epoch_0.pt.tmp").exists()


def test_preemption_mid_write_keeps_the_previous_checkpoint(model, tmp_path, monkeypatch):
    """A kill during torch.save must never leave a truncated file under the final name."""
    path = tmp_path / "best.pt"
    save_checkpoint(model, None, 2, str(path))
    good = path.read_bytes()

    def preempted_save(obj, f):
        with open(f, "wb") as fh:
            fh.write(b"truncated")
        raise RuntimeError("preempted mid-write")

    monkeypatch.setattr(torch, "save", preempted_save)
    with pytest.raises(RuntimeError):
        save_checkpoint(model, None, 3, str(path))
    assert path.read_bytes() == good
