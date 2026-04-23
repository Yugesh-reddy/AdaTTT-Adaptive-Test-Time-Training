"""
Config values must load with the types the training code expects.

PyYAML implements YAML 1.1, where a float needs a decimal point: `1e-4` loads as
the string "1e-4". That string reached torch.optim.AdamW as the learning rate and
killed both Phase 1 runs at optimizer construction, 0.49 h into an A100 run.
"""

import os
import re

import pytest
import torch
import yaml

from ttt.utils import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER_SHAPED = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _walk(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_dotless_exponent_loads_as_float(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lr: 1e-4\nwd: 5E-5\nsteps: 10\nname: 1e4x\nquoted: '1e-4'\n")
    c = load_config(str(p))
    assert isinstance(c["lr"], float) and c["lr"] == pytest.approx(1e-4)
    assert isinstance(c["wd"], float)
    assert c["steps"] == 10 and isinstance(c["steps"], int)
    assert c["name"] == "1e4x"
    assert c["quoted"] == "1e-4", "explicit quotes must still mean a string"


def test_global_safe_loader_is_untouched():
    assert yaml.safe_load("x: 1e-4")["x"] == "1e-4"


def test_no_number_shaped_strings_in_the_real_config():
    c = load_config(os.path.join(ROOT, "config", "config.yaml"))
    bad = [(k, v) for k, v in _walk(c) if isinstance(v, str) and NUMBER_SHAPED.match(v)]
    assert not bad, f"numbers that will load as strings: {bad}"


def test_optimizer_and_schedule_build_from_the_real_config():
    """The exact path that crashed: AdamW, then the warmup LambdaLR."""
    c = load_config(os.path.join(ROOT, "config", "config.yaml"))
    params = [torch.nn.Parameter(torch.zeros(3))]
    opt = torch.optim.AdamW(params, lr=c["train_lr"], weight_decay=c["train_weight_decay"])
    warmup = max(1, int(100 * c["train_warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] > 0


def test_train_base_builds_datasets_through_the_router():
    """Inline VQADataset(...) silently falls back to BERT WordPiece and ImageNet stats."""
    src = open(os.path.join(ROOT, "gpu", "train_base.py")).read()
    assert "build_dataset(" in src and "build_tokenizer(" in src
    assert "VQADataset(" not in src and "Memotion2Dataset(" not in src
