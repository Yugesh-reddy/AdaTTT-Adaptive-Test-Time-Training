import os
"""End-to-end smoke test of gpu/train_base.py on a tiny synthetic COCO (CPU/MPS).

Both Phase 1 configs, 2 epochs, then --resume into a 3rd. Exercises everything
the 00:24 attempt never reached: optimizer, training step, eval, soft metric,
atomic checkpoints, preprocessing routing, and schedule-preserving resume.
"""
import json, os, random, shutil, subprocess, sys
import torch, yaml
from PIL import Image
D = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(S, "smoke"); shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
vocab = json.load(open(f"{D}/data/answer_vocab.json"))
pool = [a for a in ("yes", "no", "2", "1", "white", "red", "blue") if a in vocab]
random.seed(0)
for split, base in (("train", 1000), ("val", 2000)):
    os.makedirs(f"{T}/{split}2014")
    qs, anns = [], []
    for i in range(6 if split == "train" else 4):
        img = base + i
        Image.new("RGB", (64, 48), tuple(random.randrange(256) for _ in range(3))).save(
            f"{T}/{split}2014/COCO_{split}2014_{img:012d}.jpg")
        for j in range(2):
            qid, ans = img * 1000 + j, random.choice(pool)
            qs.append({"question_id": qid, "image_id": img, "question": f"what is object {j}?"})
            anns.append({"question_id": qid, "image_id": img, "question_type": "what",
                         "answer_type": "yes/no" if ans in ("yes", "no") else "other",
                         "multiple_choice_answer": ans,
                         "answers": [{"answer": ans, "answer_confidence": "yes", "answer_id": k + 1} for k in range(10)]})
    json.dump({"questions": qs}, open(f"{T}/v2_OpenEnded_mscoco_{split}2014_questions.json", "w"))
    json.dump({"annotations": anns}, open(f"{T}/v2_mscoco_{split}2014_annotations.json", "w"))
shutil.copyfile(f"{D}/data/answer_vocab.json", f"{T}/answer_vocab.json")
sys.path.insert(0, D)
from ttt.utils import load_config
cfgs = {"clip": load_config(f"{D}/config/config.yaml"),
        "vitbert": yaml.safe_load(open(f"{S}/payload/AdaTTT/cfg_vitbert.yaml"))}
expect = {"clip": ("CLIPTokenizerFast", "0.4815"), "vitbert": ("BertTokenizer", "0.485")}
allok = True
for name, c in cfgs.items():
    c = dict(c, data_dir=f"{T}/", checkpoint_dir=f"{T}/ckpt_{name}/", results_dir=f"{T}/results_{name}/",
             train_batch_size=4)
    cfg = f"{T}/cfg_{name}.yaml"; yaml.safe_dump(c, open(cfg, "w"))
    ck = f"{T}/ckpt_{name}/base"
    def run(extra, tag):
        p = subprocess.run([sys.executable, f"{D}/gpu/train_base.py", "--config", cfg, "--num-workers", "0"] + extra,
                           capture_output=True, text=True, cwd=T, env=dict(os.environ, PYTHONPATH=D), timeout=900)
        open(f"{T}/{name}_{tag}.out", "w").write(p.stdout + p.stderr)
        return p.returncode, p.stdout + p.stderr
    rc1, out1 = run(["--epochs", "2"], "run1")
    have = sorted(os.listdir(ck)) if os.path.isdir(ck) else []
    state = torch.load(f"{ck}/epoch_1.pt", map_location="cpu", weights_only=True) if "epoch_1.pt" in have else {}
    rc2, out2 = run(["--epochs", "3", "--resume", f"{ck}/epoch_1.pt"], "run2")
    tok, mean = expect[name]
    prep = next((l for l in out1.splitlines() if "Preprocessing:" in l), "")
    checks = [
        ("run exits 0", rc1 == 0),
        ("optimizer + training step + eval", "Epoch 2 | Val accuracy" in out1),
        ("official soft line", "Official VQA soft" in out1),
        ("training completes", "Training complete" in out1),
        (f"tokenizer is {tok}", tok in prep),
        (f"image mean starts {mean}", mean in prep),
        (f"epoch_0/1 + best written (have {have})", all(f in have for f in ("epoch_0.pt", "epoch_1.pt", "best.pt"))),
        ("no .tmp left behind", not any(f.endswith(".tmp") for f in have)),
        ("checkpoint carries training state", all(k in state for k in ("scheduler", "scaler", "best_val_acc"))),
        ("resume exits 0", rc2 == 0),
        ("resume restores schedule", "Resumed from epoch 2 | lr" in out2),
        ("epoch 3 after resume", "Epoch 3 | Val accuracy" in out2 and os.path.exists(f"{ck}/epoch_2.pt")),
    ]
    print(f"  --- {name} ---")
    for label, ok in checks:
        allok &= ok
        print(f"    {'ok ' if ok else 'BAD'} {label}")
print("  SMOKE TEST", "PASSED" if allok else "FAILED")
sys.exit(0 if allok else 1)
