#!/usr/bin/env python3
"""
Phase 2 first A100 session: preflight, then one blur condition on eval 8k.

Runs ON the Colab VM. Builds the 5-view cache on /content (never Drive),
evaluates methods, writes compact npz/json, deletes the cache. Does not
pull caches to the Mac. Does not run the full shift grid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROGRESS = os.environ.get("PHASE2_PROGRESS", "/content/phase2_progress.json")
DONE = os.environ.get("PHASE2_DONE", "/content/PHASE2_DONE")
PREFLIGHT = os.environ.get("PHASE2_PREFLIGHT", "/content/preflight.json")
CACHE = os.environ.get("PHASE2_CACHE", "/content/phase2_cache/blur_s3.pt")
OUTPUT = os.environ.get("PHASE2_OUTPUT", "/content/AdaTTT/results/phase2/blur_s3")
CHECKPOINT = os.environ.get(
    "PHASE2_CHECKPOINT", "/content/AdaTTT/checkpoints/phase1_clip/best.pt"
)
SUBSET = os.environ.get("PHASE2_SUBSET", "data/eval_subset_8k.json")


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _progress(**kwargs) -> None:
    payload = {"done": False, "crash": False}
    payload.update(kwargs)
    _write_json(PROGRESS, payload)


def preflight() -> dict:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    df = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True, check=False)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    info = {
        "nvidia_smi": (smi.stdout or smi.stderr or "").strip(),
        "df_content": (df.stdout or "").strip(),
        "gpu_name": gpu,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(PREFLIGHT, info)
    print("PREFLIGHT_JSON=" + json.dumps(info))
    if "A100" not in gpu:
        raise SystemExit(f"expected an A100, got {gpu}")
    return info


def _delete_cache() -> None:
    for path in (CACHE, os.path.splitext(CACHE)[0] + ".manifest.json"):
        if os.path.exists(path):
            os.remove(path)
            print(f"deleted {path}")
    cache_dir = os.path.dirname(CACHE)
    if cache_dir and os.path.isdir(cache_dir) and not os.listdir(cache_dir):
        os.rmdir(cache_dir)


def main() -> int:
    os.chdir("/content/AdaTTT")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    _progress(stage="preflight", step="preflight")
    try:
        info = preflight()
        free_gb = shutil.disk_usage("/content").free / 1e9
        info["free_gb"] = round(free_gb, 1)
        _write_json(PREFLIGHT, info)
        if free_gb < 20:
            raise SystemExit(f"not enough /content space for a 12GB cache ({free_gb:.1f} GB free)")
        resume_ckpt = "/content/resume/clip_best.pt"
        if os.path.isfile(resume_ckpt):
            os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
            if not os.path.isfile(CHECKPOINT):
                shutil.copyfile(resume_ckpt, CHECKPOINT)
                print(f"copied {resume_ckpt} -> {CHECKPOINT}")

        gpu_dir = os.path.dirname(os.path.abspath(__file__))
        if gpu_dir not in sys.path:
            sys.path.insert(0, gpu_dir)
        import eval_phase2 as eval_mod
        import precompute_shift_features as pre_mod
        precompute_main = pre_mod.main
        eval_main = eval_mod.main

        _progress(stage="precompute", step="precompute 0", n=0)
        rc = precompute_main([
            "--corruption", "gaussian_blur",
            "--severity", "3",
            "--output", CACHE,
            "--subset", SUBSET,
            "--dataset", "vqa_v2",
            "--split", "val",
            "--progress-file", PROGRESS,
        ])
        if rc:
            raise SystemExit(rc)

        import torch
        torch.cuda.empty_cache()

        _progress(stage="eval", step="eval start")
        rc = eval_main([
            "--features", CACHE,
            "--checkpoint", CHECKPOINT,
            "--output", OUTPUT,
            "--source", "corruption_gaussian_blur_s3",
            "--progress-file", PROGRESS,
        ])
        if rc:
            raise SystemExit(rc)

        _delete_cache()
        artifacts = sorted(
            f for f in os.listdir(OUTPUT)
            if f.endswith((".json", ".npz", ".log"))
        )
        done = {
            "done": True,
            "crash": False,
            "stage": "done",
            "step": "PHASE2_DONE",
            "condition": "gaussian_blur_s3",
            "subset": SUBSET,
            "artifacts": artifacts,
            "cache_deleted": not os.path.exists(CACHE),
        }
        _write_json(PROGRESS, done)
        with open(DONE, "w") as fh:
            json.dump(done, fh, indent=2)
            fh.write("\nPHASE2_DONE\n")
        print("PHASE2_DONE")
        return 0
    except Exception:
        traceback.print_exc()
        _progress(stage="failed", step="crash", crash=True, done=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
