#!/usr/bin/env python3
"""
Phase 2 Session C: preflight, identity (skip+MEMO), then gaussian noise s5.

Runs ON the Colab VM. Builds each 5-view cache on /content (never Drive),
evaluates, writes compact npz/json, deletes the cache. A condition with
gated_memo_sar first fits τ on gate_train_subset_8k under the same corruption
(one more cache build), then evaluates the eval 8k with that τ. Does not pull caches
to the Mac. Does not run the remaining 8k grid.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt.phase2_session import (
    GATE_SUBSET as _GATE_SUBSET,
    active_conditions,
    artifacts_for,
    eval_artifacts_for,
    lr_tag,
    needs_tau_fit,
    select_step,
    tau_fit_source,
)

PROGRESS = os.environ.get("PHASE2_PROGRESS", "/content/phase2_progress.json")
DONE = os.environ.get("PHASE2_DONE", "/content/PHASE2_DONE")
PREFLIGHT = os.environ.get("PHASE2_PREFLIGHT", "/content/preflight.json")
CACHE_DIR = os.environ.get("PHASE2_CACHE_DIR", "/content/phase2_cache")
RESULT_ROOT = os.environ.get(
    "PHASE2_RESULT_ROOT", "/content/AdaTTT/results/phase2"
)
CHECKPOINT = os.environ.get(
    "PHASE2_CHECKPOINT", "/content/AdaTTT/checkpoints/phase1_clip/best.pt"
)
SUBSET = os.environ.get("PHASE2_SUBSET", "data/eval_subset_8k.json")
GATE_SUBSET = os.environ.get("PHASE2_GATE_SUBSET", _GATE_SUBSET)


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


def delete_cache(cache_path: str) -> None:
    for path in (cache_path, os.path.splitext(cache_path)[0] + ".manifest.json"):
        if os.path.exists(path):
            os.remove(path)
            print(f"deleted {path}")
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and os.path.isdir(cache_dir) and not os.listdir(cache_dir):
        os.rmdir(cache_dir)


def condition_output(cond: dict) -> str:
    return os.path.join(RESULT_ROOT, cond["result_name"])


def condition_cache(cond: dict) -> str:
    return os.path.join(CACHE_DIR, cond["cache_name"])


def condition_already_landed(cond: dict) -> bool:
    """Skip a condition whose summary is already on this VM (preemption)."""
    return os.path.isfile(os.path.join(condition_output(cond), "summary.json"))


def tau_fit_dir(cond: dict) -> str:
    return os.path.join(condition_output(cond), "tau_fit")


def tau_fit_cache(cond: dict) -> str:
    stem, ext = os.path.splitext(cond["cache_name"])
    return os.path.join(CACHE_DIR, f"{stem}_gate_train{ext}")


def _precompute(precompute_main, cond: dict, cache: str, subset: str) -> None:
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    rc = precompute_main([
        "--corruption", cond["corruption"],
        "--severity", str(cond["severity"]),
        "--output", cache,
        "--subset", subset,
        "--dataset", "vqa_v2",
        "--split", "val",
        "--progress-file", PROGRESS,
    ])
    if rc:
        raise SystemExit(rc)


def fit_tau_on_gate_train(cond: dict, precompute_main, eval_main, extra_args=()) -> str:
    """Held-out τ: fit on GATE_SUBSET under the same corruption, return tau.json.

    Costs one more cache build plus two methods per gated condition. Reuses a
    tau.json already on this VM, so a preempted session does not refit, and a
    gate-train cache a step sweep already built.
    """
    out = tau_fit_dir(cond)
    tau_path = os.path.join(out, "tau.json")
    if os.path.isfile(tau_path):
        print(f"reuse {tau_path}")
        return tau_path
    cache = tau_fit_cache(cond)
    if not os.path.isfile(cache):
        _progress(stage="tau_fit", step=f"{cond['id']} gate-train precompute", condition=cond["id"])
        _precompute(precompute_main, cond, cache, GATE_SUBSET)
    rc = eval_main([
        "--features", cache,
        "--checkpoint", CHECKPOINT,
        "--output", out,
        "--source", tau_fit_source(cond),
        "--progress-file", PROGRESS,
        "--methods", "no_adapt", "memo_sar",
        "--fit-tau",
        *extra_args,
    ])
    if rc:
        raise SystemExit(rc)
    delete_cache(cache)
    if not os.path.isfile(tau_path):
        raise SystemExit(f"{cond['id']}: τ fit wrote no {tau_path}")
    return tau_path


def run_condition(cond: dict, precompute_main, eval_main) -> None:
    cache = condition_cache(cond)
    output = condition_output(cond)
    os.makedirs(output, exist_ok=True)
    if condition_already_landed(cond):
        print(f"skip {cond['id']}: {output}/summary.json already present")
        return
    tau_args = []
    if needs_tau_fit(cond):
        tau_args = ["--tau-file", fit_tau_on_gate_train(cond, precompute_main, eval_main)]

    _progress(
        stage="precompute",
        step=f"{cond['id']} precompute 0",
        condition=cond["id"],
        n=0,
    )
    _precompute(precompute_main, cond, cache, SUBSET)

    import torch
    torch.cuda.empty_cache()

    _progress(stage="eval", step=f"{cond['id']} eval start", condition=cond["id"])
    argv = [
        "--features", cache,
        "--checkpoint", CHECKPOINT,
        "--output", output,
        "--source", cond["source"],
        "--progress-file", PROGRESS,
        "--methods", *cond["methods"],
        *tau_args,
    ]
    rc = eval_main(argv)
    if rc:
        raise SystemExit(rc)
    delete_cache(cache)
    missing = [
        name for name in artifacts_for(cond)
        if not os.path.isfile(os.path.join(output, name))
    ]
    if missing:
        raise SystemExit(f"{cond['id']} missing artifacts: {missing}")


def sweep_dir(cond: dict, lr: float) -> str:
    return os.path.join(condition_output(cond), "sweep", lr_tag(lr))


def _sweep_row(cond: dict, lr: float) -> dict:
    """Gate-train skip vs dense MEMO at one lr, read from its summary."""
    with open(os.path.join(sweep_dir(cond, lr), "summary.json")) as fh:
        summary = json.load(fh)
    runs = {r["config"]: r for r in summary["runs"]}
    base, memo = runs["no_adapt"]["soft"], runs["memo"]["soft"]
    oracle = (summary.get("oracle") or {}).get("oracle_soft")
    return {
        "lr": lr,
        "skip_soft": base,
        "memo_soft": memo,
        "gain_pp": 100.0 * (memo - base),
        "oracle_gain_pp": None if oracle is None else 100.0 * (oracle - base),
        "pred_flips": runs["memo"].get("pred_flips_vs_no_adapt"),
    }


def run_step_sweep(cond: dict, precompute_main, eval_main) -> None:
    """Session D: choose the step size on GATE_SUBSET, then score the eval 8k once.

    decision.json records the pre-registered choice before the eval 8k is
    touched. If no lr reaches cond["min_gain_pp"] the session stops there.
    """
    output = condition_output(cond)
    os.makedirs(output, exist_ok=True)
    decision_path = os.path.join(output, "decision.json")
    gt_cache = tau_fit_cache(cond)

    if os.path.isfile(decision_path):
        with open(decision_path) as fh:
            decision = json.load(fh)
        print(f"reuse {decision_path}: proceed={decision['proceed']}")
    else:
        for lr in cond["lrs"]:
            if os.path.isfile(os.path.join(sweep_dir(cond, lr), "summary.json")):
                continue
            if not os.path.isfile(gt_cache):
                _progress(stage="step_sweep", step=f"{cond['id']} gate-train precompute",
                          condition=cond["id"])
                _precompute(precompute_main, cond, gt_cache, GATE_SUBSET)
            _progress(stage="step_sweep", step=f"{cond['id']} gate-train lr {lr:g}",
                      condition=cond["id"])
            rc = eval_main([
                "--features", gt_cache,
                "--checkpoint", CHECKPOINT,
                "--output", sweep_dir(cond, lr),
                "--source", tau_fit_source(cond),
                "--progress-file", PROGRESS,
                "--methods", "no_adapt", "memo",
                "--lr", str(lr),
            ])
            if rc:
                raise SystemExit(rc)
        rows = [_sweep_row(cond, lr) for lr in cond["lrs"]]
        chosen, record = select_step({r["lr"]: r["gain_pp"] for r in rows}, cond["min_gain_pp"])
        decision = {
            **record,
            "chosen_lr": chosen,
            "sweep": rows,
            "subset": GATE_SUBSET,
            "source": tau_fit_source(cond),
            "eval_subset_touched": chosen is not None,
        }
        _write_json(decision_path, decision)
        print(f"step sweep decision: {json.dumps(decision)}")

    if not decision["proceed"]:
        delete_cache(gt_cache)
        print(f"{cond['id']}: no lr reached {cond['min_gain_pp']} pp on gate-train; "
              "eval 8k not touched")
        return
    if os.path.isfile(os.path.join(output, "summary.json")):
        print(f"skip {cond['id']} eval: summary.json already present")
        return

    lr_args = ["--lr", str(decision["chosen_lr"])]
    tau_path = fit_tau_on_gate_train(cond, precompute_main, eval_main, extra_args=lr_args)
    cache = condition_cache(cond)
    _progress(stage="precompute", step=f"{cond['id']} eval precompute", condition=cond["id"])
    _precompute(precompute_main, cond, cache, SUBSET)
    import torch
    torch.cuda.empty_cache()
    _progress(stage="eval", step=f"{cond['id']} eval start", condition=cond["id"])
    rc = eval_main([
        "--features", cache,
        "--checkpoint", CHECKPOINT,
        "--output", output,
        "--source", cond["source"],
        "--progress-file", PROGRESS,
        "--methods", *cond["methods"],
        "--tau-file", tau_path,
        *lr_args,
    ])
    if rc:
        raise SystemExit(rc)
    delete_cache(cache)
    missing = [n for n in eval_artifacts_for(cond) if not os.path.isfile(os.path.join(output, n))]
    if missing:
        raise SystemExit(f"{cond['id']} missing artifacts: {missing}")


def main() -> int:
    os.chdir("/content/AdaTTT")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    conditions = active_conditions()
    done_ids = []
    _progress(stage="preflight", step="preflight", conditions_done=done_ids)
    try:
        info = preflight()
        free_gb = shutil.disk_usage("/content").free / 1e9
        info["free_gb"] = round(free_gb, 1)
        _write_json(PREFLIGHT, info)
        if free_gb < 20:
            raise SystemExit(
                f"not enough /content space for a 12GB cache ({free_gb:.1f} GB free)"
            )
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

        for cond in conditions:
            _progress(
                stage=cond["id"],
                step=f"{cond['id']} start",
                condition=cond["id"],
                conditions_done=done_ids,
            )
            runner = run_step_sweep if cond.get("kind") == "step_sweep" else run_condition
            runner(cond, pre_mod.main, eval_mod.main)
            done_ids.append(cond["id"])
            _progress(
                stage=cond["id"],
                step=f"{cond['id']} landed",
                condition=cond["id"],
                conditions_done=list(done_ids),
            )

        artifacts = {}
        for cond in conditions:
            out = condition_output(cond)
            artifacts[cond["id"]] = sorted(
                f for f in os.listdir(out)
                if f.endswith((".json", ".npz", ".log"))
            ) if os.path.isdir(out) else []
        done = {
            "done": True,
            "crash": False,
            "stage": "done",
            "step": "PHASE2_DONE",
            "conditions": [c["id"] for c in conditions],
            "conditions_done": done_ids,
            "subset": SUBSET,
            "artifacts": artifacts,
            "cache_deleted": not os.path.exists(CACHE_DIR) or not os.listdir(CACHE_DIR),
        }
        _write_json(PROGRESS, done)
        with open(DONE, "w") as fh:
            json.dump(done, fh, indent=2)
            fh.write("\nPHASE2_DONE\n")
        print("PHASE2_DONE")
        return 0
    except Exception:
        traceback.print_exc()
        _progress(
            stage="failed",
            step="crash",
            crash=True,
            done=False,
            conditions_done=done_ids,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
