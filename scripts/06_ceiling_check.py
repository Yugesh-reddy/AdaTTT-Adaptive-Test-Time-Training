"""
Phase 1 gate: did the backbone swap actually move the representation ceiling?

The v1 run reached its ceiling by epoch 5 and then spent ten epochs memorising:
training loss fell 14.6x from epoch 5 to 15 while validation accuracy moved
+0.33pp. The whole v2 thesis rests on that ceiling being a property of frozen,
mutually-unaligned ImageNet-ViT and BERT features rather than of the fusion
module. This script decides whether it was.

Two-part criterion — a run is KILLED only if BOTH hold:

  1. Flat:      val gain from epoch 5 to epoch 8 is <= --flat-threshold (0.5pp)
  2. No better: final val is within --near-v1 (2.0pp) of the v1 reference

Part 2 matters. A run that saturates by epoch 5 at 62% has moved the ceiling
by 8 points and is a success, not a failure — judging slope alone would kill it.

Usage:
    python scripts/06_ceiling_check.py --run logs/train_clip.log
    python scripts/06_ceiling_check.py --run logs/train_clip.log \\
        --control logs/train_vitbert.log

The --control argument is strongly recommended. The current fusion stack is not
the one v1 trained: the prediction head gained a second hidden layer and query
pooling was added, so "CLIP vs the archived 54.30%" confounds three changes.
A vit_bert run under today's architecture isolates the encoders.

Accepts either a training log (parses the per-epoch lines train_base.py emits)
or a JSON history: {"epochs": [{"epoch": 1, "val": 0.4221}, ...]}.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# v1 reference on the official VQA soft metric, recomputed from
# results/ttt_predictions/val/k0_baseline.json against the val annotations.
V1_REFERENCE_SOFT = 54.30
V1_REFERENCE_EXACT = 49.56

EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Val accuracy:\s*([\d.]+)%")
# train_base.py also logs the official metric on its own line. A soft-BCE
# objective optimizes that metric, and exact-match can lag it by ~20 points.
SOFT_LINE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Official VQA soft:\s*([\d.]+)%")


def parse_log_runs(path, pattern=EPOCH_LINE):
    """Split a training log into separate runs.

    train_base.py appends, so one log can hold several runs — logs/train.log
    holds a VQA-v2 run followed by Memotion2. Merging them by epoch number
    silently reads whichever run happened to write last.

    A new run starts only when the epoch counter returns to 1. Any other
    non-increasing epoch is a --resume after preemption replaying epochs whose
    checkpoints had not been saved off the VM yet: that is the same run, so the
    replayed values overwrite the originals. Splitting there instead would drop
    everything before the preemption — epoch 5 included — and leave the
    criterion nothing to read.

    Returns:
        List of {epoch: accuracy} dicts, in file order.
    """
    runs, current = [], {}
    with open(path) as fh:
        for line in fh:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch == 1 and current:
                runs.append(current)
                current = {}
            current[epoch] = float(match.group(2))
    if current:
        runs.append(current)
    return runs


def read_history(path, run_index=-1, metric="exact"):
    """Per-epoch validation accuracy in percent, as {epoch: accuracy}.

    Args:
        path: A train_base.py log file, or a JSON history.
        run_index: Which run to read from a multi-run log (default: the last).

    Returns:
        Dict mapping epoch number to accuracy in percent.

    Raises:
        ValueError: If no epoch records could be read, or run_index is invalid.
    """
    if path.endswith(".json"):
        blob = json.load(open(path))
        rows = blob["epochs"] if isinstance(blob, dict) else blob
        history = {}
        for row in rows:
            acc = row.get("val", row.get("val_accuracy", row.get("accuracy")))
            history[int(row["epoch"])] = acc * 100 if acc <= 1.0 else acc
        if not history:
            raise ValueError(f"No epoch records in {path}.")
        return history

    runs = parse_log_runs(path, SOFT_LINE if metric == "soft" else EPOCH_LINE)
    if not runs:
        raise ValueError(
            f"No epoch records in {path}. Expected lines like "
            f"'Epoch 5 | Val accuracy: 49.11%' or a JSON history."
        )
    if len(runs) > 1:
        spans = ", ".join(
            f"[{i}] {len(r)} epochs, final {r[max(r)]:.2f}%"
            for i, r in enumerate(runs)
        )
        print(
            f"note: {path} contains {len(runs)} runs ({spans}). "
            f"Reading run {run_index}. Use --run-index to pick another.",
            file=sys.stderr,
        )
    try:
        return runs[run_index]
    except IndexError:
        raise ValueError(
            f"--run-index {run_index} out of range; {path} has {len(runs)} runs."
        )


def evaluate(history, flat_threshold, near_v1, reference):
    """Apply the two-part criterion to one run."""
    epochs = sorted(history)
    early, late = 5, 8

    if early not in history or late not in history:
        available = ", ".join(str(e) for e in epochs)
        raise ValueError(
            f"Criterion reads epochs {early} and {late}; log has: {available}. "
            f"Train at least {late} epochs."
        )

    slope = history[late] - history[early]
    final = history[max(epochs)]
    is_flat = slope <= flat_threshold
    is_near_v1 = abs(final - reference) <= near_v1
    # Flat *below* the reference is not a moved ceiling. It means training
    # under this setup underperforms v1, and nothing can be read into the
    # encoders until that regression is explained.
    is_below = final < reference - near_v1

    return {
        "epochs_seen": len(epochs),
        "acc_ep5": history[early],
        "acc_ep8": history[late],
        "final": final,
        "best": max(history.values()),
        "slope_5_to_8": slope,
        "is_flat": is_flat,
        "is_near_v1": is_near_v1,
        "is_below": is_below,
        "verdict": ("KILL" if (is_flat and is_near_v1) else
                    "REGRESSION" if (is_flat and is_below) else "CONTINUE"),
    }


def _evaluate_pair(args, metric, reference):
    runs = {}
    for name, path in (("clip", args.run), ("control (vit_bert)", args.control)):
        if path:
            runs[name] = evaluate(read_history(path, args.run_index, metric),
                                  args.flat_threshold, args.near_v1, reference)
    return runs


def _print_table(title, runs, reference):
    width = max(len(name) for name in runs)
    print(f"\n{title} — reference {reference:.2f}%\n")
    print(f"{'run'.ljust(width)}  {'ep5':>7} {'ep8':>7} {'final':>7} {'best':>7} "
          f"{'slope':>8}  verdict")
    for name, r in runs.items():
        print(f"{name.ljust(width)}  {r['acc_ep5']:>6.2f}% {r['acc_ep8']:>6.2f}% "
              f"{r['final']:>6.2f}% {r['best']:>6.2f}% {r['slope_5_to_8']:>+7.2f}pp  "
              f"{r['verdict']}")


def _explain(r, reference, near_v1):
    gap = r["final"] - reference
    if r["verdict"] == "KILL":
        return (f"KILL — flat by epoch 5 AND within {near_v1:.1f}pp of the reference. The "
                "ceiling did not move, so the frozen-unaligned-encoder hypothesis is wrong. "
                "Do NOT start Phase 2.")
    if r["verdict"] == "REGRESSION":
        return (f"REGRESSION — flat by epoch 5 and {gap:+.2f}pp BELOW the reference. This is "
                "not a moved ceiling: training under this setup underperforms v1. Explain the "
                "regression (objective, architecture) before reading anything into the encoders.")
    if not r["is_flat"]:
        return (f"CONTINUE — still climbing at epoch 8 ({r['slope_5_to_8']:+.2f}pp from epoch 5). "
                "Train to convergence before reading the ceiling, then re-run this check.")
    return (f"CONTINUE — saturated, but {gap:+.2f}pp above the reference. The ceiling moved; "
            "that is the Phase 1 result.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ceiling check")
    parser.add_argument("--run", required=True,
                        help="Training log or JSON history for the CLIP run")
    parser.add_argument("--control", default=None,
                        help="Same, for a vit_bert run under today's architecture")
    parser.add_argument("--run-index", type=int, default=-1,
                        help="Which run to read from a multi-run log (default: last)")
    parser.add_argument("--metric", choices=("exact", "soft"), default="exact",
                        help="Which logged metric drives the verdict (both are reported)")
    parser.add_argument("--flat-threshold", type=float, default=0.5,
                        help="Max ep5->ep8 gain (pp) still counted as flat")
    parser.add_argument("--near-v1", type=float, default=2.0,
                        help="Distance (pp) from the reference counted as no better")
    parser.add_argument("--reference", type=float, default=None,
                        help=f"v1 reference (default {V1_REFERENCE_EXACT} exact / {V1_REFERENCE_SOFT} soft)")
    parser.add_argument("--output", default=None, help="Write the verdict as JSON")
    args = parser.parse_args()

    refs = {"exact": V1_REFERENCE_EXACT, "soft": V1_REFERENCE_SOFT}
    reference = args.reference if args.reference is not None else refs[args.metric]
    runs = _evaluate_pair(args, args.metric, reference)
    _print_table(f"Phase 1 ceiling check [{args.metric}]", runs, reference)
    clip = runs["clip"]
    print("\n" + _explain(clip, reference, args.near_v1))
    if "control (vit_bert)" in runs:
        delta = clip["final"] - runs["control (vit_bert)"]["final"]
        print(f"\nAttributable to the encoders [{args.metric}]: {delta:+.2f}pp "
              f"(CLIP vs vit_bert, same fusion stack).")
    else:
        print("\nNo --control run. The comparison against v1's archived number "
              "also carries a deeper prediction head and added query pooling, so "
              "any gain is not attributable to the encoders alone.")

    other = "soft" if args.metric == "exact" else "exact"
    try:
        other_runs = _evaluate_pair(args, other, refs[other])
        _print_table(f"Same runs on the {other} metric", other_runs, refs[other])
        if "control (vit_bert)" in other_runs:
            d = other_runs["clip"]["final"] - other_runs["control (vit_bert)"]["final"]
            print(f"\nAttributable to the encoders [{other}]: {d:+.2f}pp")
    except ValueError as e:
        other_runs = {}
        print(f"\n({other} metric not available: {e})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"metric": args.metric, "reference": reference, "runs": runs,
                   other: {"reference": refs[other], "runs": other_runs}},
                  open(args.output, "w"), indent=2)
        print(f"\nWrote {args.output}")

    return {"KILL": 1, "REGRESSION": 3}.get(clip["verdict"], 0)


if __name__ == "__main__":
    sys.exit(main())
