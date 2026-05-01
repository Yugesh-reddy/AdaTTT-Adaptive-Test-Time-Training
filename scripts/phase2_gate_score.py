#!/usr/bin/env python3
"""
Score the frozen Session E gate on eval_sealed, once.

Refuses to run unless results/writeup/gate_spec_session_e.json is committed and
unmodified in git, so the eval data cannot shape the gate. If the spec records
a stop, eval_sealed is not opened at all.

    python scripts/phase2_gate_score.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttt import benefit_gate as bg  # noqa: E402
from ttt.phase2_report import mcnemar_exact_p, paired_bootstrap_ci_pp  # noqa: E402

SPEC = os.path.join(ROOT, "results", "writeup", "gate_spec_session_e.json")
PART = os.path.join(ROOT, "results", "phase2", "noise_s5_signals", "eval_sealed")
OUT = os.path.join(ROOT, "results", "writeup", "gate_score_session_e.json")


def spec_committed(path: str, root: str = ROOT) -> bool:
    """True if `path` is tracked in git and identical to HEAD."""
    rel = os.path.relpath(os.path.abspath(path), root)
    tracked = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", root, "diff", "--quiet", "HEAD", "--", rel],
                           capture_output=True).returncode == 0
    return tracked and clean


def _row(name: str, adapt: np.ndarray, data: dict, gate_name: str) -> dict:
    skip, memo = data["skip_soft"], data["memo_soft"]
    soft = np.where(adapt, memo, skip)
    correct = np.where(adapt, data["memo_correct"], data["skip_correct"])
    b1a0 = int((data["skip_correct"] & ~correct).sum())
    b0a1 = int((~data["skip_correct"] & correct).sum())
    gflops = bg.per_sample_gflops(gate_name, adapt)
    return {
        "method": name,
        "soft": round(100 * float(soft.mean()), 3),
        "delta_pp": round(100 * float((soft - skip).mean()), 3),
        "delta_ci95_pp": paired_bootstrap_ci_pp(soft - skip),
        "adapt_rate": round(float(adapt.mean()), 4),
        "avg_gflops": round(float(gflops.mean()), 2),
        "p50_gflops": round(float(np.percentile(gflops, 50)), 2),
        "p95_gflops": round(float(np.percentile(gflops, 95)), 2),
        "helped": int((adapt & (memo > skip + 1e-6)).sum()),
        "hurt": int((adapt & (memo < skip - 1e-6)).sum()),
        "mcnemar_skip_right_method_wrong": b1a0,
        "mcnemar_skip_wrong_method_right": b0a1,
        "mcnemar_exact_p": round(mcnemar_exact_p(b1a0, b0a1), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the frozen Session E gate once")
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--part", default=PART, help="eval_sealed directory")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    if not spec_committed(args.spec):
        ap.error(f"{args.spec} is not committed (or has local edits). Commit the frozen "
                 "gate first; eval_sealed stays unread until then.")
    with open(args.spec) as fh:
        spec = json.load(fh)
    if not spec["proceed"]:
        print("the stop rule fired on gate_train; eval_sealed was not opened")
        return 0

    data = bg.load_part(args.part)
    n = len(data["skip_soft"])
    gate = spec["gate"]
    adapt = bg.apply_gate(gate, data)
    gain = data["memo_soft"] - data["skip_soft"]
    result = {
        "session": "E",
        "gate": gate["name"],
        "threshold": gate["threshold"],
        "n": n,
        "skip_soft": round(100 * float(data["skip_soft"].mean()), 3),
        "rows": [
            _row("dense MEMO", np.ones(n, dtype=bool), data, "free"),
            _row(f"gated ({gate['name']})", adapt, data, gate["name"]),
        ],
        "oracle_gain_pp": round(100 * float(np.maximum(gain, 0).mean()), 3),
        "spec_gate_train_cv": spec["cv"][gate["name"]],
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"eval_sealed (scored once): skip {result['skip_soft']:.3f}, "
          f"oracle {result['oracle_gain_pp']:+.3f} pp")
    for r in result["rows"]:
        print(f"  {r['method']:20s} {r['delta_pp']:+.3f} pp  CI {r['delta_ci95_pp']}  "
              f"adapt {r['adapt_rate']:.3f}  {r['avg_gflops']:.1f} GFLOPs  "
              f"helped/hurt {r['helped']}/{r['hurt']}  McNemar p={r['mcnemar_exact_p']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
