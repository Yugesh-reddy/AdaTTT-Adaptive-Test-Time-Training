#!/usr/bin/env python3
"""
Fit the Session E benefit gate on gate_train only, and freeze it.

Reads results/phase2/noise_s5_signals/gate_train/ and never opens eval_sealed/.
Writes results/writeup/gate_spec_session_e.json: cross-validated estimates for
every candidate gate, the pre-registered choice (ttt.benefit_gate), and either
the frozen gate or a stop. Commit that file before scripts/phase2_gate_score.py.

    python scripts/phase2_gate_fit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttt import benefit_gate as bg  # noqa: E402

PART = os.path.join(ROOT, "results", "phase2", "noise_s5_signals", "gate_train")
SPEC = os.path.join(ROOT, "results", "writeup", "gate_spec_session_e.json")
INPUTS = ("no_adapt.npz", "memo.npz", "memo_signals.npz")


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit and freeze the Session E gate")
    ap.add_argument("--part", default=PART, help="gate_train directory")
    ap.add_argument("--spec", default=SPEC, help="where to write the frozen spec")
    args = ap.parse_args(argv)
    if "eval_sealed" in os.path.abspath(args.part):
        ap.error("the gate is fit on gate_train only; eval_sealed stays unread")

    data = bg.load_part(args.part)
    gain = data["memo_soft"] - data["skip_soft"]
    cv = {name: bg.cross_validate(name, data) for name in bg.GATES}
    chosen, selection = bg.select_gate(cv)
    spec = {
        "session": "E",
        "fit_on": os.path.relpath(os.path.abspath(args.part), ROOT),
        "inputs_sha256": {f: _sha256(os.path.join(args.part, f)) for f in INPUTS},
        "gate_train": {
            "n": int(len(gain)),
            "skip_soft": round(100 * float(data["skip_soft"].mean()), 3),
            "memo_gain_pp": round(100 * float(gain.mean()), 3),
            "oracle_gain_pp": round(100 * float(np.maximum(gain, 0).mean()), 3),
            "helped": int((gain > 0).sum()),
            "hurt": int((gain < 0).sum()),
        },
        "settings": {"l2": bg.L2, "folds": bg.N_FOLDS, "seed": bg.SEED,
                     "tie_pp": bg.TIE_PP, "min_gain_pp": bg.MIN_GAIN_PP},
        "cv": cv,
        "selection": selection,
        "proceed": selection["proceed"],
        "gate": bg.fit_final(chosen, data) if chosen else None,
    }
    os.makedirs(os.path.dirname(args.spec), exist_ok=True)
    with open(args.spec, "w") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")

    print(f"gate_train: MEMO {spec['gate_train']['memo_gain_pp']:+.3f} pp, "
          f"oracle {spec['gate_train']['oracle_gain_pp']:+.3f} pp")
    print(f"{'gate':11s} {'CV gain':>8} {'adapt':>6} {'GFLOPs':>7} {'AUROC':>6}")
    for name, r in cv.items():
        print(f"{name:11s} {r['gain_pp']:+8.3f} {r['adapt_rate']:6.3f} {r['avg_gflops']:7.1f} "
              f"{r['auroc_helped_vs_hurt']:6.3f}")
    print(f"choice: {chosen or 'STOP'} ({selection['rule']})")
    print(f"wrote {args.spec}; commit it before scoring eval_sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
