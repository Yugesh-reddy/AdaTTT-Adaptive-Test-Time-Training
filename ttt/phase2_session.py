"""Phase 2 Session C: ID control on eval 8k, then one hard visual shift.

Torch-free so the Mac orchestrator can load this file without importing ttt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

ALL_METHODS: Sequence[str] = (
    "no_adapt",
    "tent",
    "eata",
    "memo",
    "memo_sar",
    "gated_memo_sar",
)

ID_METHODS: Sequence[str] = ("no_adapt", "memo")


def session_c_conditions() -> List[Dict[str, Any]]:
    """Identity skip+MEMO, then gaussian noise s5 with the full method set."""
    return [
        {
            "id": "identity",
            "corruption": "identity",
            "severity": 1,
            "methods": list(ID_METHODS),
            "cache_name": "identity.pt",
            "result_name": "identity",
            "source": "identity_eval8k",
        },
        {
            "id": "gaussian_noise_s5",
            "corruption": "gaussian_noise",
            "severity": 5,
            "methods": list(ALL_METHODS),
            "cache_name": "noise_s5.pt",
            "result_name": "noise_s5",
            "source": "corruption_gaussian_noise_s5",
        },
    ]


def artifacts_for(condition: Dict[str, Any]) -> List[str]:
    """Compact files to land for one condition. No feature cache."""
    methods = list(condition["methods"])
    names = ["summary.json", "pareto.json", "flops_ladder.json"]
    names.extend(f"{m}.npz" for m in methods)
    if "no_adapt" in methods and "memo" in methods:
        names.append("oracle.json")
    if "gated_memo_sar" in methods:
        names.append("tau.json")
    return names
