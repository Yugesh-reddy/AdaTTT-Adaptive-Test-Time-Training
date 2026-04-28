# Artifact pointers (not in git)

The per-sample results stay gitignored under `results/`. `numbers.json` is generated from them with `python scripts/writeup_numbers.py`; the figures with `python scripts/writeup_figures.py` (CPU; reads `numbers.json`).

| What | Path |
|---|---|
| Phase 1 ceiling | `results/phase1/ceiling_check.json` |
| Phase 1 run | `results/phase1/run_summary.json` |
| Phase 2 per-sample outcomes | `results/phase2/{identity,blur_s3,noise_s5}/<method>.npz` |
| Phase 2 summaries | `results/phase2/<condition>/summary.json` |
| τ (in-sample, pre-2026-09-18) | `results/phase2/{blur_s3,noise_s5}/tau.json` |
| blur s3 orchestrator | `results/phase2/blur_s3/orch_summary.json` |
| Session C orchestrator (identity + noise s5) | `results/phase2/orch_summary.json` |

The npz `flops_g` fields use the pre-2026-09-18 cost accounting (MEMO4 billed 138.6G); `writeup_numbers.py` recomputes FLOPs from each sample's `adapted` flag.

Frozen subsets (in git): `data/eval_subset_8k.json`, `data/gate_train_subset_8k.json`.
