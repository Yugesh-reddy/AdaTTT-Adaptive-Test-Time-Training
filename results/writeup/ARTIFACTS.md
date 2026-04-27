# Artifact pointers (not in git)

Large JSON / npz stay gitignored under `results/`. This writeup copies the numbers; it does not vendor the caches.

| What | Path |
|---|---|
| Phase 1 ceiling | `results/phase1/ceiling_check.json` |
| Phase 1 run | `results/phase1/run_summary.json` |
| blur s3 summary | `results/phase2/blur_s3/summary.json` |
| identity summary | `results/phase2/identity/summary.json` |
| identity oracle | `results/phase2/identity/oracle.json` |
| noise s5 summary | `results/phase2/noise_s5/summary.json` |
| noise s5 oracle | `results/phase2/noise_s5/oracle.json` |
| Session C orch | `/tmp/adattt-p2/orch-c/orch_summary.json` (ephemeral) |

Frozen subsets (in git): `data/eval_subset_8k.json`, `data/gate_train_subset_8k.json`.

Figures regenerated with `python scripts/writeup_figures.py` (CPU, reads `numbers.json`).
