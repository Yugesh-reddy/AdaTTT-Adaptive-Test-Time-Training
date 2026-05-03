# Artifact pointers (not in git)

The per-sample results and the trained CLIP checkpoint are published in the [v2.0 release](https://github.com/Yugesh-reddy/AdaTTT-Adaptive-Test-Time-Training/releases/tag/v2.0); the paths below are where they belong in a checkout.

The per-sample results stay gitignored under `results/`. `numbers.json` is generated from them with `python scripts/writeup_numbers.py`; the figures with `python scripts/writeup_figures.py` (CPU; reads `numbers.json`).

| What | Path |
|---|---|
| Phase 1 ceiling | `results/phase1/ceiling_check.json` |
| Phase 1 run | `results/phase1/run_summary.json` |
| Phase 2 per-sample outcomes | `results/phase2/{identity,blur_s3,noise_s5}/<method>.npz` |
| Phase 2 summaries | `results/phase2/<condition>/summary.json` |
| τ (in-sample, pre-2026-09-18) | `results/phase2/{blur_s3,noise_s5}/tau.json` |
| Session D step decision | `results/phase2/noise_s5_step_sweep/decision.json` |
| Session D gate-train sweep | `results/phase2/noise_s5_step_sweep/sweep/lr_*/summary.json`, `memo.npz` |
| Session D eval 8k + held-out τ | `results/phase2/noise_s5_step_sweep/{summary.json,tau.json,tau_fit/,*.npz}` |
| blur s3 orchestrator | `results/phase2/blur_s3/orch_summary.json` |
| Session C orchestrator (identity + noise s5) | `results/phase2/orch_summary_c.json` |
| Session D orchestrator (step sweep) | `results/phase2/orch_summary_d.json` |
| Session E outcomes + gate signals | `results/phase2/noise_s5_signals/{gate_train,eval_sealed}/*.npz` |
| Session E frozen gate (in git) | `results/writeup/gate_spec_session_e.json` |
| Session E eval score (in git) | `results/writeup/gate_score_session_e.json` |
| Session E orchestrator | `results/phase2/orch_summary_e.json` |
| Abstention analysis (in git) | `results/writeup/abstention.json`, from `scripts/abstention.py` |

The npz `flops_g` fields use the pre-2026-09-18 cost accounting (MEMO4 billed 138.6G); `writeup_numbers.py` recomputes FLOPs from each sample's `adapted` flag.

Frozen subsets (in git): `data/eval_subset_8k.json`, `data/gate_train_subset_8k.json`.
