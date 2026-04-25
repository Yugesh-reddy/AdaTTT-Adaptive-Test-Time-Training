#!/usr/bin/env bash
# AdaTTT Phase 2 — launch Session C (identity then noise s5). No train_base, no grid.
set -euo pipefail
ts() { date -u +%H:%M:%S; }
REPO=/content/AdaTTT
start=$(grep -n "setup start" /content/setup.log | tail -1 | cut -d: -f1)
tail -n +"$start" /content/setup.log | grep -q SETUP_DONE || { echo "setup has not finished"; exit 1; }

# CLIP fusion checkpoint is uploaded by the Mac orchestrator while COCO fetches.
echo "[$(ts)] waiting for CLIP checkpoint"
for _ in $(seq 1 120); do
  if [ -f /content/resume/clip_best.pt ]; then
    break
  fi
  sleep 5
done
[ -f /content/resume/clip_best.pt ] || { echo "[$(ts)] FAILED: CLIP checkpoint missing"; exit 1; }
mkdir -p "$REPO/checkpoints/phase1_clip" "$REPO/logs" /content/phase2_cache
cp -f /content/resume/clip_best.pt "$REPO/checkpoints/phase1_clip/best.pt"
# Read-only copy on the VM. Never write back to the Mac checkpoint dir.

cd "$REPO"
echo "[$(ts)] launching session C: identity then noise s5"
PYTHONUNBUFFERED=1 nohup python gpu/vm_run_phase2.py \
  >> /content/phase2.log 2>&1 &
echo "$!" > /content/blur.pid
sleep 5
if [ -f /content/blur.pid ] && [ -d "/proc/$(cat /content/blur.pid)" ]; then
  echo "[$(ts)] vm_run_phase2 pid $(cat /content/blur.pid)"
else
  echo "[$(ts)] WARNING: vm_run_phase2 not visible"
  tail -n 40 /content/phase2.log || true
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
