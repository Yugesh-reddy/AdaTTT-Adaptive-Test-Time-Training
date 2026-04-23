#!/usr/bin/env bash
# AdaTTT Phase 1 — launch the 8-epoch runs concurrently on one A100.
# Called by vm_setup.sh once setup is done and /content/launch.ready exists.
#   --resume-<run> <ckpt>   continue that run from a relayed checkpoint
#   --skip-<run>            that run already finished; don't relaunch it
# Checkpoints and logs go to VM-local disk; the orchestrator relays them off.
set -euo pipefail
ts() { date -u +%H:%M:%S; }
REPO=/content/AdaTTT
start=$(grep -n "setup start" /content/setup.log | tail -1 | cut -d: -f1)
tail -n +"$start" /content/setup.log | grep -q SETUP_DONE || { echo "setup has not finished"; exit 1; }

RESUME_V=""; RESUME_C=""; SKIP_V=0; SKIP_C=0
while [ $# -gt 0 ]; do case "$1" in
  --resume-vitbert) RESUME_V="--resume $2"; shift 2;;
  --resume-clip)    RESUME_C="--resume $2"; shift 2;;
  --skip-vitbert)   SKIP_V=1; shift;;
  --skip-clip)      SKIP_C=1; shift;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

cd "$REPO"; mkdir -p logs
# On a fresh VM after a loss, restore each run's log so `tee -a` appends to the
# full history — the ceiling check needs epoch 5 even if it ran on a dead VM.
for n in vitbert clip; do
  if [ -f "/content/resume/train_$n.log" ] && [ ! -f "logs/train_$n.log" ]; then
    cp "/content/resume/train_$n.log" "logs/train_$n.log"
  fi
done

launch() {  # name config resume-args
  local name=$1 cfg=$2 extra=$3
  echo "[$(ts)] launching $name ${extra:+($extra)}"
  PYTHONUNBUFFERED=1 nohup bash -c \
    "python gpu/train_base.py --config $cfg --epochs 8 $extra 2>&1 | tee -a logs/train_$name.log" \
    > /dev/null 2>&1 &
  echo "$!" > "/content/$name.pid"
}
[ "$SKIP_V" = 1 ] && echo "[$(ts)] vitbert already complete — skipped" || launch vitbert /tmp/cfg_vitbert.yaml "$RESUME_V"
[ "$SKIP_C" = 1 ] && echo "[$(ts)] clip already complete — skipped"    || launch clip    config/config.yaml    "$RESUME_C"
pgrep -f vm_sync.sh > /dev/null || { nohup bash /content/vm_sync.sh >> /content/sync.log 2>&1 & }
sleep 45
ps -eo pid,etime,cmd | grep "[t]rain_base" || echo "WARNING: no train_base process visible"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
