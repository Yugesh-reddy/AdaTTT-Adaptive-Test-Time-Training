#!/usr/bin/env bash
# Mirrors Phase 1 checkpoints and logs from VM-local disk to Drive every 3 min.
# Waits for the mount rather than failing, so it can start before drivemount.
# Source checkpoints are written atomically, so any *.pt under its final name is
# complete; the copy also lands under a temp name first, so a preemption
# mid-copy never leaves a truncated file on Drive either.
set -uo pipefail
REPO=/content/AdaTTT
DRIVE=/content/drive/MyDrive/AdaTTT
ts() { date -u +%H:%M:%S; }
mirror() {  # src_dir dst_dir
  local src=$1 dst=$2 f base
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  for f in "$src"/*.pt; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    if [ ! -e "$dst/$base" ] || [ "$f" -nt "$dst/$base" ]; then
      cp "$f" "$dst/.$base.partial" && mv -f "$dst/.$base.partial" "$dst/$base" \
        && echo "[$(ts)] synced $base -> ${dst#$DRIVE/}"
    fi
  done
}
while :; do
  if [ -d /content/drive/MyDrive ]; then
    mirror "$REPO/checkpoints/base"                "$DRIVE/checkpoints/phase1_clip"
    mirror "$REPO/checkpoints/phase1_vitbert/base" "$DRIVE/checkpoints/phase1_vitbert"
    mkdir -p "$DRIVE/logs"
    for n in vitbert clip; do
      [ -f "$REPO/logs/train_$n.log" ] && cp "$REPO/logs/train_$n.log" "$DRIVE/logs/train_$n.log"
    done
    echo "[$(ts)] mirrored to Drive" > /content/sync.status
  else
    echo "[$(ts)] waiting for drive mount" > /content/sync.status
  fi
  sleep 180
done
