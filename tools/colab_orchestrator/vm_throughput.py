# Executed via `colab exec -f`. Projects each run's remaining wall time from
# its logged batch cadence. The two runs share one GPU and 12 vCPUs, so the
# measured rate already includes the contention between them.
import os, re
from datetime import datetime, timedelta

BATCH = re.compile(r"^\[(\d\d:\d\d:\d\d)\].*Epoch (\d+)/(\d+), Batch (\d+)/(\d+)")
DONE = re.compile(r"^\[(\d\d:\d\d:\d\d)\].*Epoch (\d+)/\d+ done in")
VAL = re.compile(r"^\[(\d\d:\d\d:\d\d)\].*Epoch (\d+) \| Val accuracy")
VAL_BATCHES = -(-214354 // 64)  # full val split at batch 64

def ts(s):
    return datetime.strptime(s, "%H:%M:%S")

def gap(a, b):  # seconds from a to b, tolerant of a midnight wrap
    return ((ts(b) - ts(a)) % timedelta(days=1)).total_seconds()

for name in ("vitbert", "clip"):
    path = f"/content/AdaTTT/logs/train_{name}.log"
    if not os.path.exists(path):
        print(f"{name}: no log yet"); continue
    lines = open(path).read().splitlines()
    pts = [m.groups() for m in map(BATCH.search, lines) if m]
    if len(pts) < 2:
        print(f"{name}: {len(pts)} batch stamps so far — too early"); continue

    first, last = pts[max(0, len(pts) - 8)], pts[-1]
    steps = (int(last[1]) - int(first[1])) * int(last[4]) + int(last[3]) - int(first[3])
    s_per_batch = gap(first[0], last[0]) / max(steps, 1)
    n_train, epochs = int(last[4]), int(last[2])

    done = {int(m.group(2)): m.group(1) for m in map(DONE.search, lines) if m}
    val = {int(m.group(2)): m.group(1) for m in map(VAL.search, lines) if m}
    measured = [gap(done[e], val[e]) for e in val if e in done]
    val_s = sum(measured) / len(measured) if measured else 0.4 * s_per_batch * VAL_BATCHES
    val_src = "measured" if measured else "estimated"

    epoch_now, batch_now = int(last[1]), int(last[3])
    remaining = ((epochs - epoch_now) * n_train + (n_train - batch_now)) * s_per_batch \
        + (epochs - epoch_now + 1) * val_s
    print(f"{name}: {s_per_batch:.3f} s/batch | train epoch {s_per_batch*n_train/60:.1f} min "
          f"+ val {val_s/60:.1f} min ({val_src}) | at epoch {epoch_now}/{epochs} "
          f"batch {batch_now}/{n_train} | REMAINING_H={remaining/3600:.2f}")
