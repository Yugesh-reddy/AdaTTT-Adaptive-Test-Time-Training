#!/usr/bin/env python3
"""
Phase 1 orchestrator — two concurrent 8-epoch runs on one Colab A100.

Runs on this Mac, detached from the agent session, so neither an agent outage
nor the user being away can leave an A100 billing idle or a finished run
unstopped.

Every colab CLI call runs under a hard wall-clock timeout. The CLI's own HTTP
calls have none: on 2026-09-10 one `colab exec` sat on a half-open websocket
for three hours (19:19 -> 22:23) while the VM was reclaimed underneath it.
Each poll is also kernel activity, the only thing that visibly kept that VM up.

  allocate + gate     exec failure is never mistaken for a wrong GPU
  upload + setup      setup auto-launches both runs when it finishes
  poll                liveness, progress, crashes — every ORCH_POLL_S
  relay               checkpoints (chunked, sha256-verified) and logs
  resume              on VM loss, from the newest relayed checkpoint
  stop                on completion, crash, or the wall-clock budget

Exit codes: 0 done | 2 exec path broken | 3 wrong GPU | 4 a run crashed
            5 budget reached | 6 projected over budget | 7 out of allocations
            8 artifacts did not land — VM left up for rescue, budget guard armed
            9 transfer path broken — VM stopped before any training

Runtime tokens. Each VM's runtime-proxy token expires about an hour after
allocation. The stock CLI then reads the 404 as a dead VM, deletes the
session and kills its keep-alive, and the still-billing VM is reclaimed
minutes later — that ended every run on 2026-09-10/11 at ~62 min. The
orchestrator refreshes the token every poll (colab_refresh.py), re-adopts a
pruned session whose VM is still assigned, and before allocating anything
releases any orphaned VM it created.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime

SESSION = os.environ.get("ORCH_SESSION", "adattt-p1")
COLAB = os.environ.get("ORCH_COLAB", "colab")
BUDGET_H = float(os.environ.get("ORCH_BUDGET_H", "9"))
POLL_S = int(os.environ.get("ORCH_POLL_S", "240"))
MAX_ALLOC = int(os.environ.get("ORCH_MAX_ALLOC", "3"))
# Wait between allocation attempts when no A100 comes up.
ALLOC_RETRY_S = int(os.environ.get("ORCH_ALLOC_RETRY_S", "300"))
WEDGED_AFTER = int(os.environ.get("ORCH_WEDGED_AFTER", "6"))
PROJECTION_AFTER_S = int(os.environ.get("ORCH_PROJECTION_AFTER_S", "1200"))
# Setup normally takes 8-26 min. On 2026-09-11 a VM sat in setup for 57 min
# billing an idle A100 before it was lost; past this deadline, cut it loose.
SETUP_TIMEOUT_S = int(os.environ.get("ORCH_SETUP_TIMEOUT_S", "2700"))
GUARD_ENABLED = os.environ.get("ORCH_GUARD", "1") == "1"
# ORCH_HELPER is parsed like a shell command line, so a path with spaces
# ("My Drive") must be quoted. The default is a list and is never split.
_HELPER_CMD = os.environ.get("ORCH_HELPER")
HELPER = shlex.split(_HELPER_CMD) if _HELPER_CMD else [
    "/Users/yugesh/.local/share/uv/tools/google-colab-cli/bin/python",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_refresh.py"),
]
PART = int(os.environ.get("ORCH_PART_BYTES", str(64 * 1024 * 1024)))
PREFLIGHT_SIZES = [int(x) for x in os.environ.get(
    "ORCH_PREFLIGHT_SIZES", f"{64 << 20},{16 << 20},{4 << 20}").split(",")]

HERE = os.path.dirname(os.path.abspath(__file__))
# Built by build_payload.py; untracked. Tests point this at a dummy file.
PAYLOAD = os.environ.get("ORCH_PAYLOAD", os.path.join(HERE, "adattt_phase1.tgz"))
PROJECT = os.environ.get(
    "ORCH_PROJECT",
    "/Users/yugesh/Library/CloudStorage/GoogleDrive-yugeshreddysappidi@gmail.com/My Drive/AdaTTT",
)
WORK = os.environ.get("ORCH_WORK", os.path.join(HERE, "orch"))
STATE = os.path.join(WORK, "state.json")
LOG = os.path.join(WORK, "orch.log")
STATUS = os.path.join(WORK, "status.txt")

RUNS = {
    "vitbert": {
        "vm_ckpt": "/content/AdaTTT/checkpoints/phase1_vitbert/base",
        "vm_log": "/content/AdaTTT/logs/train_vitbert.log",
        "out_ckpt": os.path.join(PROJECT, "checkpoints", "phase1_vitbert"),
        "out_log": os.path.join(PROJECT, "logs", "train_vitbert.log"),
    },
    "clip": {
        "vm_ckpt": "/content/AdaTTT/checkpoints/base",
        "vm_log": "/content/AdaTTT/logs/train_clip.log",
        "out_ckpt": os.path.join(PROJECT, "checkpoints", "phase1_clip"),
        "out_log": os.path.join(PROJECT, "logs", "train_clip.log"),
    },
}

PROBE = r'''
import json, os, re, subprocess
EP = re.compile(r"Epoch (\d+) \| Val accuracy: ([\d.]+)%")
runs = {"vitbert": ("/content/AdaTTT/logs/train_vitbert.log", "/content/AdaTTT/checkpoints/phase1_vitbert/base", "/content/vitbert.pid"),
        "clip":    ("/content/AdaTTT/logs/train_clip.log",    "/content/AdaTTT/checkpoints/base",               "/content/clip.pid")}
out = {}
for n, (log, ck, pidf) in runs.items():
    t = open(log).read() if os.path.exists(log) else ""
    L = t.splitlines()
    # Only the latest launch counts. A log restored onto a fresh VM keeps any
    # earlier run's Traceback, which would otherwise flag a healthy run.
    tail = t[t.rfind("Device: "):] if "Device: " in t else t
    vals = {}
    for l in L:
        m = EP.search(l)
        if m:
            vals[m.group(1)] = m.group(2)  # last write wins: a --resume replay supersedes
    bat = [l for l in L if ", Batch " in l]
    pid = open(pidf).read().strip() if os.path.exists(pidf) else ""
    ck_files = {f: os.path.getsize(os.path.join(ck, f)) for f in sorted(os.listdir(ck))
                if f.endswith(".pt")} if os.path.isdir(ck) else {}
    out[n] = {"vals": vals, "at": bat[-1].split("Batch ")[1].split(",")[0] if bat else "",
              "done": "Training complete" in t, "crash": "Traceback" in tail,
              "alive": bool(pid) and os.path.exists("/proc/" + pid), "launched": bool(pid),
              "ckpts": ck_files}
sl = open("/content/setup.log").read() if os.path.exists("/content/setup.log") else ""
sl = sl[sl.rfind("setup start"):] if "setup start" in sl else ""
out["setup"] = ("done" if "SETUP_DONE" in sl else
                "failed" if ("FAILED" in sl or "MISMATCH" in sl) else
                "running" if sl else "absent")
try:
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out["ram_avail_g"] = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()[-1]
except Exception:
    pass
print("PROBE_JSON=" + json.dumps(out))
'''

SPLIT = r'''
import hashlib, json, os, shutil
src, out, part = {src!r}, {out!r}, {part}
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
whole, parts, i = hashlib.sha256(), [], 0
with open(src, "rb") as f:
    while True:
        b = f.read(part)
        if not b:
            break
        whole.update(b)
        name = "part_%03d" % i
        open(os.path.join(out, name), "wb").write(b)
        parts.append({{"name": name, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}})
        i += 1
print("MANIFEST=" + json.dumps({{"size": os.path.getsize(src), "sha256": whole.hexdigest(), "parts": parts}}))
'''

JOIN = r'''
import hashlib, os, shutil
src_dir, dest, want = {src_dir!r}, {dest!r}, {sha!r}
names = sorted(n for n in os.listdir(src_dir) if n.startswith("part_"))
h = hashlib.sha256()
with open(dest + ".tmp", "wb") as out:
    for n in names:
        b = open(os.path.join(src_dir, n), "rb").read()
        h.update(b); out.write(b)
if h.hexdigest() == want:
    os.replace(dest + ".tmp", dest); shutil.rmtree(src_dir, ignore_errors=True)
    print("JOIN_OK")
else:
    print("JOIN_BAD", h.hexdigest())
'''

SHA = r'''
import hashlib, json, os
res = {{}}
for p in {paths!r}:
    if os.path.exists(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        res[p] = h.hexdigest()
print("SHA_JSON=" + json.dumps(res))
'''


# ---------------------------------------------------------------- utilities

def now():
    return time.time()


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"allocations": 0, "vm_seconds": 0.0, "alloc_ts": None, "launch_ts": None,
            "relayed": {r: {"latest": None, "best": None} for r in RUNS},
            "projection_checked": False, "events": []}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def vm_hours(st):
    live = (now() - st["alloc_ts"]) if st["alloc_ts"] else 0.0
    return (st["vm_seconds"] + live) / 3600


def epoch_vals(info):
    """{epoch: val}. Keyed by epoch number, never list position: after a resume
    the log legitimately repeats epochs, and position would point past the end."""
    return {int(k): float(v) for k, v in info.get("vals", {}).items()}


def best_epoch(vals):
    """Earliest epoch at the maximum — train_base saves best.pt on strict >."""
    top = max(vals.values())
    return min(e for e, v in vals.items() if v == top)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cli(args, timeout, stdin=None):
    """Run the colab CLI under a hard wall-clock timeout. Never raises."""
    try:
        p = subprocess.run([COLAB] + args, input=stdin, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: colab {' '.join(args[:2])}"
    except Exception as e:  # CLI missing, OS error
        return 125, f"CLI ERROR: {e}"


def helper(args, timeout=120):
    """colab_refresh.py under a hard timeout. Never raises."""
    try:
        p = subprocess.run(HELPER + args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: helper {args[0]}"
    except Exception as e:
        return 125, f"HELPER ERROR: {e}"


def release_orphans(st):
    """Unassign any VM *this orchestrator* allocated that has lost its session.

    Scoped to our own recorded endpoints: a browser notebook's runtime also
    looks like an orphan to the CLI and must never be touched.
    """
    ours = set(st.get("endpoints", []))
    rc, out = helper(["orphans"], 120)
    for line in out.splitlines():
        if line.startswith("ORPHAN="):
            ep = line.split()[0].split("=", 1)[1]
            if ep in ours:
                rc2, _ = helper(["unassign", ep], 180)
                log(f"  released orphaned VM {ep} (rc={rc2})")


def keep_token_fresh(st):
    """Swap in a current runtime token before the stored one expires.

    Returns "ok", "readopted" (the CLI had pruned the session but the VM was
    still assigned — progress kept), "gone" (the assignment is no longer
    listed), or "unknown" (the check itself failed; the probe decides).
    """
    rc, out = helper(["refresh", SESSION], 120)
    if rc == 0:
        return "ok"
    if rc == 3:
        return "gone"
    if rc == 4 and st.get("endpoint"):
        rc2, out2 = helper(["adopt", SESSION, st["endpoint"]], 180)
        if rc2 == 0:
            log(f"  session had been pruned — re-adopted {st['endpoint']}, training untouched")
            st["events"].append(f"{datetime.now():%H:%M} re-adopted")
            save_state(st)
            return "readopted"
        return "gone" if rc2 == 3 else "unknown"
    return "unknown"


def vm_exec(code, timeout=180):
    return cli(["exec", "-s", SESSION, "--timeout", str(max(30, timeout - 30))], timeout, stdin=code)


def marker(out, key):
    for line in out.splitlines():
        if line.startswith(key + "="):
            return line[len(key) + 1:]
    return None


def session_alive():
    """True / False, or None when the check itself failed."""
    rc, out = cli(["sessions"], 60)
    if rc != 0:
        return None
    if "No active sessions" in out:
        return False
    return SESSION in out


def stop_vm(st, why):
    log(f"stopping VM ({why})")
    cli(["stop", "-s", SESSION], 180)
    release_orphans(st)  # `colab stop` cannot see a session the CLI already pruned
    sweep_failed_new(st, adopt=False)
    if st["alloc_ts"]:
        st["vm_seconds"] += now() - st["alloc_ts"]
    st["alloc_ts"] = None
    st["launch_ts"] = None
    st["setup_ts"] = None
    st["events"].append(f"{datetime.now():%H:%M} stop: {why}")
    save_state(st)


def write_status(st, text):
    with open(STATUS, "w") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  VM time {vm_hours(st):.2f} h of {BUDGET_H} h budget"
                f"  |  allocations {st['allocations']}/{MAX_ALLOC}\n{text}\n")


# ---------------------------------------------------------------- transfers

def fetch_small(remote, local, timeout=180):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    rc, _ = cli(["download", "-s", SESSION, remote, local + ".part"], timeout)
    if rc == 0 and os.path.exists(local + ".part"):
        os.replace(local + ".part", local)
        return True
    return False


def relay_file(remote, local, tag):
    """VM -> local in PART-sized, individually verified chunks. Returns sha or None."""
    t0 = now()
    stage = f"/content/xfer/{tag}"
    rc, out = vm_exec(SPLIT.format(src=remote, out=stage, part=PART), timeout=420)
    m = marker(out, "MANIFEST")
    if not m:
        log(f"  relay {tag}: split failed: {out.strip()[-200:]}")
        return None
    manifest = json.loads(m)
    tmp = os.path.join(WORK, "xfer", tag)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for p in manifest["parts"]:
        dest = os.path.join(tmp, p["name"])
        for attempt in range(3):
            rc, _ = cli(["download", "-s", SESSION, f"{stage}/{p['name']}", dest], 420)
            if rc == 0 and os.path.exists(dest) and sha256_file(dest) == p["sha256"]:
                break
            log(f"  relay {tag}: {p['name']} attempt {attempt + 1} failed (rc={rc})")
        else:
            return None
    joined = local + ".tmp"
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(joined, "wb") as out_f:
        for p in manifest["parts"]:
            with open(os.path.join(tmp, p["name"]), "rb") as f:
                shutil.copyfileobj(f, out_f)
    if os.path.getsize(joined) != manifest["size"] or sha256_file(joined) != manifest["sha256"]:
        log(f"  relay {tag}: reassembled file failed verification")
        os.remove(joined)
        return None
    os.replace(joined, local)
    shutil.rmtree(tmp, ignore_errors=True)
    vm_exec(f"import shutil; shutil.rmtree({stage!r}, ignore_errors=True)", timeout=90)
    mb = manifest["size"] / 1e6
    log(f"  relayed {tag}: {mb:.0f} MB in {now() - t0:.0f}s ({mb / max(now() - t0, 1):.1f} MB/s), sha ok")
    return manifest["sha256"]


def upload_file(local, remote, tag):
    """local -> VM in verified chunks, reassembled atomically on the VM."""
    stage = f"/content/xfer_in/{tag}"
    vm_exec(f"import os, shutil; shutil.rmtree({stage!r}, ignore_errors=True); os.makedirs({stage!r})", 90)
    tmp = os.path.join(WORK, "up", tag)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with open(local, "rb") as f:
        i = 0
        while True:
            b = f.read(PART)
            if not b:
                break
            part = os.path.join(tmp, "part_%03d" % i)
            open(part, "wb").write(b)
            for attempt in range(3):
                rc, _ = cli(["upload", "-s", SESSION, part, f"{stage}/part_{i:03d}"], 420)
                if rc == 0:
                    break
            else:
                return False
            i += 1
    rc, out = vm_exec(JOIN.format(src_dir=stage, dest=remote, sha=sha256_file(local)), 300)
    shutil.rmtree(tmp, ignore_errors=True)
    return "JOIN_OK" in out


def preflight_transfer(st):
    """Find the largest chunk the CLI moves reliably, before any training.

    The CLI moves every file as one base64 JSON body through the Jupyter
    contents API. Learning at the end of a five-hour run that checkpoint-sized
    chunks don't survive would mean stopping the VM with the results still on
    it, so it is measured in minute two, in both directions.
    """
    global PART
    dest = os.path.join(WORK, "xfer_probe.bin")
    for size in PREFLIGHT_SIZES:
        rc, out = vm_exec("import hashlib, os\nb = os.urandom(%d)\n"
                          "open('/content/xfer_probe.bin', 'wb').write(b)\n"
                          "print('PSHA=' + hashlib.sha256(b).hexdigest())" % size, 240)
        want = marker(out, "PSHA")
        t0 = now()
        rc, _ = cli(["download", "-s", SESSION, "/content/xfer_probe.bin", dest], 420)
        ok = bool(want) and rc == 0 and os.path.exists(dest) and sha256_file(dest) == want
        if ok:
            rc, _ = cli(["upload", "-s", SESSION, dest, "/content/xfer_probe_up.bin"], 420)
            _, out2 = vm_exec("import hashlib\nprint('USHA=' + hashlib.sha256("
                              "open('/content/xfer_probe_up.bin', 'rb').read()).hexdigest())", 180)
            ok = rc == 0 and marker(out2, "USHA") == want
        took = now() - t0
        vm_exec("import os\nfor p in ('/content/xfer_probe.bin', '/content/xfer_probe_up.bin'):\n"
                "    os.path.exists(p) and os.remove(p)", 90)
        if os.path.exists(dest):
            os.remove(dest)
        if ok:
            PART = size
            st["part_bytes"] = size
            save_state(st)
            log(f"  transfer preflight: {size / 2**20:.2f} MiB chunks round-trip ok ({took:.0f}s)")
            return True
        log(f"  transfer preflight: {size / 2**20:.2f} MiB chunks failed — trying smaller")
    return False


# ---------------------------------------------------------------- lifecycle

def server_orphans():
    """{endpoint: accelerator} for assignments with no local session, or None on failure."""
    rc, out = helper(["orphans"], 120)
    if rc != 0:
        return None
    found = {}
    for line in out.splitlines():
        if line.startswith("ORPHAN="):
            fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
            found[fields["ORPHAN"]] = fields.get("ACC", "")
    return found


def adopt_or_release_new(st, before, adopt=True):
    """Handle VMs that a failed `colab new` left assigned.

    On 2026-09-11 `colab new` reported failure after the server had assigned
    an A100, once only minutes after returning, and those VMs sat untracked
    and billing until released by hand. Everything assigned since `before` is
    recorded; with adopt=True the first A100 becomes this session and the rest
    are released. Assignments in `before` are never touched, though a browser
    notebook started inside that window would be taken for ours.
    Returns the adopted endpoint or None.
    """
    after = server_orphans()
    if before is None or after is None:
        return None
    fresh = sorted(set(after) - set(before))
    for ep in fresh:
        if ep not in st.setdefault("endpoints", []):
            st["endpoints"].append(ep)
    save_state(st)
    adopted = None
    for ep in fresh:
        if (adopt and adopted is None and after[ep] == "A100"
                and helper(["adopt", SESSION, ep], 180)[0] == 0):
            adopted = ep
            log(f"  colab new failed after assigning {ep}; adopted it")
        else:
            helper(["unassign", ep], 120)
            log(f"  released {ep}, left assigned by a failed colab new")
    return adopted


def sweep_failed_new(st, adopt):
    """Deal with a VM the last failed `colab new` assigned only after returning."""
    before = st.pop("orphans_before_failed_new", None)
    if before is None:
        return None
    save_state(st)
    return adopt_or_release_new(st, before, adopt)


def allocate(st):
    log(f"allocating A100 (allocation {st['allocations'] + 1}/{MAX_ALLOC})")
    st["allocations"] += 1
    save_state(st)
    if sweep_failed_new(st, adopt=True) is None:
        before = server_orphans()
        cli(["new", "-s", SESSION, "--gpu", "A100"], 480)
        alive = session_alive()
        if alive is None:  # the listing failed, not the allocation; look once more
            time.sleep(10)
            alive = session_alive()
        if alive is not True and adopt_or_release_new(st, before) is None:
            if before is not None:
                # Its VM can still appear later; the next attempt or exit sweeps it.
                st["orphans_before_failed_new"] = sorted(before)
                save_state(st)
            return "alloc_failed"
    st["alloc_ts"] = now()
    rc, out = helper(["show", SESSION], 120)
    endpoint = marker(out, "ENDPOINT")
    if endpoint:
        st["endpoint"] = endpoint
        if endpoint not in st.setdefault("endpoints", []):
            st["endpoints"].append(endpoint)
    save_state(st)
    rc, out = cli(["exec", "-s", SESSION, "-f", os.path.join(HERE, "gpu_check.py"),
                   "--timeout", "150"], 200)
    name = marker(out, "GPU_NAME")
    if name is None:
        log(f"  exec path failed before reaching the VM: {out.strip()[-240:]}")
        return "exec_failed"
    if "A100" not in name or "CUDA_AVAILABLE=True" not in out:
        log(f"  wrong hardware: {name}")
        return "wrong_gpu"
    log(f"  confirmed {name}")
    return "ok"


def local_latest(run):
    d = os.path.join(WORK, run)
    eps = [f for f in os.listdir(d) if f.startswith("epoch_") and f.endswith(".pt")] if os.path.isdir(d) else []
    return max(eps, key=lambda f: int(f[6:-3]), default=None)


def run_finished(run):
    p = os.path.join(WORK, run, f"train_{run}.log")
    return os.path.exists(p) and "Training complete" in open(p, errors="replace").read()


def prepare_vm(st):
    if not preflight_transfer(st):
        return "transfer_broken"
    for local, remote in [(PAYLOAD, "/content/adattt_phase1.tgz"),
                          (os.path.join(HERE, "vm_setup.sh"), "/content/vm_setup.sh"),
                          (os.path.join(HERE, "vm_launch.sh"), "/content/vm_launch.sh"),
                          (os.path.join(HERE, "vm_sync.sh"), "/content/vm_sync.sh")]:
        rc, out = cli(["upload", "-s", SESSION, local, remote], 600)
        if rc != 0:
            log(f"  upload {os.path.basename(local)} failed: {out.strip()[-160:]}")
            return "failed"
    rc, out = cli(["exec", "-s", SESSION, "-f", os.path.join(HERE, "vm_bootstrap.py"),
                   "--timeout", "60"], 120)
    if "setup started" not in out:
        log(f"  setup bootstrap failed: {out.strip()[-200:]}")
        return "failed"
    log("  setup started on the VM (COCO download ~15 min)")
    st["setup_ts"] = now()
    save_state(st)

    # Resume material goes up while COCO downloads; setup waits for launch.ready.
    vm_exec("import os; os.makedirs('/content/resume', exist_ok=True)", 90)
    args = []
    for run in RUNS:
        if run_finished(run):
            args.append(f"--skip-{run}")
            continue
        log_local = os.path.join(WORK, run, f"train_{run}.log")
        if os.path.exists(log_local):
            cli(["upload", "-s", SESSION, log_local, f"/content/resume/train_{run}.log"], 300)
        latest = local_latest(run)
        if latest:
            log(f"  uploading {run}/{latest} for resume")
            if not upload_file(os.path.join(WORK, run, latest), f"/content/resume/{run}.pt", run):
                log(f"  resume upload for {run} failed")
                return "failed"
            args += [f"--resume-{run}", f"/content/resume/{run}.pt"]
    code = (f"open('/content/resume.args','w').write({' '.join(args)!r})\n"
            "open('/content/launch.ready','w').write('1')\nprint('READY_OK')")
    rc, out = vm_exec(code, 90)
    if "READY_OK" not in out:
        log("  could not write launch.ready")
        return "failed"
    log(f"  launch armed {('with ' + ' '.join(args)) if args else '(fresh start)'}")
    return "ok"


def relay(st, run, info):
    """Pull the log every poll; pull each new epoch checkpoint once."""
    d = os.path.join(WORK, run)
    os.makedirs(d, exist_ok=True)
    log_local = os.path.join(d, f"train_{run}.log")
    if fetch_small(RUNS[run]["vm_log"], log_local):
        try:
            os.makedirs(os.path.dirname(RUNS[run]["out_log"]), exist_ok=True)
            shutil.copyfile(log_local, RUNS[run]["out_log"])
        except OSError as e:
            log(f"  mirror {run} log to project failed: {e}")

    epochs = sorted((f for f in info.get("ckpts", {}) if f.startswith("epoch_")),
                    key=lambda f: int(f[6:-3]))
    newest = epochs[-1] if epochs else None
    rel = st["relayed"][run]
    if newest and newest != rel["latest"]:
        sha = relay_file(f"{RUNS[run]['vm_ckpt']}/{newest}", os.path.join(d, newest), f"{run}_{newest[:-3]}")
        if sha:
            rel["latest"] = newest
            vals = epoch_vals(info)
            if vals:
                rel["best"] = f"epoch_{best_epoch(vals) - 1}.pt"
            keep = {rel["latest"], rel["best"]}
            for f in os.listdir(d):
                if f.startswith("epoch_") and f.endswith(".pt") and f not in keep:
                    os.remove(os.path.join(d, f))
            save_state(st)


def finalize(st, probe):
    """Both runs complete: land best + final checkpoints and logs in the project.

    Returns (summary, missing). The caller stops the VM only when nothing is
    missing — once it is stopped, anything not yet relayed is gone.
    """
    summary, missing = {}, []
    for run, info in RUNS.items():
        d = os.path.join(WORK, run)
        relay(st, run, probe.get(run, {}))
        vals = epoch_vals(probe.get(run, {}))
        final = f"epoch_{max(vals) - 1}.pt"
        best = f"epoch_{best_epoch(vals) - 1}.pt"
        for need in {final, best}:
            if not os.path.exists(os.path.join(d, need)):
                relay_file(f"{info['vm_ckpt']}/{need}", os.path.join(d, need), f"{run}_{need[:-3]}")
        rc, out = vm_exec(SHA.format(paths=[f"{info['vm_ckpt']}/best.pt"]), 300)
        vm_best = json.loads(marker(out, "SHA_JSON") or "{}").get(f"{info['vm_ckpt']}/best.pt")
        best_local = os.path.join(d, best)
        if vm_best and os.path.exists(best_local) and sha256_file(best_local) != vm_best:
            log(f"  {run}: best.pt differs from {best} byte-wise — relaying best.pt itself")
            relay_file(f"{info['vm_ckpt']}/best.pt", os.path.join(d, "best.pt"), f"{run}_best")
        else:
            shutil.copyfile(best_local, os.path.join(d, "best.pt"))
        os.makedirs(info["out_ckpt"], exist_ok=True)
        for f in ("best.pt", final):
            src = os.path.join(d, f)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(info["out_ckpt"], f + ".tmp"))
                os.replace(os.path.join(info["out_ckpt"], f + ".tmp"), os.path.join(info["out_ckpt"], f))
        summary[run] = {"epochs": len(vals), "final": vals[max(vals)], "best": max(vals.values()),
                        "best_epoch": best_epoch(vals), "saved": sorted(os.listdir(info["out_ckpt"]))}
        missing += [f"{run}/{f}" for f in ("best.pt", final)
                    if not os.path.exists(os.path.join(info["out_ckpt"], f))]
    return summary, missing


def spawn_guard(st):
    """Stops the VM at the budget deadline even if this orchestrator dies."""
    if not GUARD_ENABLED:
        return None
    left = max(60, int((BUDGET_H - vm_hours(st)) * 3600))
    # Stops by endpoint, not just by name: a session the CLI pruned is
    # invisible to `colab stop` while its VM goes on billing.
    g = subprocess.Popen(["sh", "-c", f"sleep {left}; '{sys.executable}' '{os.path.abspath(__file__)}' --emergency-stop"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                         env=dict(os.environ))
    log(f"budget guard armed: hard stop in {left / 3600:.2f} h (pid {g.pid})")
    return g


def main():
    os.makedirs(WORK, exist_ok=True)
    st = load_state()
    log(f"orchestrator start | budget {BUDGET_H} h | poll {POLL_S}s | session {SESSION}")
    if shutil.disk_usage(WORK).free < 4e9:
        log("WARNING: <4 GB free for checkpoint relay")
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    global PART
    PART = st.get("part_bytes", PART)
    guard = spawn_guard(st)
    code = run_loop(st)
    # Exit 8 leaves a VM up on purpose; the guard stays armed as its backstop.
    if guard and guard.poll() is None and code != 8:
        guard.kill()
    log(f"orchestrator exit {code} | VM time used {vm_hours(st):.2f} h")
    return code


def run_loop(st):
    have_vm = False
    if st.get("endpoint") and st.get("alloc_ts"):
        token = keep_token_fresh(st)
        have_vm = token in ("ok", "readopted")
        if have_vm:
            log(f"attaching to existing VM {st['endpoint']} ({token})")
        else:
            release_orphans(st)
    fails = 0
    while True:
        if vm_hours(st) >= BUDGET_H:
            stop_vm(st, "budget reached")
            return 5

        if not have_vm:
            if st["allocations"] >= MAX_ALLOC:
                sweep_failed_new(st, adopt=False)
                log("out of allocations")
                return 7
            status = allocate(st)
            if status == "exec_failed":
                stop_vm(st, "exec path broken")
                return 2
            if status == "wrong_gpu":
                stop_vm(st, "wrong GPU")
                return 3
            if status == "alloc_failed":
                log(f"  no A100 assigned; retrying in {ALLOC_RETRY_S}s")
                time.sleep(ALLOC_RETRY_S)
                continue
            prep = prepare_vm(st)
            if prep == "transfer_broken":
                stop_vm(st, "transfer path broken")
                return 9
            if prep != "ok":
                stop_vm(st, "prepare failed")
                continue
            have_vm, fails = True, 0
            st["projection_checked"] = False
            save_state(st)

        time.sleep(POLL_S)

        token = keep_token_fresh(st)
        alive = False if token == "gone" else (True if token in ("ok", "readopted") else None)
        if alive is False:
            log("VM LOST — assignment no longer listed; resuming on a fresh VM")
            st["vm_seconds"] += now() - st["alloc_ts"] if st["alloc_ts"] else 0
            st["alloc_ts"], st["launch_ts"], st["setup_ts"] = None, None, None
            st["events"].append(f"{datetime.now():%H:%M} vm lost")
            save_state(st)
            have_vm = False
            continue

        rc, out = vm_exec(PROBE, 180)
        raw = marker(out, "PROBE_JSON")
        if raw is None:
            fails += 1
            log(f"probe failed ({fails}/{WEDGED_AFTER}): {out.strip()[-140:]}")
            # `alive` is None when `colab sessions` itself failed. Waiting on
            # that forever is how the first session sat for hours; an
            # unconfirmed VM counts toward recovery the same as a wedged one.
            # Relayed checkpoints bound the cost of a false positive to one epoch.
            if fails >= WEDGED_AFTER and alive is not False:
                stop_vm(st, f"exec path wedged for {fails} polls (liveness {alive})")
                have_vm, fails = False, 0
            continue
        fails = 0
        p = json.loads(raw)

        if p["setup"] == "failed":
            stop_vm(st, "setup failed on the VM")
            have_vm = False
            continue
        if (p["setup"] in ("running", "absent") and st.get("setup_ts")
                and now() - st["setup_ts"] > SETUP_TIMEOUT_S):
            stop_vm(st, f"setup still running after {SETUP_TIMEOUT_S // 60} min")
            have_vm = False
            continue

        lines = [f"setup {p['setup']} | gpu {p.get('gpu', '?')} | ram free {p.get('ram_avail_g', '?')}G"]
        for run in RUNS:
            info = p[run]
            vals = epoch_vals(info)
            if info["launched"] or vals:
                if st["launch_ts"] is None and info["launched"]:
                    st["launch_ts"] = now()
                relay(st, run, info)
            lines.append(f"{run:8s} epochs {len(vals)}/8 last {str(vals[max(vals)]) + '%' if vals else '-'}"
                         f" batch {info['at'] or '-'} alive={info['alive']} done={info['done']}"
                         f" relayed={st['relayed'][run]['latest']}")
        write_status(st, "\n".join(lines))
        log(" | ".join(lines))

        for run in RUNS:
            info = p[run]
            if info["crash"] or (info["launched"] and not info["alive"] and not info["done"]):
                log(f"{run} CRASHED or died — logs relayed to {RUNS[run]['out_log']}")
                stop_vm(st, f"{run} crashed")
                return 4

        if (not st["projection_checked"] and st["launch_ts"]
                and now() - st["launch_ts"] > PROJECTION_AFTER_S):
            rc, out = cli(["exec", "-s", SESSION, "-f", os.path.join(HERE, "vm_throughput.py"),
                           "--timeout", "90"], 150)
            rem = [float(l.split("REMAINING_H=")[1]) for l in out.splitlines() if "REMAINING_H=" in l]
            if rem:
                projected = vm_hours(st) + max(rem) * 1.1 + 0.15
                log(f"projection: {max(rem):.2f} h left on the slower run -> {projected:.2f} h total VM time")
                st["projection_checked"] = True
                save_state(st)
                if projected > BUDGET_H:
                    stop_vm(st, f"projected {projected:.1f} h exceeds {BUDGET_H} h budget")
                    return 6

        if all(p[r]["done"] for r in RUNS):
            log("BOTH RUNS COMPLETE — landing artifacts")
            summary, missing = finalize(st, p)
            if missing:
                log(f"ARTIFACTS NOT LANDED {missing} — leaving the VM up for rescue; budget guard armed")
                write_status(st, "RESCUE NEEDED — not landed: " + ", ".join(missing))
                return 8
            stop_vm(st, "complete")
            ceiling = subprocess.run(
                [sys.executable, os.path.join(PROJECT, "scripts", "06_ceiling_check.py"),
                 "--run", RUNS["clip"]["out_log"], "--control", RUNS["vitbert"]["out_log"],
                 # Official soft vs v1's 54.30: both sides give "<UNK>" no credit.
                 # v1's exact 49.56 holds 4.55pp of <UNK> hits the fixed targets can't earn.
                 "--metric", "soft", "--reference", "54.30",
                 "--output", os.path.join(PROJECT, "results", "phase1", "ceiling_check.json")],
                capture_output=True, text=True)
            log("ceiling check:\n" + ceiling.stdout + ceiling.stderr)
            summary["vm_hours"] = round(vm_hours(st), 2)
            summary["allocations"] = st["allocations"]
            os.makedirs(os.path.join(PROJECT, "results", "phase1"), exist_ok=True)
            json.dump(summary, open(os.path.join(PROJECT, "results", "phase1", "run_summary.json"), "w"), indent=2)
            write_status(st, "COMPLETE\n" + json.dumps(summary, indent=1))
            return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--emergency-stop"]:
        os.makedirs(WORK, exist_ok=True)
        st = load_state()
        log("budget guard fired — emergency stop")
        cli(["stop", "-s", SESSION], 180)
        release_orphans(st)
        sweep_failed_new(st, adopt=False)
        sys.exit(0)
    sys.exit(main())
