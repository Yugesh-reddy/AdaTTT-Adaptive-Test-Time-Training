#!/usr/bin/env python3
"""
Phase 2 orchestrator — Session C: identity (eval 8k) then gaussian noise s5.

Reuses allocate / token refresh / budget guard from orch.py without editing
that file (the guard re-reads orch.py from disk). Session is namespaced
(ORCH_SESSION=adattt-p2). Relays compact json/npz only — never a feature cache.
"""
import importlib.util
import json
import os
import sys
import time

# Defaults must land before `import orch` — orch binds SESSION/WORK/HELPER at import.
os.environ.setdefault("ORCH_SESSION", "adattt-p2")
os.environ.setdefault("ORCH_BUDGET_H", "4")
os.environ.setdefault("ORCH_MAX_ALLOC", "8")
os.environ.setdefault("ORCH_WORK", "/tmp/adattt-p2/orch-c")
os.environ.setdefault("ORCH_SETUP_TIMEOUT_S", "3600")
os.environ.setdefault("ORCH_POLL_S", "120")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import orch  # noqa: E402

SESSION = orch.SESSION
PROJECT = orch.PROJECT
WORK = orch.WORK
RESULT_REMOTE = "/content/AdaTTT/results/phase2"
RESULT_LOCAL = os.path.join(PROJECT, "results", "phase2")
CLIP_CKPT = os.path.join(PROJECT, "checkpoints", "phase1_clip", "best.pt")


def _phase2_session():
    """Load ttt/phase2_session.py without importing the torch-heavy ttt package."""
    path = os.path.join(PROJECT, "ttt", "phase2_session.py")
    spec = importlib.util.spec_from_file_location("phase2_session", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orch.RUNS = {
    "blur": {
        "vm_ckpt": RESULT_REMOTE,
        "vm_log": "/content/phase2.log",
        "out_ckpt": RESULT_LOCAL,
        "out_log": os.path.join(RESULT_LOCAL, "phase2.log"),
    },
}

orch.PROBE = r'''
import json, os, subprocess
out = {}
sl = open("/content/setup.log").read() if os.path.exists("/content/setup.log") else ""
sl = sl[sl.rfind("setup start"):] if "setup start" in sl else ""
out["setup"] = ("done" if "SETUP_DONE" in sl else
                "failed" if ("FAILED" in sl or "MISMATCH" in sl) else
                "running" if sl else "absent")
prog = {}
if os.path.exists("/content/phase2_progress.json"):
    try:
        prog = json.load(open("/content/phase2_progress.json"))
    except Exception:
        prog = {"step": "progress-unreadable"}
pid = open("/content/blur.pid").read().strip() if os.path.exists("/content/blur.pid") else ""
alive = bool(pid) and os.path.exists("/proc/" + pid)
done = bool(os.path.exists("/content/PHASE2_DONE") or prog.get("done") is True)
log = ""
if os.path.exists("/content/phase2.log"):
    try:
        log = open("/content/phase2.log", errors="replace").read()[-12000:]
    except Exception:
        log = ""
crash = (not done) and (prog.get("crash") is True or (bool(pid) and not alive) or
                        ("Traceback" in log and not alive and not done))
out["blur"] = {
    "vals": {},
    "at": str(prog.get("step") or ""),
    "done": done,
    "crash": crash,
    "alive": alive,
    "launched": bool(pid) or bool(prog),
    "ckpts": {},
    "progress": {k: prog[k] for k in prog if k != "preflight"},
}
try:
    out["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out["ram_avail_g"] = subprocess.run(
        ["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()[-1]
    out["df_content"] = subprocess.run(
        ["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1]
except Exception:
    pass
if os.path.exists("/content/preflight.json"):
    try:
        out["preflight"] = json.load(open("/content/preflight.json"))
    except Exception:
        pass
print("PROBE_JSON=" + json.dumps(out))
'''


def prepare_vm(st):
    if not orch.preflight_transfer(st):
        return "transfer_broken"
    tgz = os.path.join(HERE, "adattt_phase2.tgz")
    if not os.path.isfile(tgz):
        orch.log("  adattt_phase2.tgz missing — run build_payload_phase2.py")
        return "failed"
    if not os.path.isfile(CLIP_CKPT):
        orch.log(f"  CLIP checkpoint missing: {CLIP_CKPT}")
        return "failed"
    uploads = [
        (tgz, "/content/adattt_phase2.tgz"),
        (os.path.join(HERE, "vm_setup_phase2.sh"), "/content/vm_setup.sh"),
        (os.path.join(HERE, "vm_launch_phase2.sh"), "/content/vm_launch.sh"),
    ]
    for local, remote in uploads:
        rc, out = orch.cli(["upload", "-s", SESSION, local, remote], 600)
        if rc != 0:
            orch.log(f"  upload {os.path.basename(local)} failed: {out.strip()[-160:]}")
            return "failed"
    rc, out = orch.cli(
        ["exec", "-s", SESSION, "-f", os.path.join(HERE, "vm_bootstrap.py"),
         "--timeout", "60"], 120)
    if "setup started" not in out:
        orch.log(f"  setup bootstrap failed: {out.strip()[-200:]}")
        return "failed"
    orch.log("  setup started on the VM (val2014 download)")
    st["setup_ts"] = orch.now()
    orch.save_state(st)

    orch.vm_exec("import os; os.makedirs('/content/resume', exist_ok=True)", 90)
    orch.log("  uploading phase1_clip/best.pt (read-only copy onto the VM)")
    if not orch.upload_file(CLIP_CKPT, "/content/resume/clip_best.pt", "clip_best"):
        orch.log("  CLIP checkpoint upload failed")
        return "failed"
    rc, out = orch.vm_exec(
        "open('/content/resume.args','w').write('')\n"
        "open('/content/launch.ready','w').write('1')\nprint('READY_OK')",
        90,
    )
    if "READY_OK" not in out:
        orch.log("  could not write launch.ready")
        return "failed"
    session = _phase2_session()
    orch.log(f"  launch armed (session {session.ACTIVE_SESSION.upper()}: "
             f"{', '.join(c['id'] for c in session.active_conditions())})")
    return "ok"


def _fetch_results(st):
    """Pull compact json/npz/logs. Never a .pt cache."""
    os.makedirs(RESULT_LOCAL, exist_ok=True)
    landed = []
    missing = []
    orch.fetch_small("/content/preflight.json", os.path.join(WORK, "preflight.json"))
    orch.fetch_small("/content/phase2_progress.json", os.path.join(WORK, "phase2_progress.json"))
    orch.fetch_small("/content/PHASE2_DONE", os.path.join(WORK, "PHASE2_DONE"))
    orch.fetch_small("/content/phase2.log", os.path.join(RESULT_LOCAL, "phase2.log"))
    orch.fetch_small("/content/setup.log", os.path.join(WORK, "setup.log"))
    session = _phase2_session()
    for cond in session.active_conditions():
        remote_dir = f"{RESULT_REMOTE}/{cond['result_name']}"
        local_dir = os.path.join(RESULT_LOCAL, cond["result_name"])
        os.makedirs(local_dir, exist_ok=True)
        for name in session.artifacts_for(cond):
            remote = f"{remote_dir}/{name}"
            local = os.path.join(local_dir, name)
            rel = f"{cond['result_name']}/{name}"
            if orch.fetch_small(remote, local):
                landed.append(rel)
            else:
                missing.append(rel)
    orch.vm_exec(
        "import glob, os\n"
        "paths=glob.glob('/content/phase2_cache/*.pt')\n"
        "print('CACHE_PRESENT=' + str(bool(paths)))\n"
        "print('CACHE_SIZE=' + str(sum(os.path.getsize(p) for p in paths)))",
        90,
    )
    orch.log(f"  landed {landed}; missing {missing}")
    return landed, missing


def finalize(st, probe):
    landed, missing = _fetch_results(st)
    session = _phase2_session()
    conditions = session.active_conditions()
    essential = {
        f"{c['result_name']}/{name}" for c in conditions
        for name in session.essential_artifacts(c)
    }
    summary = {
        "session": session.ACTIVE_SESSION.upper(),
        "conditions": [c["id"] for c in conditions],
        "subset": "data/eval_subset_8k.json",
        "landed": landed,
        "missing": missing,
        "progress": (probe.get("blur") or {}).get("progress"),
        "preflight": probe.get("preflight"),
        "vm_hours": round(orch.vm_hours(st), 2),
        "allocations": st["allocations"],
    }
    os.makedirs(RESULT_LOCAL, exist_ok=True)
    json.dump(summary, open(os.path.join(RESULT_LOCAL, "orch_summary.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(WORK, "run_summary.json"), "w"), indent=2)
    still = [m for m in missing if m in essential]
    return summary, still


def _have_live_vm(st):
    """Attach to the namespaced session / recorded endpoint. Never `colab new` over it."""
    alive = orch.session_alive()
    if alive is True:
        token = orch.keep_token_fresh(st)
        orch.log(f"  session {SESSION} is live (refresh={token})")
        return True
    ep = st.get("endpoint")
    if not ep:
        return False
    rc, out = orch.helper(["adopt", SESSION, ep], 180)
    orch.log(f"  adopt {ep} rc={rc} {out.strip()[:120]}")
    return rc == 0


def run_loop(st):
    have_vm = False
    if _have_live_vm(st):
        have_vm = True
        if not st.get("alloc_ts"):
            st["alloc_ts"] = orch.now()
            orch.save_state(st)
        orch.log(f"attaching to existing VM {st.get('endpoint')}")
    elif st.get("endpoint"):
        orch.log(f"recorded endpoint {st['endpoint']} is gone; will allocate a replacement")
        orch.release_orphans(st)
    fails = 0
    while True:
        if orch.vm_hours(st) >= orch.BUDGET_H:
            orch.stop_vm(st, "budget reached")
            return 5

        if not have_vm:
            if orch.session_alive() is True:
                orch.log("session already live; attaching instead of allocating")
                have_vm = True
                continue
            if st["allocations"] >= orch.MAX_ALLOC:
                orch.sweep_failed_new(st, adopt=False)
                orch.log("out of allocations")
                return 7
            status = orch.allocate(st)
            if status == "exec_failed":
                orch.stop_vm(st, "exec path broken")
                return 2
            if status == "wrong_gpu":
                orch.stop_vm(st, "wrong GPU")
                return 3
            if status == "alloc_failed":
                orch.log(f"  no A100 assigned; retrying in {orch.ALLOC_RETRY_S}s")
                time.sleep(orch.ALLOC_RETRY_S)
                continue
            prep = prepare_vm(st)
            if prep == "transfer_broken":
                orch.stop_vm(st, "transfer path broken")
                return 9
            if prep != "ok":
                orch.stop_vm(st, "prepare failed")
                continue
            have_vm, fails = True, 0
            st["projection_checked"] = True  # no train-time projection for this job
            orch.save_state(st)

        time.sleep(orch.POLL_S)

        token = orch.keep_token_fresh(st)
        alive = False if token == "gone" else (True if token in ("ok", "readopted") else None)
        if alive is False:
            orch.log("VM LOST — assignment no longer listed; resuming on a fresh VM")
            st["vm_seconds"] += orch.now() - st["alloc_ts"] if st["alloc_ts"] else 0
            st["alloc_ts"], st["launch_ts"], st["setup_ts"] = None, None, None
            st["events"].append(f"{__import__('datetime').datetime.now():%H:%M} vm lost")
            orch.save_state(st)
            have_vm = False
            continue

        rc, out = orch.vm_exec(orch.PROBE, 180)
        raw = orch.marker(out, "PROBE_JSON")
        if raw is None:
            fails += 1
            orch.log(f"probe failed ({fails}/{orch.WEDGED_AFTER}): {out.strip()[-140:]}")
            if fails >= orch.WEDGED_AFTER and alive is not False:
                orch.stop_vm(st, f"exec path wedged for {fails} polls (liveness {alive})")
                have_vm, fails = False, 0
            continue
        fails = 0
        p = json.loads(raw)

        if p["setup"] == "failed":
            orch.stop_vm(st, "setup failed on the VM")
            have_vm = False
            continue
        if (p["setup"] in ("running", "absent") and st.get("setup_ts")
                and orch.now() - st["setup_ts"] > orch.SETUP_TIMEOUT_S):
            orch.stop_vm(st, f"setup still running after {orch.SETUP_TIMEOUT_S // 60} min")
            have_vm = False
            continue

        info = p.get("blur") or {}
        if info.get("launched") and st.get("launch_ts") is None:
            st["launch_ts"] = orch.now()
            orch.save_state(st)
        orch.fetch_small("/content/phase2_progress.json", os.path.join(WORK, "phase2_progress.json"))
        orch.fetch_small("/content/preflight.json", os.path.join(WORK, "preflight.json"))
        lines = [
            f"setup {p['setup']} | gpu {p.get('gpu', '?')} | ram free {p.get('ram_avail_g', '?')}G",
            f"blur step {info.get('at') or '-'} alive={info.get('alive')} "
            f"done={info.get('done')} launched={info.get('launched')}",
        ]
        orch.write_status(st, "\n".join(lines))
        orch.log(" | ".join(lines))

        if info.get("crash") or (
            info.get("launched") and not info.get("alive") and not info.get("done")
        ):
            orch.log("blur CRASHED or died — fetching logs then stopping")
            _fetch_results(st)
            orch.stop_vm(st, "blur crashed")
            return 4

        if info.get("done"):
            orch.log("SESSION C COMPLETE — landing small artifacts (no cache)")
            summary, missing = finalize(st, p)
            if missing:
                orch.log(f"summary.json not landed {missing} — retrying once")
                time.sleep(15)
                summary, missing = finalize(st, p)
            # Always stop: a 12 GB cache must not keep an A100 billing.
            orch.stop_vm(st, "complete" if not missing else "complete, summary missing")
            orch.write_status(st, "COMPLETE\n" + json.dumps(summary, indent=1))
            return 0 if not missing else 8


orch.prepare_vm = prepare_vm
orch.finalize = finalize
orch.run_loop = run_loop


if __name__ == "__main__":
    if sys.argv[1:] == ["--emergency-stop"]:
        # Guard normally re-invokes orch.py; keep this path so a misplaced
        # `python orch_phase2.py --emergency-stop` still stops the namespaced VM.
        os.makedirs(WORK, exist_ok=True)
        st = orch.load_state()
        orch.log("budget guard fired — emergency stop (phase2)")
        orch.cli(["stop", "-s", SESSION], 180)
        orch.release_orphans(st)
        orch.sweep_failed_new(st, adopt=False)
        sys.exit(0)
    sys.exit(orch.main())
