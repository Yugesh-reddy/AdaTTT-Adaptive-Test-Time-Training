#!/usr/bin/env python3
"""
Fake `colab` CLI for testing orch.py without touching Google.

Simulates one VM on local disk under $FAKE_DIR/vm (mapped as the VM's /): an
allocation, a setup that finishes after FAKE_T_SETUP s, auto-launch once
/content/launch.ready exists, an epoch every FAKE_T_EPOCH s with atomic
checkpoints and best.pt on strict improvement, resume from an uploaded
checkpoint, and injectable failures:

  FAKE_KILL=<vm>:<run>:<epoch>  lose the VM (disk wiped) once <run> logs <epoch> on VM number <vm>
  FAKE_CRASH=<run>:<epoch>      that run dies with a Traceback after <epoch>
  FAKE_EXEC_BROKEN=1            every exec fails before reaching the VM
  FAKE_REMAINING_H=<h>          what vm_throughput.py reports
  FAKE_MAX_DOWNLOAD_BYTES=<n>   downloads larger than n fail mid-response
  FAKE_HANG_SETUP=<vm>          setup never finishes on VM number <vm>
  FAKE_TOKEN_TTL=<s>            runtime token lifetime; past it, exec/upload/download
                                404, the session is pruned and the VM orphaned
  FAKE_ROTATE=0|1               whether `helper refresh` issues a new token (default 1)
  FAKE_RECLAIM_S=<s>            an orphaned VM is reclaimed after this long (default 60)
  FAKE_ADOPT_FAIL=<vm>          `helper adopt` fails on VM number <vm>
  FAKE_NEW_FAILS_AFTER_ASSIGN=<vm>[,<vm>]
                                `new` reports failure on those VM numbers after the
                                server already assigned the VM (no local session)
  FAKE_NEW_ASSIGNS_LATE=<vm>[,<vm>]
                                `new` reports failure with nothing assigned; the VM
                                is assigned FAKE_LATE_S seconds later (default 2)
"""
import json, os, shutil, subprocess, sys, time

FAKE = os.environ["FAKE_DIR"]
SIM = os.path.join(FAKE, "sim.json")
ROOT = os.path.join(FAKE, "vm")
T_SETUP = float(os.environ.get("FAKE_T_SETUP", "1"))
T_EPOCH = float(os.environ.get("FAKE_T_EPOCH", "1.5"))
PAD = int(os.environ.get("FAKE_PAD", "200000"))
RUNS = {"vitbert": ("/content/AdaTTT/logs/train_vitbert.log", "/content/AdaTTT/checkpoints/phase1_vitbert/base"),
        "clip": ("/content/AdaTTT/logs/train_clip.log", "/content/AdaTTT/checkpoints/base")}
VALS = {"vitbert": [42.21, 44.95, 46.41, 47.56, 49.11, 49.12, 49.10, 49.08],
        "clip": [47.90, 51.20, 53.60, 55.10, 56.40, 57.50, 58.30, 58.90]}


def vm(path):
    # Idempotent: exec code is path-rewritten, so a path the VM wrote into a
    # file (resume.args) comes back already mapped and must not map twice.
    return path if path.startswith(ROOT) else os.path.join(ROOT, path.lstrip("/"))


def load():
    return json.load(open(SIM)) if os.path.exists(SIM) else {"session": None, "endpoint": None, "vm_id": 0, "history": []}


def save(s):
    json.dump(s, open(SIM + ".tmp", "w"), indent=1)
    os.replace(SIM + ".tmp", SIM)


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path + ".tmp", "wb").write(data)
    os.replace(path + ".tmp", path)


def advance(s):
    late = s.get("late_assign")
    if late and time.time() >= late["at"]:
        if s.get("endpoint"):
            s["max_concurrent"] = 2  # assigned while another VM was still assigned
        s.update({"endpoint": late["ep"], "session": None, "orphan_since": time.time(),
                  "late_assign": None})
    reclaim = float(os.environ.get("FAKE_RECLAIM_S", "60"))
    if s.get("orphan_since") and time.time() - s["orphan_since"] > reclaim:
        s["endpoint"], s["orphan_since"] = None, None
        shutil.rmtree(ROOT, ignore_errors=True)
        return
    # Training runs detached on the VM: it continues while the VM is assigned,
    # whether or not the CLI still has a session for it.
    if not s.get("endpoint") or s.get("setup_start") is None:
        return
    t = time.time()
    hang = os.environ.get("FAKE_HANG_SETUP")
    if not s.get("setup_done") and t - s["setup_start"] >= T_SETUP and hang != str(s["vm_id"]):
        with open(vm("/content/setup.log"), "a") as f:
            f.write("[00:00:10] SETUP_DONE\n")
        s["setup_done"] = True
    if s.get("setup_done") and not s.get("launch_t") and os.path.exists(vm("/content/launch.ready")):
        rp = vm("/content/resume.args")
        args = open(rp).read().split() if os.path.exists(rp) else []
        s["launch_t"] = t
        s["history"].append({"vm": s["vm_id"], "launch_args": args})
        os.makedirs(vm("/proc"), exist_ok=True)
        for run, (logp, _) in RUNS.items():
            if f"--skip-{run}" in args:
                s["runs"][run] = {"skip": True, "done": True}
                continue
            start = 0
            if f"--resume-{run}" in args:
                ck = args[args.index(f"--resume-{run}") + 1]
                start = int(open(vm(ck), "rb").read().split(b"\n", 1)[0].split(b"=")[1])
            restored = vm(f"/content/resume/train_{run}.log")
            os.makedirs(os.path.dirname(vm(logp)), exist_ok=True)
            if os.path.exists(restored) and not os.path.exists(vm(logp)):
                shutil.copyfile(restored, vm(logp))
            with open(vm(logp), "a") as f:
                f.write("[00:00:00] INFO - Device: cuda, AMP: True\n")
            pid = f"{s['vm_id']}{'0' if run == 'vitbert' else '1'}77"
            open(vm(f"/content/{run}.pid"), "w").write(pid)
            open(vm(f"/proc/{pid}"), "w").write("alive")
            s["runs"][run] = {"start": start, "logged": start, "pid": pid, "done": False,
                              "best_val": max(VALS[run][:start]) if start else -1.0}
    if not s.get("launch_t"):
        return
    for run, r in s["runs"].items():
        if r.get("skip") or r.get("done") or r.get("crashed"):
            continue
        logp, ck = RUNS[run]
        target = min(8, r["start"] + int((t - s["launch_t"]) / T_EPOCH))
        while r["logged"] < target:
            e = r["logged"] + 1
            val = VALS[run][e - 1]
            os.makedirs(os.path.dirname(vm(logp)), exist_ok=True)
            with open(vm(logp), "a") as f:
                f.write(f"[00:00:00] INFO - Epoch {e}/8, Batch 6900/6934, Loss: 0.1000\n")
                f.write(f"[00:00:00] INFO - Epoch {e} | Val accuracy: {val:.2f}%\n")
                # Soft sits above exact by the gap between v1's two references
                # (54.30 vs 49.56), so both metrics reach the same verdicts.
                f.write(f"[00:00:00] INFO - Epoch {e} | Official VQA soft: {val + 4.74:.2f}%\n")
            body = f"EPOCH={e}\n".encode() + os.urandom(PAD)
            write_atomic(os.path.join(vm(ck), f"epoch_{e - 1}.pt"), body)
            if val > r["best_val"]:
                r["best_val"] = val
                write_atomic(os.path.join(vm(ck), "best.pt"), body)
            r["logged"] = e
            if os.environ.get("FAKE_CRASH") == f"{run}:{e}":
                with open(vm(logp), "a") as f:
                    f.write("Traceback (most recent call last):\n  RuntimeError: boom\n")
                os.remove(vm(f"/proc/{r['pid']}"))
                r["crashed"] = True
                break
            if e == 8:
                with open(vm(logp), "a") as f:
                    f.write(f"\nTraining complete! Best val accuracy: {r['best_val']:.2f}%\n")
                os.remove(vm(f"/proc/{r['pid']}"))
                r["done"] = True
    kill = os.environ.get("FAKE_KILL")
    if kill:
        kv, krun, kep = kill.split(":")
        if s["vm_id"] == int(kv) and s["runs"].get(krun, {}).get("logged", 0) >= int(kep):
            s["session"], s["endpoint"], s["lost"] = None, None, f"killed at {krun} epoch {kep}"
            shutil.rmtree(ROOT, ignore_errors=True)


def main():
    a = sys.argv[1:]
    s = load()
    advance(s)
    save(s)
    opt = lambda flag: a[a.index(flag) + 1] if flag in a else None
    cmd = a[0] if a else ""
    if cmd == "new":
        if s.get("endpoint"):
            s["max_concurrent"] = 2  # allocated while another VM was still assigned
        vm_id = s["vm_id"] + 1
        s.update({"session": opt("-s"), "endpoint": f"fake-ep-{vm_id}", "vm_id": vm_id,
                  "token_t": time.time(), "orphan_since": None, "runs": {}, "setup_start": None,
                  "setup_done": False, "launch_t": None, "lost": None})
        shutil.rmtree(ROOT, ignore_errors=True)
        os.makedirs(vm("/content"))
        if str(vm_id) in os.environ.get("FAKE_NEW_FAILS_AFTER_ASSIGN", "").split(","):
            s["session"], s["orphan_since"] = None, time.time()
            save(s)
            print("[colab] Error: runtime did not become ready. Try again later.")
            return 1
        if str(vm_id) in os.environ.get("FAKE_NEW_ASSIGNS_LATE", "").split(","):
            s["session"], s["endpoint"] = None, None
            s["late_assign"] = {"ep": f"fake-ep-{vm_id}",
                                "at": time.time() + float(os.environ.get("FAKE_LATE_S", "2"))}
            save(s)
            print("[colab] Error: runtime did not become ready. Try again later.")
            return 1
        save(s)
        print(f"[colab] Creating session '{s['session']}'...\n[colab] Session READY.")
        return 0
    if cmd == "sessions":
        if s.get("endpoint"):
            print(f"[{s.get('session') or '?'}] {s['endpoint']} | Hardware: A100 | Variant: GPU")
        else:
            print("[colab] No active sessions found on server.")
        return 0
    if cmd == "stop":
        if s.get("session") and s["session"] == opt("-s"):
            s["session"], s["endpoint"], s["orphan_since"] = None, None, None
            s["stops"] = s.get("stops", 0) + 1
            save(s)
            print("[colab] Session terminated.")
            return 0
        print(f"[colab] Error: session '{opt('-s')}' not found.")
        return 1
    if cmd == "helper":
        sub, rest = a[1], a[2:]
        ttl = float(os.environ.get("FAKE_TOKEN_TTL", "0") or 0) or 3600
        if sub == "show":
            if s.get("session") != rest[0]:
                return 4
            print(f"ENDPOINT={s['endpoint']}\nTTL={int(ttl)}")
            return 0
        if sub == "refresh":
            if not s.get("endpoint"):
                return 3
            if s.get("session") != rest[0]:
                return 4
            rotate = os.environ.get("FAKE_ROTATE", "1") != "0"
            if rotate:
                s["token_t"] = time.time()
            save(s)
            print(f"REFRESH_OK ttl={int(ttl)} changed={rotate}")
            return 0
        if sub == "adopt":
            name, ep = rest
            if s.get("endpoint") != ep:
                return 3
            if os.environ.get("FAKE_ADOPT_FAIL") == str(s["vm_id"]):
                print("adopt failed")
                return 5
            s["session"], s["token_t"], s["orphan_since"] = name, time.time(), None
            s["adoptions"] = s.get("adoptions", 0) + 1
            save(s)
            print("ADOPTED")
            return 0
        if sub == "orphans":
            if s.get("endpoint") and not s.get("session"):
                print(f"ORPHAN={s['endpoint']} ACC=A100")
            return 0
        if sub == "unassign":
            if s.get("endpoint") != rest[0]:
                return 3
            s["session"], s["endpoint"], s["orphan_since"] = None, None, None
            s["unassigns"] = s.get("unassigns", 0) + 1
            shutil.rmtree(ROOT, ignore_errors=True)
            save(s)
            print("UNASSIGNED")
            return 0
        return 2
    ttl = float(os.environ.get("FAKE_TOKEN_TTL", "0") or 0)
    if cmd in ("exec", "upload", "download") and s.get("session") and ttl and time.time() - s["token_t"] > ttl:
        s["session"], s["orphan_since"] = None, time.time()
        save(s)
        print(f"[colab] Session '{opt('-s')}' appears to be lost (404/401). Cleaning up.")
        return 1
    if not s["session"]:
        print(f"[colab] Session '{opt('-s')}' appears to be lost (404/401). Cleaning up.")
        return 1
    if cmd == "upload":
        local, remote = a[-2], a[-1]
        if not os.path.isdir(os.path.dirname(vm(remote))):
            print(f"FileNotFoundError: File or directory not found: {remote}")
            return 1
        shutil.copyfile(local, vm(remote))
        print(f"[colab] Uploaded '{local}' to '{remote}'")
        return 0
    if cmd == "download":
        remote, local = a[-2], a[-1]
        if not os.path.exists(vm(remote)):
            print(f"FileNotFoundError: File or directory not found: {remote}")
            return 1
        limit = int(os.environ.get("FAKE_MAX_DOWNLOAD_BYTES", "0"))
        if limit and os.path.getsize(vm(remote)) > limit:
            print("requests.exceptions.ChunkedEncodingError: Response ended prematurely")
            return 1
        shutil.copyfile(vm(remote), local)
        return 0
    if cmd == "exec":
        if os.environ.get("FAKE_EXEC_BROKEN") == "1":
            print("AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'")
            return 1
        f = opt("-f")
        if f:
            base = os.path.basename(f)
            if base == "gpu_check.py":
                print("GPU_NAME=NVIDIA A100-SXM4-40GB\nCUDA_AVAILABLE=True torch=2.11.0 cuda=12.8\nVCPUS=12")
                return 0
            if base == "vm_bootstrap.py":
                with open(vm("/content/setup.log"), "a") as fh:
                    fh.write("[00:00:00] setup start\n")
                s["setup_start"] = time.time()
                save(s)
                print("setup started, pid 4242")
                return 0
            if base == "vm_throughput.py":
                h = os.environ.get("FAKE_REMAINING_H", "0.01")
                print(f"vitbert: 0.4 s/batch | REMAINING_H={h}\nclip: 0.4 s/batch | REMAINING_H={h}")
                return 0
            code = open(f).read()
        else:
            code = sys.stdin.read()
        code = code.replace("/content", vm("/content")).replace('"/proc/"', repr(vm("/proc") + "/"))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=vm("/content"))
        sys.stdout.write(p.stdout + p.stderr)
        return p.returncode
    print(f"fake colab: unsupported {a}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
