"""Scenario tests for orch.py against fake_colab.py — no Google calls."""
import importlib.util, json, os, shutil, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
REAL_PROJECT = "/Users/yugesh/Library/CloudStorage/GoogleDrive-yugeshreddysappidi@gmail.com/My Drive/AdaTTT"
CEILING = os.path.join(REAL_PROJECT, "scripts", "06_ceiling_check.py")
spec = importlib.util.spec_from_file_location("ceiling", CEILING)
ceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ceiling)
BEST = {"clip": 8, "vitbert": 6}  # argmax epoch of the fake's val curves
# What the 00:24 attempt left behind: a launch banner, then a crash before epoch 1.
STALE = ("[05:22:34] INFO - Device: cuda, AMP: True\n[05:23:53] INFO - Val samples: 214354\n"
         "Traceback (most recent call last):\n  TypeError: '<=' not supported\n")


def header_epoch(path):
    return int(open(path, "rb").read().split(b"\n", 1)[0].split(b"=")[1])


def artifacts(proj, sim, env):
    missing = [f"{r}/{f}" for r in ("clip", "vitbert") for f in ("best.pt", "epoch_7.pt")
               if not os.path.exists(os.path.join(proj, "checkpoints", f"phase1_{r}", f))]
    return not missing, "all present" if not missing else f"missing {missing}"


def best_is_argmax(proj, sim, env):
    got = {r: header_epoch(os.path.join(proj, "checkpoints", f"phase1_{r}", "best.pt")) for r in BEST}
    return got == BEST, f"best.pt epochs {got}, want {BEST}"


def logs_complete(proj, sim, env):
    out = {}
    for r in ("clip", "vitbert"):
        h = ceiling.read_history(os.path.join(proj, "logs", f"train_{r}.log"))
        out[r] = sorted(h) == list(range(1, 9))
    return all(out.values()), f"epochs 1..8 recoverable: {out}"


def ceiling_verdicts(proj, sim, env):
    j = json.load(open(os.path.join(proj, "results", "phase1", "ceiling_check.json")))
    v = {k: r["verdict"] for k, r in j["runs"].items()}
    return v == {"clip": "CONTINUE", "control (vit_bert)": "KILL"}, str(v)


def summary_written(proj, sim, env):
    p = os.path.join(proj, "results", "phase1", "run_summary.json")
    return os.path.exists(p), json.load(open(p)).get("allocations") if os.path.exists(p) else "absent"


def allocations(n):
    return lambda proj, sim, env: (sim["vm_id"] == n, f"{sim['vm_id']} VM(s) allocated, want {n}")


def resumed(run):
    def check(proj, sim, env):
        second = [h for h in sim["history"] if h["vm"] == 2]
        args = second[0]["launch_args"] if second else []
        return f"--resume-{run}" in args, f"VM 2 launch args: {args}"
    return check


def scenario(name, extra, want, checks, seed=None):
    base = os.path.join(S, "orchtest", name)
    shutil.rmtree(base, ignore_errors=True)
    for rel, content in (seed or {}).items():
        os.makedirs(os.path.dirname(os.path.join(base, "work", rel)), exist_ok=True)
        open(os.path.join(base, "work", rel), "w").write(content)
    proj = os.path.join(base, "project")
    os.makedirs(os.path.join(proj, "scripts"))
    shutil.copyfile(CEILING, os.path.join(proj, "scripts", "06_ceiling_check.py"))
    env = dict(os.environ, FAKE_DIR=os.path.join(base, "fake"), ORCH_COLAB=os.path.join(S, "fake_colab.py"),
               ORCH_PROJECT=proj, ORCH_WORK=os.path.join(base, "work"), ORCH_POLL_S="1",
               ORCH_PART_BYTES="65536", ORCH_GUARD="0", ORCH_PROJECTION_AFTER_S="2",
               ORCH_WEDGED_AFTER="3", ORCH_PREFLIGHT_SIZES="65536",
               ORCH_HELPER=f"{sys.executable} {os.path.join(S, 'fake_colab.py')} helper",
               FAKE_T_SETUP="1", FAKE_T_EPOCH="1.5")
    os.makedirs(env["FAKE_DIR"])
    env.update(extra)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(S, "orch.py")], env=env,
                           capture_output=True, text=True, timeout=240)
        code, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        code, out = "TIMEOUT", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    sim = json.load(open(os.path.join(env["FAKE_DIR"], "sim.json")))
    res = [("exit code", code == want, f"{code} (want {want})"),
           ("VM released", sim.get("endpoint") is None, f"endpoint={sim.get('endpoint')}"),
           ("never two VMs at once", sim.get("max_concurrent", 1) == 1,
            f"max concurrent assignments {sim.get('max_concurrent', 1)}")]
    for label, fn in checks:
        try:
            ok, detail = fn(proj, sim, env)
        except Exception as ex:
            ok, detail = False, f"{type(ex).__name__}: {ex}"
        res.append((label, ok, detail))
    passed = all(ok for _, ok, _ in res)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  ({time.time() - t0:.0f}s)")
    for label, ok, detail in res:
        print(f"    {'ok ' if ok else 'BAD'} {label}: {detail}")
    if not passed:
        print("    --- orchestrator output (tail) ---")
        print("\n".join("    " + l for l in out.splitlines()[-30:]))
    return passed


full = [("artifacts landed", artifacts), ("best.pt = argmax epoch", best_is_argmax),
        ("logs hold epochs 1..8", logs_complete), ("ceiling verdicts", ceiling_verdicts),
        ("summary written", summary_written)]
results = [
    scenario("happy_path", {}, 0, full + [("single VM", allocations(1))]),
    scenario("vm_lost_mid_run", {"FAKE_KILL": "1:clip:3"}, 0,
             full + [("re-allocated once", allocations(2)), ("clip resumed from relay", resumed("clip"))]),
    scenario("run_crash", {"FAKE_CRASH": "clip:2"}, 4, []),
    scenario("budget_reached", {"ORCH_BUDGET_H": "0.0015", "ORCH_PROJECTION_AFTER_S": "99999"}, 5, []),
    scenario("exec_path_broken", {"FAKE_EXEC_BROKEN": "1"}, 2, []),
    scenario("projection_over_budget", {"FAKE_REMAINING_H": "50"}, 6, []),
    scenario("transfer_fallback", {"FAKE_MAX_DOWNLOAD_BYTES": "100000",
                                   "ORCH_PREFLIGHT_SIZES": "262144,65536"}, 0,
             full + [("single VM", allocations(1))]),
    scenario("transfer_broken", {"FAKE_MAX_DOWNLOAD_BYTES": "1000"}, 9, []),
    scenario("setup_hangs_on_first_vm", {"FAKE_HANG_SETUP": "1", "ORCH_SETUP_TIMEOUT_S": "4"}, 0,
             full + [("second VM after the hang", allocations(2))]),
    scenario("token_refreshed_every_poll", {"FAKE_TOKEN_TTL": "4", "FAKE_ROTATE": "1"}, 0,
             full + [("single VM", allocations(1))]),
    scenario("token_expires_then_readopted", {"FAKE_TOKEN_TTL": "10", "FAKE_ROTATE": "0"}, 0,
             full + [("single VM — progress kept", allocations(1)),
                     ("re-adopted at least once", lambda p, sim, e: (sim.get("adoptions", 0) >= 1, f"{sim.get('adoptions', 0)} adoption(s)"))]),
    scenario("orphan_released_when_adoption_fails",
             {"FAKE_TOKEN_TTL": "10", "FAKE_ROTATE": "0", "FAKE_ADOPT_FAIL": "1", "FAKE_RECLAIM_S": "1000"}, 0,
             full + [("second VM after releasing the first", allocations(2)),
                     ("orphan was unassigned", lambda p, sim, e: (sim.get("unassigns", 0) >= 1, f"{sim.get('unassigns', 0)} unassign(s)"))]),
    scenario("stale_traceback_in_restored_log", {}, 0, full + [("single VM", allocations(1))],
             seed={f"{r}/train_{r}.log": STALE for r in ("clip", "vitbert")}),
    scenario("new_fails_after_assigning_is_adopted", {"FAKE_NEW_FAILS_AFTER_ASSIGN": "1"}, 0,
             full + [("single VM: the stray one adopted", allocations(1)),
                     ("adopted", lambda p, sim, e: (sim.get("adoptions", 0) >= 1,
                                                    f"{sim.get('adoptions', 0)} adoption(s)"))]),
    scenario("new_fails_after_assigning_released_when_adopt_fails",
             {"FAKE_NEW_FAILS_AFTER_ASSIGN": "1", "FAKE_ADOPT_FAIL": "1",
              "FAKE_RECLAIM_S": "1000", "ORCH_ALLOC_RETRY_S": "1"}, 0,
             full + [("second VM after releasing the stray", allocations(2)),
                     ("stray was unassigned", lambda p, sim, e: (sim.get("unassigns", 0) >= 1,
                                                                 f"{sim.get('unassigns', 0)} unassign(s)"))]),
    scenario("late_stray_from_failed_new_is_adopted",
             {"FAKE_NEW_ASSIGNS_LATE": "1", "FAKE_LATE_S": "2", "ORCH_ALLOC_RETRY_S": "3",
              "FAKE_RECLAIM_S": "1000"}, 0,
             full + [("single VM: the late stray adopted", allocations(1)),
                     ("adopted", lambda p, sim, e: (sim.get("adoptions", 0) >= 1,
                                                    f"{sim.get('adoptions', 0)} adoption(s)"))]),
    scenario("late_stray_released_before_giving_up",
             {"FAKE_NEW_ASSIGNS_LATE": "1", "FAKE_LATE_S": "1", "ORCH_ALLOC_RETRY_S": "3",
              "ORCH_MAX_ALLOC": "1", "FAKE_RECLAIM_S": "1000"}, 7,
             [("late stray unassigned", lambda p, sim, e: (sim.get("unassigns", 0) >= 1,
                                                          f"{sim.get('unassigns', 0)} unassign(s)"))]),
]
def emergency_stop_releases_pruned_orphan():
    """The budget guard's stop must reach a VM whose session the CLI already pruned."""
    base = os.path.join(S, "orchtest", "emergency_stop")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(os.path.join(base, "fake")); os.makedirs(os.path.join(base, "work"))
    fake = os.path.join(S, "fake_colab.py")
    env = dict(os.environ, FAKE_DIR=os.path.join(base, "fake"), FAKE_TOKEN_TTL="0.5",
               ORCH_COLAB=fake, ORCH_WORK=os.path.join(base, "work"),
               ORCH_HELPER=f"{sys.executable} {fake} helper")
    subprocess.run([fake, "new", "-s", "adattt-p1"], env=env, capture_output=True)
    time.sleep(1)
    subprocess.run([fake, "exec", "-s", "adattt-p1"], env=env, input="print(1)", capture_output=True, text=True)
    sim = json.load(open(os.path.join(base, "fake", "sim.json")))
    pruned = sim["session"] is None and sim["endpoint"] == "fake-ep-1"
    json.dump({"endpoints": ["fake-ep-1"]}, open(os.path.join(base, "work", "state.json"), "w"))
    subprocess.run([sys.executable, os.path.join(S, "orch.py"), "--emergency-stop"], env=env, capture_output=True)
    sim = json.load(open(os.path.join(base, "fake", "sim.json")))
    ok = pruned and sim["endpoint"] is None
    print(f"{'PASS' if ok else 'FAIL'}  emergency_stop_releases_pruned_orphan")
    print(f"    {'ok ' if pruned else 'BAD'} session pruned, VM still assigned before the stop")
    print(f"    {'ok ' if sim['endpoint'] is None else 'BAD'} VM released by the guard's stop: endpoint={sim['endpoint']}")
    return ok


results.append(emergency_stop_releases_pruned_orphan())
print(f"\n{sum(results)}/{len(results)} scenarios passed")
sys.exit(0 if all(results) else 1)
