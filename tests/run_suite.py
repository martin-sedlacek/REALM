"""Run REALM's test suite end to end and record what actually happened.

WHY THIS EXISTS, AND WHY IT IS NOT PYTEST
-----------------------------------------
Every file in tests/ is named `test_*.py` but NONE of them defines a pytest-collectable test:
there is no `def test_*`, no `class Test*`, no `import pytest`. `pytest tests/` collects zero
items -- and it collects them by IMPORTING each module, which for three of these files means
booting a full Isaac instance at module scope purely to find nothing. (pytest IS installed in the
container, at /opt/conda/envs/behavior/lib/python3.11/site-packages; it is absent from the login
python. So "pytest is missing" is not the reason.) They are standalone scripts with a
`if __name__ == "__main__":` block and `sys.exit(1)` on failure, and that is how this driver runs
them: `python -u tests/<file>.py`, one process each.

WHY THE EXIT CODE IS RECORDED BUT NEVER GATED ON
------------------------------------------------
Isaac tears down with a segfault on essentially every run, passing or failing, after all work is
done -- so `returncode` carries no information about the test. Every verdict here comes from
matching the test's own printed verdict lines. The exit code is stored in the JSON as an
observation, never as a pass condition. This is the same rule tests/test_vector_integrity.py
applies to its child processes, for the same reason.

`-u` on every child: Isaac's teardown can hang, so a time-limit kill is routine, and a
block-buffered child loses everything it printed when it is killed.

RESULTS ARE WRITTEN BEFORE THEY ARE PRINTED. The JSON is rewritten after every test, so a driver
that is itself killed still leaves a complete record of the tests that finished.

    python3 tests/run_suite.py --jobid 191441 --out /path/results.json --only fast
    python3 tests/run_suite.py --jobid 191441 --list
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
RR = str(PROJECT_ROOT / "scripts/clara/interactive/rr")

# ---------------------------------------------------------------------------------------------
# The suite. `verdict` is an ORDERED list of (regex, status). Each pattern is searched against the
# WHOLE log and the first pattern that matches anywhere wins -- not the match that appears earliest
# in the file. So failure patterns must come before success patterns: a sweep that fails one cell
# and passes the rest must read FAIL, however late the failing line appears.
# `cells` extracts the per-item lines a sweep prints, for the detail column.
# ---------------------------------------------------------------------------------------------
SUITE = {
    "test_task_progression_rubrics": dict(
        argv=["tests/test_task_progression_rubrics.py"],
        local=True,   # runs on the login python: no container, no allocation, ~0.06 s
        needs_gpu=False, needs_server=False, timeout=120, tier="local",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^\[\d\] .*",
        note="rubric stages vs success_conditions; static, no container.",
    ),
    "test_joint_reset_batching": dict(
        argv=["tests/test_joint_reset_batching.py"],
        needs_gpu=False, needs_server=False, timeout=900, tier="fast",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^\[\d\] .*",
        note="stubbed og.sim; no simulator, no GPU. Asserts scheduling only.",
    ),
    "test_single_task": dict(
        argv=["tests/test_single_task.py", "--task_id", "0"],
        needs_gpu=True, needs_server=False, timeout=1800, tier="medium",
        verdict=[(r"^Task \d+ \(.*\) FAILED!", "FAIL"),
                 (r"^Evaluation failed for ", "FAIL"),
                 (r"^Task \d+ \(.*\) PASSED!", "PASS")],
        cells=r"^  \w+_(?:csv|parquet): \w+",
        note="one task, 1 step, 1 repeat, --no_render.",
    ),
    "test_single_task_drawer": dict(
        # Task 8 is open_drawer, whose main object is custom_assets/impact_drawer/usd/cabinet.usd.
        # It is the ONLY task that needs OmniSurfaceMaterialPrim's preset_name default, which the
        # stock 3.9.1 image does not have and OG-lite / stock_patch do. Run this under MODE=oglite
        # against the MODE=stock result from test_integrity: same task, same code, different bind.
        argv=["tests/test_single_task.py", "--task_id", "8"],
        needs_gpu=True, needs_server=False, timeout=1800, tier="medium",
        verdict=[(r"^Task \d+ \(.*\) FAILED!", "FAIL"),
                 (r"^Evaluation failed for ", "FAIL"),
                 (r"^Task \d+ \(.*\) PASSED!", "PASS")],
        cells=r"^  \w+_(?:csv|parquet): \w+",
        note="task 8 open_drawer -- the mode-dependence control for the stock preset_name gap.",
    ),
    "test_integrity": dict(
        argv=["tests/test_integrity.py"],
        needs_gpu=True, needs_server=False, timeout=10800, tier="slow",
        verdict=[(r"^\S+: FAILED EXECUTION", "FAIL"),
                 (r"^\S+: FAIL \(", "FAIL"),
                 (r"^ALL TASKS PASSED INTEGRITY CHECK!", "PASS")],
        cells=r"^(?:--- Testing Task .*|Ran evaluation for .*|CRASHED during .*|"
              r"\S+: (?:PASS|FAIL|FAILED EXECUTION)\b.*)$",
        note="10 tasks x 1 step x 1 repeat, --no_render.",
    ),
    "test_perturbations_integrity": dict(
        # --repeats 3 --max_steps 1 is the invocation docs/og391_cluster_port_prompt.md calls
        # "the reference result ... 16/16", so run exactly that rather than a cheaper variant whose
        # outcome could not be compared against it. It also costs almost nothing: the Isaac boot
        # dominates a cell, and it is the only place in the suite that exercises the per-repeat
        # reset path.
        argv=["tests/test_perturbations_integrity.py", "--repeats", "3", "--max_steps", "1"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        verdict=[(r"^\S+: FAILED EXECUTION", "FAIL"),
                 (r"^\S+: FAIL \(", "FAIL"),
                 (r"^ALL PERTURBATIONS PASSED INTEGRITY CHECK!", "PASS")],
        cells=r"^(?:--- Testing Perturbation .*|Ran evaluation for .*|CRASHED during .*|"
              r"[\w-]+: (?:PASS|FAIL|FAILED EXECUTION)\b.*)$",
        note="16 perturbations on task 0, 3 repeats x 1 step, rendering ON.",
    ),
    "test_vector_integrity_tasks": dict(
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN)\b.*)$",
        note="10 tasks under Default through the vector path, num_envs=2, rendering ON.",
    ),
    "test_vector_integrity_tasks_shard0of2": dict(
        # Half the task matrix -- cells 0,2,4,6,8, i.e. tasks 0,2,4,6,8 -- for when the allocation
        # cannot hold the full ten. Deliberately a COMPLETED half rather than a truncated whole: a
        # run killed on the time limit never prints its verdict line. The sample is not arbitrary:
        # it covers a PrimitiveObject main object (0), a rotate task (2), a DatasetObject pick (4),
        # a stack (6) and open_drawer (8) -- the one that could not load at all until 2026-08-14.
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2",
              "--shard", "0/2"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN)\b.*)$",
        note="tasks 0,2,4,6,8 under Default through the vector path, num_envs=2, rendering ON.",
    ),
    "test_vector_integrity_drawers": dict(
        # The two drawer cells only. A sibling measured 8:Default crashing under MODE=stock with
        # material_prim.py's missing preset_name default; this is that cell, re-run against the
        # rebuilt image. Vector rather than single-env because open_drawer/close_drawer are the
        # only task types that reach run_joint_resets(), and the batching only exists at num_envs>1.
        argv=["tests/test_vector_integrity.py", "--cells", "8:Default,9:Default", "--num_envs", "2"],
        needs_gpu=True, needs_server=False, timeout=5400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN)\b.*)$",
        note="open_drawer + close_drawer through the vector path -- the run_joint_resets cells.",
    ),
    "test_vector_integrity_perturbations": dict(
        argv=["tests/test_vector_integrity.py", "--matrix", "perturbations", "--num_envs", "2"],
        needs_gpu=True, needs_server=False, timeout=21600, tier="slow",
        verdict=[(r"^\d+ passed, \d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN)\b.*)$",
        note="16 perturbations on task 0 through the vector path, num_envs=2.",
    ),
    "test_pi0_integration": dict(
        argv=["tests/test_pi0_integration.py"],
        needs_gpu=True, needs_server=True, timeout=7200, tier="server",
        verdict=[(r"^SKIP: preconditions not met", "SKIP"),
                 (r"^FAIL: ", "FAIL"),
                 (r"^Failed to run 01_pi0_eval\.py", "FAIL"),
                 (r"^PASS: Pi0-FAST integration test successful!", "PASS")],
        cells=r"^(?:Task progression: |  - ).*",
        note="needs a live openpi policy server on :8000. 500 steps.",
    ),
}

# tests/debug_eval.py and tests/debug_ee_control.py are NOT in the suite: neither prints a verdict,
# neither asserts anything, and debug_ee_control.py is a hand-driven scratch script. They are
# covered in the report as debug scripts, which is what they are.


def run_one(name, spec, args, outdir):
    log_path = outdir / f"{name}.log"
    if spec.get("local"):
        # No container, no allocation: a pure-Python test that reads source and config. Sending it
        # through rr+srun would cost ~40 s of apptainer start to run something that takes 0.06 s.
        cmd = [sys.executable, "-u"] + [str(PROJECT_ROOT / spec["argv"][0])] + spec["argv"][1:]
    else:
        inner = "cd %s && exec ./scripts/clara/interactive/rr python -u %s" % (
            PROJECT_ROOT, " ".join(spec["argv"]))
        if args.jobid:
            cmd = ["srun", "--jobid", str(args.jobid), "--overlap", "bash", "-c", inner]
        else:
            cmd = ["bash", "-c", inner]

    env = dict(os.environ)
    if args.mode:
        env["MODE"] = args.mode

    timeout = args.timeout or spec["timeout"]
    started = time.time()
    timed_out = False
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
                                env=env)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGTERM first, and give srun time to forward it to the step's tasks. SIGKILL straight
            # to the local srun can leave the Isaac process running on the compute node, holding the
            # GPU against the next test in the sweep. Escalate only if it does not go.
            proc.terminate()
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    elapsed = time.time() - started

    text = log_path.read_text(errors="replace")
    status, matched = "NO_VERDICT", None
    for pattern, verdict in spec["verdict"]:
        m = re.search(pattern, text, re.M)
        if m:
            status, matched = verdict, m.group(0)[:200]
            break
    if timed_out:
        status = "TIMEOUT" if status == "NO_VERDICT" else status + "_AFTER_TIMEOUT"

    cells = re.findall(spec["cells"], text, re.M) if spec.get("cells") else []
    return {
        "name": name,
        "status": status,
        "verdict_line": matched,
        "exit_code": rc,          # recorded, never gated on -- Isaac segfaults at teardown
        "timed_out": timed_out,
        "seconds": round(elapsed, 1),
        "log": str(log_path),
        "cells": cells,
        "argv": spec["argv"],
        "mode": env.get("MODE", "stock"),
        "note": spec["note"],
    }


def print_table(results, out, blob):
    """The pass/fail table. Exit codes are shown because they are recorded, not because they gate."""
    if blob:
        print(f"generated {blob.get('generated')}  jobid={blob.get('jobid')}  "
              f"mode={blob.get('mode')}")
    print("=" * 104)
    print(f"{'test':<40}{'status':<22}{'seconds':>9}  {'exit':>5}  {'cells':>6}  timed_out")
    print("-" * 104)
    for r in results:
        print(f"{r['name']:<40}{r['status']:<22}{r['seconds']:>9}  {r['exit_code']:>5}  "
              f"{len(r.get('cells', [])):>6}  {r['timed_out']}")
    print("=" * 104)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"results: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobid", default=None, help="Slurm job to srun --overlap into. Omit to run "
                                                 "on the current node.")
    p.add_argument("--mode", default=None, help="MODE for rr (stock/stockfix/oglite)")
    p.add_argument("--out", default="suite_results.json")
    p.add_argument("--logdir", default=None, help="where per-test logs go (default: next to --out)")
    p.add_argument("--only", default=None,
                   help="comma-separated test names, or a tier: fast/medium/slow/server")
    p.add_argument("--list", action="store_true")
    p.add_argument("--report", action="store_true",
                   help="print the table from an existing --out JSON and exit, running nothing. "
                        "The JSON is the record; this only formats it.")
    p.add_argument("--timeout", type=int, default=None,
                   help="override every test's per-run timeout, in seconds. A killed test is "
                        "recorded as TIMEOUT (or <verdict>_AFTER_TIMEOUT if it had already "
                        "printed one), never silently dropped.")
    args = p.parse_args()

    if args.list:
        for name, spec in SUITE.items():
            print(f"{name:<38}{spec['tier']:<8}gpu={int(spec['needs_gpu'])} "
                  f"server={int(spec['needs_server'])}  {spec['note']}")
        return 0

    if args.report:
        out = Path(args.out).absolute()
        if not out.exists():
            print(f"no results at {out}", file=sys.stderr)
            return 2
        blob = json.loads(out.read_text())
        print_table(blob.get("results", []), out, blob)
        return 0

    names = list(SUITE)
    if args.only:
        tiers = {s["tier"] for s in SUITE.values()}
        wanted = args.only.split(",")
        if set(wanted) <= tiers:
            names = [n for n in names if SUITE[n]["tier"] in wanted]
        else:
            names = [n for n in wanted if n in SUITE]
            missing = [n for n in wanted if n not in SUITE]
            if missing:
                print(f"unknown test(s): {missing}", file=sys.stderr)
                return 2

    out = Path(args.out).absolute()
    outdir = Path(args.logdir).absolute() if args.logdir else out.parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    if out.exists():
        try:
            results = json.loads(out.read_text()).get("results", [])
        except Exception:
            results = []
    results = [r for r in results if r["name"] not in names]

    def flush():
        out.write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "jobid": args.jobid,
             "mode": args.mode or "stock", "results": results}, indent=2))

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        rec = run_one(name, SUITE[name], args, outdir)
        results.append(rec)
        flush()                       # JSON first, then print -- a killed driver still has a record
        print(f"  -> {rec['status']}  {rec['seconds']}s  exit={rec['exit_code']}"
              f"{' TIMED OUT' if rec['timed_out'] else ''}", flush=True)

    print_table(results, out, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
