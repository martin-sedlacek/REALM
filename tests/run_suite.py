"""Run standalone REALM tests and record verdicts independent of Isaac exit codes."""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Failure verdicts must precede success verdicts because the first matching pattern wins.
SUITE = {
    "test_task_progression_rubrics": dict(
        argv=["tests/test_task_progression_rubrics.py"],
        local=True,
        needs_gpu=False, needs_server=False, timeout=120, tier="local",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^\[\d\] .*",
        note="rubric stages vs success_conditions; static, no container.",
    ),
    "test_task_type_literals": dict(
        argv=["tests/test_task_type_literals.py"],
        local=True,
        needs_gpu=False, needs_server=False, timeout=120, tier="local",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^\[\d\] .*",
        note="task_type literals in realm/ vs what the task configs declare; static, no container.",
    ),
    "test_rollout_camera_selection": dict(
        argv=["tests/test_rollout_camera_selection.py"],
        needs_gpu=False, needs_server=False, timeout=900, tier="fast",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^(?:\[\d\] .*|    task_type=.*)$",
        note="which exterior camera the drawer tasks send the policy, and the None guard.",
    ),
    "test_scene_object_placement": dict(
        argv=["tests/test_scene_object_placement.py", "--num_envs", "2"],
        needs_gpu=True, needs_server=False, timeout=1800, tier="medium",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^(?:member \d+.*|member \d+ vs member \d+.*)$",
        note="cross-member object placement + unitsResolve; MODE-sensitive by design.",
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
                 (r"^CRASHED during evaluation for ", "FAIL"),
                 (r"^Task \d+ \(.*\) PASSED!", "PASS")],
        cells=r"^  \w+_(?:csv|parquet): \w+",
        note="one task, 1 step, 1 repeat, --no_render.",
    ),
    "test_single_task_drawer": dict(
        # This drawer task exercises the OmniSurfaceMaterialPrim patch in OG-lite.
        argv=["tests/test_single_task.py", "--task_id", "8"],
        needs_gpu=True, needs_server=False, timeout=1800, tier="medium",
        verdict=[(r"^Task \d+ \(.*\) FAILED!", "FAIL"),
                 (r"^CRASHED during evaluation for ", "FAIL"),
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
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2",
              # Parquet output appends, so each suite entry needs a distinct experiment tree.
              "--experiment_name", "suite_vector_tasks"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE)\b.*)$",
        note="10 tasks under Default through the vector path, num_envs=2, rendering ON.",
    ),
    "test_vector_integrity_tasks_shard0of2": dict(
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2",
              "--shard", "0/2",
              "--experiment_name", "suite_vector_tasks_shard0of2"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE)\b.*)$",
        note="tasks 0,2,4,6,8 under Default through the vector path, num_envs=2, rendering ON.",
    ),
    "test_vector_integrity_drawers": dict(
        argv=["tests/test_vector_integrity.py", "--cells", "8:Default,9:Default", "--num_envs", "2",
              "--experiment_name", "suite_vector_drawers"],
        needs_gpu=True, needs_server=False, timeout=5400, tier="slow",
        verdict=[(r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE)\b.*)$",
        note="open_drawer + close_drawer through the vector path -- the run_joint_resets cells.",
    ),
    "test_vector_integrity_perturbations": dict(
        argv=["tests/test_vector_integrity.py", "--matrix", "perturbations", "--num_envs", "2",
              "--experiment_name", "suite_vector_perturbations"],
        needs_gpu=True, needs_server=False, timeout=21600, tier="slow",
        verdict=[(r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, [1-9]\d* failed", "FAIL"),
                 (r"^\d+ passed, \d+ refused \(intentional NotImplementedError\), "
                  r"\d+ known-broken, 0 failed", "PASS")],
        cells=r"^(?:--- \d+:\S+ .*|  -> (?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE):.*|"
              r"\d+:\S+\s+(?:PASS|CRASH|PARTIAL|NO_ARTIFACTS|KNOWN_BROKEN|"
              r"NOT_IMPL|UNDECLARED_NOT_IMPL|REFUSAL_GONE)\b.*)$",
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

LEVELS = {
    "smoke": ["test_task_progression_rubrics",   #    0.1 s
              "test_joint_reset_batching",       #   53.6 s
              "test_single_task",                #  223.3 s
              "test_scene_object_placement"],    #  428.5 s   -> 705.5 s (~12 min) total
    "suite": ["test_task_progression_rubrics",   #     0.1 s
              "test_joint_reset_batching",       #    83.4 s
              "test_single_task",                #   223.3 s
              "test_scene_object_placement",     #   329.2 s
              "test_single_task_drawer",         #   301.3 s
              "test_vector_integrity_drawers",   #   635.7 s
              "test_integrity",                  #  1834.8 s
              "test_perturbations_integrity"],   #  2584.3 s   -> ~1.7 h total
    "matrix": ["test_vector_integrity_tasks",
               "test_vector_integrity_perturbations"],
}

# These scene checks depend on the OG-lite up-axis fix.
OGLITE_SENSITIVE = ["test_single_task_drawer", "test_vector_integrity_drawers",
                    "test_scene_object_placement"]


def run_one(name, spec, args, outdir):
    log_path = outdir / f"{name}.log"
    if spec.get("local"):
        # No container, no allocation: a pure-Python test that reads source and config. Sending it
        # through Slurm and Apptainer would cost far more than this test itself.
        cmd = [sys.executable, "-u"] + [str(PROJECT_ROOT / spec["argv"][0])] + spec["argv"][1:]
    else:
        container_cmd = ["./scripts/run_apptainer.sh", "python", "-u", *spec["argv"]]
        inner = f"cd {shlex.quote(str(PROJECT_ROOT))} && exec {shlex.join(container_cmd)}"
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


def print_table(results, out, blob, ran=None):
    """Print current results separately from rows merged from earlier runs."""
    if blob:
        print(f"generated {blob.get('generated')}  jobid={blob.get('jobid')}")
    # MODE is per RESULT, never a header field: this file accumulates across invocations and the
    # runs that matter here differ precisely in which OmniGibson bind they used. A single header
    # mode would be the last invocation's and would mislabel every earlier row.
    print("=" * 116)
    print(f"{'test':<40}{'status':<22}{'mode':<8}{'seconds':>9}  {'exit':>5}  {'cells':>6}  "
          f"timed_out")
    print("-" * 116)
    for r in results:
        stale = ran is not None and r["name"] not in ran
        print(f"{('* ' if stale else '  ') + r['name']:<40}{r['status']:<22}"
              f"{r.get('mode', '?'):<8}{r['seconds']:>9}  "
              f"{r['exit_code']:>5}  {len(r.get('cells', [])):>6}  {r['timed_out']}")
    print("=" * 116)

    def _counts(rows):
        c = {}
        for r in rows:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return "  ".join(f"{k}={v}" for k, v in sorted(c.items())) or "none"

    if ran is None:
        print(_counts(results))
    else:
        fresh = [r for r in results if r["name"] in ran]
        merged = [r for r in results if r["name"] not in ran]
        print(f"this run: {_counts(fresh)}")
        if merged:
            # Spelled out rather than just marked: the whole point is that a reader skimming for
            # "PASS=n" does not silently credit this run with an older tier's result.
            print(f"* merged from earlier runs against this --out, NOT run now: {_counts(merged)}")
    print(f"results: {out}")


#: statuses that are not a failure of the code under test. SKIP is a test declining to run because
#: a precondition it cannot supply is absent (test_pi0_integration with no policy server), which is
#: information, not a fault. Everything else -- FAIL, TIMEOUT, NO_VERDICT, *_AFTER_TIMEOUT -- is.
#: Defined ABOVE its users (write_junit, verdict_status); it used to sit between them, which worked
#: only because both are functions and the module body had finished executing by call time.
OK_STATUSES = frozenset({"PASS", "SKIP"})


def write_junit(results, path, suite_name="realm"):
    """Write the final CI gate; absence signals that the driver died early."""
    import xml.etree.ElementTree as ET

    failures = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results
                 if r["status"] not in OK_STATUSES and r["status"] != "FAIL")

    ts = ET.Element("testsuite", name=suite_name, tests=str(len(results)),
                    failures=str(failures), errors=str(errors), skipped=str(skipped),
                    time=f"{sum(r['seconds'] for r in results):.1f}")
    for r in results:
        tc = ET.SubElement(ts, "testcase", classname=f"{suite_name}.{r.get('mode', 'stock')}",
                           name=r["name"], time=f"{r['seconds']:.1f}")
        detail = (f"status={r['status']} mode={r.get('mode')} exit={r['exit_code']} "
                  f"timed_out={r['timed_out']}\nverdict_line={r.get('verdict_line')}\n"
                  f"argv={' '.join(r['argv'])}\nlog={r['log']}\nnote={r['note']}")
        if r["status"] == "SKIP":
            ET.SubElement(tc, "skipped", message=str(r.get("verdict_line") or "skipped"))
        elif r["status"] == "FAIL":
            ET.SubElement(tc, "failure",
                          message=str(r.get("verdict_line") or "FAIL")).text = detail
        elif r["status"] not in OK_STATUSES:
            ET.SubElement(tc, "error", message=r["status"]).text = detail
        # cells are the per-item lines a sweep printed -- keep them, they are the detail a
        # reviewer wants when one cell of sixteen went wrong.
        if r.get("cells"):
            ET.SubElement(tc, "system-out").text = "\n".join(r["cells"])

    root = ET.ElementTree(ET.Element("testsuites"))
    root.getroot().append(ts)
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    root.write(path, encoding="utf-8", xml_declaration=True)
    print(f"junit: {path}  tests={len(results)} failures={failures} errors={errors} "
          f"skipped={skipped}")


def verdict_status(results, strict):
    """Exit status for the DRIVER. 0 unless --strict and something did not end PASS/SKIP."""
    if not strict:
        return 0
    bad = [r for r in results if r["status"] not in OK_STATUSES]
    if not bad:
        return 0
    print(f"\n--strict: {len(bad)} test(s) did not pass: "
          + ", ".join(f"{r['name']}={r['status']}" for r in bad), file=sys.stderr)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobid", default=None, help="Slurm job to srun --overlap into. Omit to run "
                                                 "on the current node.")
    p.add_argument("--mode", default=None, help="MODE for rr (stock/oglite)")
    p.add_argument("--out", default="suite_results.json")
    p.add_argument("--junit-xml", default=None, metavar="PATH",
                   help="also write a JUnit XML report here, ONCE, after the last test. This is "
                        "the CI gate: its ABSENCE means the driver itself died, which an exit "
                        "code cannot tell you. The JSON (--out) remains the incremental record.")
    p.add_argument("--logdir", default=None, help="where per-test logs go (default: next to --out)")
    p.add_argument("--only", default=None,
                   help="comma-separated test names, or a tier: local/fast/medium/slow/server. "
                        "'local' is the container-free tier -- no GPU, no allocation, no image.")
    p.add_argument("--strict", action="store_true",
                   help="make THIS DRIVER's exit status meaningful: 1 if any test did not end "
                        "PASS or SKIP, else 0. Off by default, because the normal use of this "
                        "script is to record what happened, and a suite with a known-failing "
                        "member is still a useful record. This flag is what CI "
                        "gate on. It says nothing about any CHILD's exit code, which is still "
                        "recorded and still never trusted -- see the header.")
    p.add_argument("--level", default=None, choices=sorted(LEVELS),
                   help="which GATE to run: smoke (~12 min, measured), suite (~1.7 h), matrix (hours). "
                        "A level says what you are gating on; --only's tiers say what a test "
                        "needs. Alternative to --only, not combinable with it.")
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
        return verdict_status(blob.get("results", []), args.strict)

    if args.level and args.only:
        print("--level and --only are alternatives; pass one", file=sys.stderr)
        return 2

    names = list(SUITE)
    if args.level:
        if args.level not in LEVELS:
            print(f"unknown level {args.level!r}; known: {', '.join(LEVELS)}", file=sys.stderr)
            return 2
        names = list(LEVELS[args.level])
    elif args.only:
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

    # Only the tests this invocation actually ran decide its status, and only they go in the XML.
    # `results` also carries rows merged in from earlier invocations against the same --out, and an
    # old FAIL from a tier this run did not select must neither fail this run nor appear as a
    # failure in this run's CI report.
    mine = [r for r in results if r["name"] in names]

    print_table(results, out, None, ran=set(names))
    if args.junit_xml:
        write_junit(mine, args.junit_xml)
    return verdict_status(mine, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
