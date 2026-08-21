"""Run REALM's test suite end to end and record what actually happened.

WHY THIS EXISTS, AND WHY IT IS NOT PYTEST
-----------------------------------------
Most files in tests/ are standalone scripts -- a `if __name__ == "__main__":` block, printed
verdict lines, `sys.exit(1)` on failure -- and that is how this driver runs them:
`python -u tests/<file>.py`, one process each. `pytest tests/` must NOT be used as the suite:
collection imports every module, and three boot a full Isaac instance at module scope
(test_joint_reset_batching and test_scene_object_placement import omnigibson directly;
test_rollout_camera_selection reaches it through realm.rollout). The eval DRIVERS
(test_integrity, test_single_task, test_perturbations_integrity, test_vector_integrity) used to
as well, through realm.eval; they now ast-parse the two lists they need (tests/_paths.py
eval_const_list) and boot Isaac only in their child processes.
(pytest IS installed in the container, at
/opt/conda/envs/behavior/lib/python3.11/site-packages; it is absent from the login python. So
"pytest is missing" is not the reason.) The exceptions: four real pytest modules, host-safe by
design (they read code/configs as text with ast/yaml rather than importing anything heavy) -- run
them directly, not through this driver:

    pytest tests/test_perturbation_task_types.py tests/test_cell_classification.py \
           tests/test_robot_base_column.py tests/test_robot_definition_parity.py

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

TIERS. `--only` takes test names or a tier: `local`, `fast`, `medium`, `slow`, `server`. Only
`local` is container-free -- no image, no GPU, no Slurm allocation, ~0.06 s. `fast` still needs the
container: test_joint_reset_batching stubs `og.sim`, but it `import omnigibson` at module scope, so
it cannot run on the login python. Do not read `needs_gpu=False` as "runs anywhere".

    python3 tests/run_suite.py --only local --strict          # container-free; what CI runs
    python3 tests/run_suite.py --jobid 191441 --out /path/results.json --only medium
    python3 tests/run_suite.py --list
    python3 tests/run_suite.py --report --out /path/results.json

LEVELS are the other axis, and the one the Makefile exposes: `--level smoke|suite|matrix` picks a
GATE (see LEVELS below) rather than a capability class. `make check` / `make test-static` are
tier 1; `make test-smoke` / `test-suite` / `test-matrix` are tier 2. `make test` runs ONLY the
`local` tier and prints what it skipped -- it is not the suite.
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
    "test_task_type_literals": dict(
        argv=["tests/test_task_type_literals.py"],
        local=True,   # login python, stdlib + yaml only: no container, no allocation, ~0.05 s
        needs_gpu=False, needs_server=False, timeout=120, tier="local",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^\[\d\] .*",
        note="task_type literals in realm/ vs what the task configs declare; static, no container.",
    ),
    "test_rollout_camera_selection": dict(
        # Imports realm.rollout, which imports omnigibson, so it needs the container -- but it
        # builds no environment and touches no GPU. Same tier as test_joint_reset_batching, for
        # the same reason.
        argv=["tests/test_rollout_camera_selection.py"],
        needs_gpu=False, needs_server=False, timeout=900, tier="fast",
        verdict=[(r"^FAILED -- \d+ problem", "FAIL"), (r"^PASSED -- ", "PASS")],
        cells=r"^(?:\[\d\] .*|    task_type=.*)$",
        note="which exterior camera the drawer tasks send the policy, and the None guard.",
    ),
    "test_scene_object_placement": dict(
        # The only test that looks at the SCENE rather than at the artifacts. It exists because
        # both drawer tests passed on a build whose scene-0 cabinet was lying on its back.
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
        # Task 8 is open_drawer, whose main object is custom_assets/impact_drawer/usd/cabinet.usd.
        # It is the ONLY task that needs OmniSurfaceMaterialPrim's preset_name default, which the
        # stock 3.9.1 image does not have and OG-lite / stock_patch do. Run this under MODE=oglite
        # against the MODE=stock result from test_integrity: same task, same code, different bind.
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
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2",
              # DISTINCT PER ENTRY. Every vector entry used to default to --experiment_name
              # "vector_integrity", so they shared ONE /logs tree with no discriminator while the
              # parquets append. Two consequences, both measured on the 2026-08-21 baseline:
              # this entry writes t8/t9:Default and so does _drawers, so _drawers reported
              # FAIL_ROWS in EVERY suite run where both ran; and rows survived across sweeps, so
              # a fresh run saw 8 where it wanted 2. tests/_paths.py::scratch_log_root documents
              # the hazard and says this test needs a distinct --experiment_name -- it now has one.
              "--experiment_name", "suite_vector_tasks"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        # Matches test_vector_integrity's summary line EXACTLY as printed since the refused count
        # was added to it (2026-08-19) -- the old `\d+ passed, \d+ known-broken` patterns matched
        # nothing once "N refused (...)" was inserted between them, so every vector entry ran its
        # cells and then reported no verdict at all.
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
        # Half the task matrix -- cells 0,2,4,6,8, i.e. tasks 0,2,4,6,8 -- for when the allocation
        # cannot hold the full ten. Deliberately a COMPLETED half rather than a truncated whole: a
        # run killed on the time limit never prints its verdict line. The sample is not arbitrary:
        # it covers a PrimitiveObject main object (0), a rotate task (2), a DatasetObject pick (4),
        # a stack (6) and open_drawer (8) -- the one that could not load at all until 2026-08-14.
        argv=["tests/test_vector_integrity.py", "--matrix", "tasks", "--num_envs", "2",
              "--shard", "0/2",
              # Distinct from suite_vector_tasks even though it is a SUBSET of the same matrix: the
              # shard writes the same cell names, so sharing the tree would make whichever ran
              # second report FAIL_ROWS against rows the other legitimately wrote.
              "--experiment_name", "suite_vector_tasks_shard0of2"],
        needs_gpu=True, needs_server=False, timeout=14400, tier="slow",
        # Matches test_vector_integrity's summary line EXACTLY as printed since the refused count
        # was added to it (2026-08-19) -- the old `\d+ passed, \d+ known-broken` patterns matched
        # nothing once "N refused (...)" was inserted between them, so every vector entry ran its
        # cells and then reported no verdict at all.
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
        # The two drawer cells only. A sibling measured 8:Default crashing under MODE=stock with
        # material_prim.py's missing preset_name default; this is that cell, re-run against the
        # rebuilt image. Vector rather than single-env because open_drawer/close_drawer are the
        # only task types that reach run_joint_resets(), and the batching only exists at num_envs>1.
        argv=["tests/test_vector_integrity.py", "--cells", "8:Default,9:Default", "--num_envs", "2",
              "--experiment_name", "suite_vector_drawers"],
        needs_gpu=True, needs_server=False, timeout=5400, tier="slow",
        # Matches test_vector_integrity's summary line EXACTLY as printed since the refused count
        # was added to it (2026-08-19) -- the old `\d+ passed, \d+ known-broken` patterns matched
        # nothing once "N refused (...)" was inserted between them, so every vector entry ran its
        # cells and then reported no verdict at all.
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
        # Matches test_vector_integrity's summary line EXACTLY as printed since the refused count
        # was added to it (2026-08-19) -- the old `\d+ passed, \d+ known-broken` patterns matched
        # nothing once "N refused (...)" was inserted between them, so every vector entry ran its
        # cells and then reported no verdict at all.
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

# Hand-driven debug scripts live in scripts/debug/, not here: they print no verdict and assert
# nothing, so they cannot be suite entries. (tests/debug_eval.py, a hardcoded duplicate of
# `02_evaluate.py --task_id 8 --model_type debug`, was deleted 2026-08-18.)


# ---------------------------------------------------------------------------------------------
# LEVELS -- the manual, GPU-side pipeline. A `tier` says what a test NEEDS; a `level` says which
# gate you are running. They are different axes on purpose: `--only slow` is "the expensive ones",
# `--level suite` is "the gate before you trust a change".
#
# Seconds are MEASURED, from logs/suite_results_v2.json and a `make test-smoke` run on job 191496
# (both 2026-08-16, MODE=stock, one L40S). They are what `make test-smoke` / `test-suite` quote, so they must not drift into
# guesses -- if you re-time the suite, update these from the JSON rather than from memory.
# ---------------------------------------------------------------------------------------------
LEVELS = {
    # The cheap gate: static checks, the scheduling test, one task end to end, and the only test
    # that looks at the SCENE, at num_envs=2. Catches "the port is broken" in about ten minutes.
    # NOTE: at MODE=stock this level is EXPECTED TO REPORT 2 FAILURES -- the rubric test (a real
    # code defect, see its docstring) and test_scene_object_placement, which is MODE-sensitive by
    # design and only passes under oglite. Measured end to end on job 191496, 2026-08-16: 705.5 s.
    "smoke": ["test_task_progression_rubrics",   #    0.1 s
              "test_joint_reset_batching",       #   53.6 s
              "test_single_task",                #  223.3 s
              "test_scene_object_placement"],    #  428.5 s   -> 705.5 s (~12 min) total
    # The gate before trusting a change: every task, every perturbation, both drawer paths.
    "suite": ["test_task_progression_rubrics",   #     0.1 s
              "test_joint_reset_batching",       #    83.4 s
              "test_single_task",                #   223.3 s
              "test_scene_object_placement",     #   329.2 s
              "test_single_task_drawer",         #   301.3 s
              "test_vector_integrity_drawers",   #   635.7 s
              "test_integrity",                  #  1834.8 s
              "test_perturbations_integrity"],   #  2584.3 s   -> ~1.7 h total
    # The full task x perturbation sweep through the vector path. NO COMPLETED RUN IS ON RECORD:
    # the half-matrix shard reached NO_VERDICT after 1315 s in job 191441. Budget hours, and
    # expect to shard it -- test_vector_integrity_tasks_shard0of2 exists for exactly that.
    "matrix": ["test_vector_integrity_tasks",
               "test_vector_integrity_perturbations"],
}

#: Cells whose SCENE correctness depends on the OmniGibson bind. The v2 image carries most of the
#: patches but NOT the up-axis fix, which lives only in the OG-lite fork; without it a drawer
#: scene's cabinet can be mis-oriented while every artifact check still passes. `make test-suite`
#: re-runs these under MODE=oglite as a second invocation, so the JSON carries both modes and the
#: per-result `mode` column says which is which.
OGLITE_SENSITIVE = ["test_single_task_drawer", "test_vector_integrity_drawers",
                    "test_scene_object_placement"]


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


def print_table(results, out, blob, ran=None):
    """The pass/fail table. Exit codes are shown because they are recorded, not because they gate.

    `ran` is the set of test names THIS invocation actually executed; anything else in `results` was
    merged in from an earlier invocation against the same --out and is marked `*`, with the counts
    split. Pass None (--report) to mark nothing, since then nothing was run.

    WHY THE MARK EXISTS. Gating was always correct -- `mine` decides the exit status and the JUnit XML
    -- but the TABLE printed merged rows indistinguishably and the `PASS=n` line counted them. So
    `make check`, which runs two container-free tests, printed a 9-row table ending `PASS=9` including
    `test_perturbations_integrity PASS 48 cells` from a stale earlier run. On 2026-08-19 that stale row
    directly contradicted a FAIL measured against the same tree twenty minutes earlier (the B-HOBJ
    dtype crash), and it read as fresh evidence that the GPU tier was green.
    In a suite whose first rule is that verdicts come from printed lines and never from exit codes, a
    printed line that silently mixes this run with an old one is the wrong failure mode.
    """
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
    """Emit a JUnit XML report, one <testcase> per suite entry.

    WHY, AND WHY IT IS WRITTEN ONLY AT THE END
    ------------------------------------------
    This is the CI GATE, and it is deliberately a different artifact from the JSON, which is the
    RECORD. The pattern is upstream BEHAVIOR-1K's (.github/workflows/tests.yml): run the tests with
    `continue-on-error`, then judge on the file --

        if [ ! -f results.xml ]; then echo "probably a segfault"; exit 1
        elif grep -Eq 'failures="[1-9][0-9]*"|errors="[1-9][0-9]*"' results.xml; then exit 1

    -- because an exit code cannot distinguish "passed" from "died before it could tell you". That
    is the same hazard REALM has everywhere: Isaac hard-exits 0 on an unhandled exception.

    run_suite.py already survives a CHILD dying: it reads the child's log and matches verdict
    lines. What it could not signal until now is THE DRIVER ITSELF dying -- an OOM, a walltime
    kill, a node failure. So the XML is written ONCE, after the last test. If the driver is killed,
    the XML is absent and CI says so, while the JSON -- rewritten after every test, as before --
    still holds the complete record of the tests that did finish. Writing the XML incrementally
    would destroy exactly the signal it exists to carry.

    Status mapping: PASS -> bare testcase; SKIP -> <skipped>; FAIL -> <failure>; TIMEOUT /
    NO_VERDICT / *_AFTER_TIMEOUT -> <error>, because those mean the harness could not establish an
    outcome, which is a different thing from the test having failed.
    """
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
    p.add_argument("--mode", default=None, help="MODE for rr (stock/stockfix/oglite)")
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
                        "member is still a useful record. This flag is what `make test` and CI "
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
