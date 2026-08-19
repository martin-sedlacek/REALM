"""Integrity test for the VECTORIZED eval path: does every task and every perturbation still run?

The vector-env counterpart of test_integrity.py (which sweeps tasks) and
test_perturbations_integrity.py (which sweeps perturbations). Those drive
examples/02_evaluate.py with one env; this drives examples/04_vector_evaluate.py with N, which is
the path that actually broke: every defect fixed during the vectorization work was invisible with a
single scene, because scene 0's origin IS the world origin and there is no sibling scene to disturb.
A single-env integrity test cannot catch any of them by construction.

Two matrices rather than the full cross product, mirroring the two existing tests. 10 tasks x 16
perturbations at ~7 min a build is ~19 hours serially; tasks-under-Default plus
perturbations-under-one-task is 26 cells and finishes in well under an hour spread over a few
allocations.

What counts as a pass, per cell:
  * the process produced the four artifacts (reports csv, qpos/actions/videos parquets)
  * the report has exactly `repeats` rows -- realm_logging saves after EVERY wave, so a run that
    dies half way leaves a complete-LOOKING prefix, and row count is what distinguishes them
  * the log carries no crash signature. Isaac's SimulationApp.close() hard-exits 0 after an
    unhandled exception, so the exit code proves nothing; three "successful" runs that produced
    nothing were shipped that way before check_run.py was wired into the eval wrapper.

Rendering is left ON (unlike test_integrity.py's --no_render) so videos/*.parquet has real frames.
--extract-videos then writes them out as mp4 for the visual cross-check, which is the only way to
catch a rollout that is numerically clean and visually wrong -- e.g. the frame-bug era, where
members scored plausibly while their object sat in another member's tile.

Usage (inside the container, e.g. via scripts/clara/interactive/rr):

    python tests/test_vector_integrity.py --matrix tasks         --num_envs 2
    python tests/test_vector_integrity.py --matrix perturbations --num_envs 2 --extract-videos
    python tests/test_vector_integrity.py --cells 4:VSB-NOBJ,0:SB-VRB --num_envs 2

Each cell is one process, so cells are independent; run several allocations in parallel with
--shard i/n rather than trying to share one simulator, since destructive perturbations (SB-VRB
rewrites task_type, VSB-NOBJ replaces the main object, V-SC replaces distractors) contaminate any
build they share.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

def _const_list(name):
    """Read a top-level list literal out of realm/eval.py WITHOUT importing it.

    `from realm.eval import SUPPORTED_TASKS` looks harmless but realm/eval.py imports omnigibson at
    module scope, which boots a full Isaac instance -- in the DRIVER, purely to read two lists of
    strings, while every cell then boots another one in its child process. Two Isaac instances in
    one process tree is a needless risk on top of the wasted minute per invocation. Parsing the
    literal keeps this driver dependency-free; it is a plain list of string constants, and ast.parse
    fails loudly if that ever stops being true.
    """
    import ast
    tree = ast.parse((PROJECT_ROOT / "realm" / "eval.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError(f"{name} not found as a top-level list in realm/eval.py")


SUPPORTED_TASKS = _const_list("SUPPORTED_TASKS")
SUPPORTED_PERTURBATIONS = _const_list("SUPPORTED_PERTURBATIONS")

# Tasks that cannot currently build at all, so a failure here is expected and is NOT evidence about
# the vector path. Reported as KNOWN rather than skipped, because silently omitting them would hide
# the day they start working -- or the day something else joins them.
#
# EMPTY since 2026-08-14. Tasks 8 and 9 (open_drawer / close_drawer, the only two whose main object
# is custom_assets/impact_drawer/usd/cabinet.usd) were the last entries and both now PASS. Three
# defects in series kept them out, and none of them was in the asset -- cabinet.usd is legitimate
# USD that happens not to satisfy an unwritten OmniGibson convention:
#
#   1. OmniSurfaceMaterialPrim.__init__ requiring preset_name          -- OG-lite 1dcc5bb
#   2. XFormPrim._set_xform_properties reading the pose through the VIRTUAL
#      self.get_position_orientation() -- which EntityPrim overrides to return the ROOT LINK's pose
#      -- while writing it back through the pinned XFormPrim.set_position_orientation, which
#      authors the ENTITY prim's xformOps. cabinet.usd's base_link sits at Ry(180) + 4 cm from the
#      entity prim, so the write moved the object by exactly that and the method's own round-trip
#      assert caught it. Every BEHAVIOR asset has base_link at the entity origin, which is why no
#      other object ever tripped it.
#   3. EntityPrim.set_position_orientation's stopped-sim branch asserting outright that the root
#      link has no relative pose to the entity prim -- the same 4 cm offset.
#
# 2 and 3 are fixed in OG-lite; both fixes are no-ops for a root link at the entity origin, so no
# other asset's placement changes. If a drawer task regresses, suspect the OG-lite bind first.
KNOWN_BROKEN_TASKS = {}

# Signatures that mean the run died even though it exited 0.
CRASH_MARKERS = re.compile(
    r"Traceback \(most recent call last\)|AssertionError|AttributeError|KeyError|TypeError|"
    r"IndexError|RuntimeError|Segmentation fault|CUDA error|out of memory", re.I)
# Isaac CAN segfault during teardown after all work is done, and matching that as a crash would fail
# a cell that actually succeeded -- hence this filter.
#
# But it does NOT happen on every run, which this comment used to claim. Measured over the 2026-08-18
# matrix re-run on examples/04_vector_evaluate.py: the only cell whose log carried a segfault or a
# traceback was 8:VB-MOBJ, which raises an intentional NotImplementedError, and a control build where
# nothing raises passed with zero segfaults. The "every run" claim is true of
# t9_vbpose_nostopplay.py, where it was first written, and was over-generalised to here.
#
# Consequence worth keeping: on this path a segfault usually means an uncaught Python exception
# propagated out, so it correlates with a real failure rather than being pure noise. Do not widen
# this filter -- the CRASH verdict comes from CRASH_MARKERS matching "Traceback", not from the exit
# code, so a cell reports CRASH even when the process exits 0, and the -11 itself carries no
# information.
TEARDOWN_NOISE = re.compile(r"Fatal Python error: Segmentation fault|"
                            r"srun: error:.*Segmentation fault|core dumped")

# Cells where a NotImplementedError is the DESIGNED answer, not a defect. Without this table those
# four cells report CRASH, identically to a cell that broke -- which is how the header comment above
# came to note in passing that "8:VB-MOBJ raises an intentional NotImplementedError" while the table
# still counted it as one of the matrix's failures. A refusal and a breakage are different results
# and the summary has to be able to say which it saw.
#
# Every entry must name the raise site, so a reader can check the claim instead of trusting this dict.
EXPECTED_NOT_IMPLEMENTED = {
    "8:VB-MOBJ": "vb_mobj.py:71 -- the resize branch only handles DatasetObject; both drawer tasks' "
                 "main object is a USDObject (custom_assets/impact_drawer/usd/cabinet.usd)",
    "9:VB-MOBJ": "vb_mobj.py:71 -- same USDObject branch",
    "8:SB-VRB": "sb_vrb.py:99 -- deliberate refusal, the drawer configs' empty target_objects sends "
                "SB-VRB down the add-a-receiver branch and it rains an unplaceable object",
    "9:SB-VRB": "sb_vrb.py:99 -- same refusal",
}
# A traceback whose LAST exception line is this and which carries no other error type. Both spellings
# matter: sb_vrb raises with a message ("NotImplementedError: SB-VRB does not support..."), vb_mobj
# raises bare ("NotImplementedError" with no colon at all).
NOT_IMPL_LINE = re.compile(r"^(?:\w+\.)*NotImplementedError(?::|\s*$)")
TRACEBACK_LINE = re.compile(r"Traceback \(most recent call last\)")


def cell_id(task_id, pert):
    return f"{task_id}:{pert}"


def classify_log(text, cid, returncode):
    """Crash verdict for one cell's log. Returns (status, detail), or None if the log looks clean.

    Split out of run_cell so it can be exercised on the login node against synthetic logs --
    tests/test_cell_classification.py. run_cell itself needs a GPU, an Isaac boot and ~7 minutes per
    cell, so the branch that decides refusal-vs-breakage would otherwise only ever be validated by
    the very runs whose verdicts depend on it.
    """
    # Drop teardown noise before looking for crashes, or every cell "fails".
    crash_lines = [ln for ln in text.splitlines()
                   if CRASH_MARKERS.search(ln) and not TEARDOWN_NOISE.search(ln)]
    if not crash_lines:
        return None

    # Separate an intentional refusal from a breakage before calling anything CRASH. Deliberately
    # narrow: the ONLY crash signatures allowed to be present are the "Traceback" header lines
    # themselves, so a NotImplementedError that arrives alongside a KeyError or an AssertionError
    # still reports CRASH -- otherwise a real failure in a declared-refusal cell would be laundered
    # into an expected result, which is worse than the conflation this fixes.
    other = [ln for ln in crash_lines if not TRACEBACK_LINE.search(ln)]
    not_impl = [ln for ln in text.splitlines() if NOT_IMPL_LINE.match(ln)]
    if not_impl and not other:
        msg = not_impl[-1].strip()[:150]
        if cid in EXPECTED_NOT_IMPLEMENTED:
            return "NOT_IMPL", msg
        # A refusal nobody declared. Reported as a failure on purpose: either the perturbation grew a
        # branch it should not refuse, or EXPECTED_NOT_IMPLEMENTED needs the entry and the reason
        # written down.
        return "UNDECLARED_NOT_IMPL", msg
    return "CRASH", f"exit={returncode} :: {crash_lines[0].strip()[:150]}"


def run_cell(task_id, pert, args, log_root):
    """Run one (task, perturbation) cell and classify it. Returns (status, detail)."""
    pert_id = SUPPORTED_PERTURBATIONS.index(pert)
    task = SUPPORTED_TASKS[task_id]
    run_id = f"t{task_id}_{pert.replace('-', '')}"
    log_path = Path(log_root) / f"{run_id}.log"

    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "examples/04_vector_evaluate.py"),
        "--num_envs", str(args.num_envs),
        "--task_id", str(task_id),
        "--perturbation_id", str(pert_id),
        "--repeats", str(args.repeats),
        "--max_steps", str(args.max_steps),
        "--model_type", "debug", "--model_name", "debug", "--port", "8000",
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--log_dir", str(args.log_dir),
        "--robot", args.robot,
        "--rendering_mode", "rt",
    ]

    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT), text=True)

    crash_verdict = classify_log(log_path.read_text(errors="replace"),
                                 cell_id(task_id, pert), proc.returncode)

    results_dir = Path(args.log_dir) / args.experiment_name / "debug" / run_id
    report = results_dir / "reports" / f"{task}_{pert}.csv"
    artifacts = {
        "report": report,
        "qpos": results_dir / "qpos" / f"{task}.parquet",
        "actions": results_dir / "actions" / f"{task}.parquet",
        "videos": results_dir / "videos" / f"{task}.parquet",
    }
    missing = [k for k, p in artifacts.items() if not p.exists() or p.stat().st_size == 0]

    n_rows = None
    if report.exists():
        with open(report) as fh:
            n_rows = sum(1 for _ in csv.DictReader(fh))

    if crash_verdict:
        return crash_verdict
    if missing:
        return "NO_ARTIFACTS", f"exit={proc.returncode} missing={','.join(missing)}"
    if n_rows != args.repeats:
        # A partial run is not a pass. This is the case the exit code cannot see.
        return "PARTIAL", f"report has {n_rows} rows, expected {args.repeats}"
    return "PASS", f"{n_rows} rollouts, all artifacts present"


def extract_videos(args, run_ids):
    """Write the mp4s out of each cell's videos/*.parquet for visual cross-check."""
    tool = PROJECT_ROOT / "scripts/videos_parquet_to_mp4.py"
    for run_id in run_ids:
        d = Path(args.log_dir) / args.experiment_name / "debug" / run_id
        if not (d / "videos").is_dir():
            continue
        subprocess.run([sys.executable, str(tool), str(d)], cwd=str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", choices=["tasks", "perturbations", "both"], default="tasks",
                   help="tasks: every task under Default. perturbations: every perturbation on "
                        "--pert-task.")
    p.add_argument("--cells", default=None,
                   help="explicit comma-separated <task_id>:<PERT> cells, overrides --matrix")
    p.add_argument("--pert-task", type=int, default=0,
                   help="task used for the perturbation matrix. NOTE task 0's main object is a "
                        "PrimitiveObject, so VB-MOBJ takes its rescale branch there; use 4 "
                        "(pick_spoon, a DatasetObject) to exercise its remove/add branch instead.")
    p.add_argument("--num_envs", type=int, default=2,
                   help="2 is the cheapest count that can see a cross-scene bug: scene 0 alone "
                        "cannot, since its origin is the world origin.")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--robot", type=str, default="DROID_robolab_v2")
    p.add_argument("--experiment_name", type=str, default="vector_integrity")
    p.add_argument("--log_dir", type=str, default="/logs")
    p.add_argument("--shard", type=str, default=None,
                   help="'i/n' -- run only every n-th cell, for fanning across allocations")
    p.add_argument("--extract-videos", action="store_true",
                   help="write each cell's mp4s out of the video parquet for visual inspection")
    args = p.parse_args()

    if args.cells:
        cells = []
        for spec in args.cells.split(","):
            t, pert = spec.split(":")
            cells.append((int(t), pert))
    else:
        cells = []
        if args.matrix in ("tasks", "both"):
            cells += [(t, "Default") for t in range(len(SUPPORTED_TASKS))]
        if args.matrix in ("perturbations", "both"):
            cells += [(args.pert_task, pert) for pert in SUPPORTED_PERTURBATIONS]

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]

    log_root = Path(args.log_dir) / args.experiment_name / "_runlogs"
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"=== vector integrity: {len(cells)} cell(s), num_envs={args.num_envs}, "
          f"repeats={args.repeats}, max_steps={args.max_steps}, robot={args.robot} ===", flush=True)

    results, run_ids = {}, []
    for task_id, pert in cells:
        cid = cell_id(task_id, pert)
        print(f"\n--- {cid} ({SUPPORTED_TASKS[task_id]}) ---", flush=True)
        status, detail = run_cell(task_id, pert, args, log_root)
        if status != "PASS" and task_id in KNOWN_BROKEN_TASKS:
            status, detail = "KNOWN_BROKEN", KNOWN_BROKEN_TASKS[task_id]
        elif status == "PASS" and cid in EXPECTED_NOT_IMPLEMENTED:
            # The declaration went stale: this cell was supposed to refuse and it ran. Good news, but
            # it must not pass silently, or EXPECTED_NOT_IMPLEMENTED rots into a list of cells nobody
            # measures -- the same way KNOWN_BROKEN_TASKS would have, had it not been emptied.
            status, detail = "REFUSAL_GONE", (
                f"ran and produced artifacts, but is declared a refusal: {EXPECTED_NOT_IMPLEMENTED[cid]}"
                " -- if that is now implemented, delete the entry")
        results[cid] = (status, detail)
        run_ids.append(f"t{task_id}_{pert.replace('-', '')}")
        print(f"  -> {status}: {detail}", flush=True)

    if args.extract_videos:
        extract_videos(args, run_ids)

    print("\n" + "=" * 78)
    print(f"{'cell':<22}{'status':<15}detail")
    print("-" * 78)
    for cid, (status, detail) in results.items():
        print(f"{cid:<22}{status:<15}{detail}")
    print("=" * 78)

    # NOT_IMPL is an expected result, so it is not a failure. REFUSAL_GONE and UNDECLARED_NOT_IMPL
    # both ARE, because both mean EXPECTED_NOT_IMPLEMENTED and the code disagree about which cells
    # refuse -- and that disagreement is precisely what this classification exists to surface.
    ok = ("PASS", "KNOWN_BROKEN", "NOT_IMPL")
    bad = {c: r for c, r in results.items() if r[0] not in ok}
    n_known = sum(1 for r in results.values() if r[0] == "KNOWN_BROKEN")
    n_refused = sum(1 for r in results.values() if r[0] == "NOT_IMPL")
    print(f"{len(results) - len(bad) - n_known - n_refused} passed, {n_refused} refused "
          f"(intentional NotImplementedError), {n_known} known-broken, {len(bad)} failed")
    if bad:
        print("failed: " + ", ".join(f"{c} [{r[0]}]" for c, r in bad.items()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
