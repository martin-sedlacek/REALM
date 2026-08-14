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
import os
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
# the vector path. Both are blocked by the same `drawer` main object
# (custom_assets/impact_drawer/usd/cabinet.usd) and fail single-env too. Reported as KNOWN rather
# than skipped, because silently omitting them would hide the day they start working -- or the day
# something else joins them.
#
# The blocker MOVED on 2026-08-14 and this string moved with it. It used to be
# "TypeError: missing a required argument: 'preset_name'" out of MaterialPrim.get_material; that is
# fixed upstream in OG-lite (OmniSurfaceMaterialPrim now defaults preset_name=None). Loading now gets
# past the material entirely and imports scene 0, then dies on the object's own authored transform in
# XFormPrim._set_xform_properties, which rewrites the xformOp order to [translate, orient, scale],
# writes back the pose it just read, re-reads it and asserts the two agree. For cabinet.usd they
# do not:
#
#   AssertionError: /World/scene_0/drawer:
#     old_pos tensor([0.0243, 0.8223, -0.0317])  new_pos tensor([2.3468e-04, 8.1015e-01, -6.3757e-02])
#     old_orn tensor([0.0007, 0.7068, 0.7073, 0.0102])  new_orn tensor([0.7071, -0.0109, -0.0109, 0.7070])
#
# The two quaternions carry the same component magnitudes in a different order, which points at the
# asset's authored orientation rather than at the `orientation:` in the task YAML (applied later).
# Not investigated further.
_DRAWER_XFORM = ("cabinet.usd: XFormPrim._set_xform_properties assert on the authored transform "
                 "(old_orn 0.0007,0.7068,0.7073,0.0102 vs new_orn 0.7071,-0.0109,-0.0109,0.7070)")
KNOWN_BROKEN_TASKS = {8: _DRAWER_XFORM, 9: _DRAWER_XFORM}

# Signatures that mean the run died even though it exited 0.
CRASH_MARKERS = re.compile(
    r"Traceback \(most recent call last\)|AssertionError|AttributeError|KeyError|TypeError|"
    r"IndexError|RuntimeError|Segmentation fault|CUDA error|out of memory", re.I)
# Isaac segfaults during teardown on EVERY run, passing or failing, after all work is done. Matching
# it as a crash would fail every cell.
TEARDOWN_NOISE = re.compile(r"Fatal Python error: Segmentation fault|"
                            r"srun: error:.*Segmentation fault|core dumped")


def cell_id(task_id, pert):
    return f"{task_id}:{pert}"


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

    text = log_path.read_text(errors="replace")
    # Drop teardown noise before looking for crashes, or every cell "fails".
    crash_lines = [ln for ln in text.splitlines()
                   if CRASH_MARKERS.search(ln) and not TEARDOWN_NOISE.search(ln)]

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

    if crash_lines:
        return "CRASH", f"exit={proc.returncode} :: {crash_lines[0].strip()[:150]}"
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

    bad = {c: r for c, r in results.items() if r[0] not in ("PASS", "KNOWN_BROKEN")}
    n_known = sum(1 for r in results.values() if r[0] == "KNOWN_BROKEN")
    print(f"{len(results) - len(bad) - n_known} passed, {n_known} known-broken, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
