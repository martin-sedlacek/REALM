"""Exercise vectorized task and perturbation evaluation cells."""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Parse constants without booting Isaac in the driver process.
from tests._paths import check_artifacts, eval_const_list

SUPPORTED_TASKS = eval_const_list("SUPPORTED_TASKS")
SUPPORTED_PERTURBATIONS = eval_const_list("SUPPORTED_PERTURBATIONS")

KNOWN_BROKEN_TASKS = {}

from tests._paths import CRASH_MARKERS, TEARDOWN_NOISE  # noqa: E402

# Deliberate unsupported combinations, kept distinct from regressions.
EXPECTED_NOT_IMPLEMENTED = {
    "8:VB-MOBJ": "vb_mobj.py:71 -- the resize branch only handles DatasetObject; both drawer tasks' "
                 "main object is a USDObject (custom_assets/impact_drawer/usd/cabinet.usd)",
    "9:VB-MOBJ": "vb_mobj.py:71 -- same USDObject branch",
    "8:SB-VRB": "sb_vrb.py:99 -- deliberate refusal, the drawer configs' empty target_objects sends "
                "SB-VRB down the add-a-receiver branch and it rains an unplaceable object",
    "9:SB-VRB": "sb_vrb.py:99 -- same refusal",
}
# Anchor to the final exception line, not the echoed source line.
NOT_IMPL_LINE = re.compile(r"^(?:\w+\.)*NotImplementedError(?::|\s*$)")
TRACEBACK_LINE = re.compile(r"Traceback \(most recent call last\)")
# Strip Isaac's logger prefix before matching the exception.
LOG_PREFIX = re.compile(r"^\S+ \[[\d,]+ms\] \[\w+\] \[[\w.]+\] \[py std(?:err|out)\]: ")


def cell_id(task_id, pert):
    return f"{task_id}:{pert}"


def classify_log(text, cid, returncode):
    """Return a crash verdict, or ``None`` for a clean log."""
    crash_lines = [ln for ln in text.splitlines()
                   if CRASH_MARKERS.search(ln) and not TEARDOWN_NOISE.search(ln)]
    if not crash_lines:
        return None

    # A mixed traceback remains a crash even in an unsupported cell.
    other = [ln for ln in crash_lines if not TRACEBACK_LINE.search(ln)]
    not_impl = [s for s in (LOG_PREFIX.sub("", ln) for ln in text.splitlines())
                if NOT_IMPL_LINE.match(s)]
    if not_impl and not other:
        msg = not_impl[-1].strip()[:150]
        if cid in EXPECTED_NOT_IMPLEMENTED:
            return "NOT_IMPL", msg
        return "UNDECLARED_NOT_IMPL", msg
    return "CRASH", f"exit={returncode} :: {crash_lines[0].strip()[:150]}"


def artifact_verdict(results_dir, task, pert, repeats):
    """Return an artifact verdict, or ``None`` when all outputs are complete."""
    art = check_artifacts(str(results_dir), task, pert, repeats)
    missing = [k for k, v in art.items()
               if v == "FAIL_MISSING" or v == "FAIL_EMPTY" or v.startswith("FAIL_UNREADABLE")]
    if missing:
        return "NO_ARTIFACTS", "missing/empty/unreadable: " + ",".join(missing)
    wrong = {k: v for k, v in art.items() if v != "PASS"}
    if wrong:
        return "PARTIAL", ", ".join(f"{k}: {v}" for k, v in wrong.items())
    return None


def run_cell(task_id, pert, args, log_root):
    """Run one (task, perturbation) cell and classify it. Returns (status, detail)."""
    pert_id = SUPPORTED_PERTURBATIONS.index(pert)
    task = SUPPORTED_TASKS[task_id]
    run_id = f"t{task_id}_{pert.replace('-', '')}"
    log_path = Path(log_root) / f"{run_id}.log"

    # Clear the cell because parquet output is append-only.
    results_dir = Path(args.log_dir) / args.experiment_name / "debug" / run_id
    # Restrict deletion to the expected nested scratch path.
    parts = (args.experiment_name, "debug", run_id)
    if (all(parts) and not any("/" in x or x in (".", "..") for x in parts)
            and results_dir.is_dir() and len(results_dir.parts) > 4):
        shutil.rmtree(results_dir)

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
    if crash_verdict:
        return crash_verdict

    # The model_name segment is hardcoded "debug" in run_cell's cmd, so it is hardcoded here too.
    results_dir = Path(args.log_dir) / args.experiment_name / "debug" / run_id
    bad = artifact_verdict(results_dir, task, pert, args.repeats)
    if bad:
        return bad
    return "PASS", f"{args.repeats} rollouts, all artifacts present"


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
    p.add_argument("--robot", type=str, default="DROID_mounted")
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
