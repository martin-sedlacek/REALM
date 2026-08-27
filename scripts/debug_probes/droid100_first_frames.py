"""Render the initial condition of every DROID100 tabletop task: 3 views, one step, no policy.

The point is to eyeball 100 agentically-generated scenes at t=0 -- are the objects on the table,
in frame, reachable, in the relation the instruction claims -- and to have the numbers next to the
pictures so the obvious defects can be grepped rather than spotted.

ONE Isaac boot per invocation. The cold start (~2-4 min) is paid once and then every task in this
shard is built, rendered and torn down in-process via `og.sim.stop()` + `og.clear()`, which is the
supported rebuild path in this OmniGibson fork (see its own tests/test_envs.py). A task that raises
is recorded and skipped; the shard carries on.

No inference server is involved. Each task gets reset() + warmup() (30 steps of "hold the arm still,
open then close the gripper" -- enough for a dropped object to settle) and then a single held step
whose observation is saved. That is exactly the observation a policy would receive as its first
input, so what is rendered here is what the model would see.

Usage (inside the container, on a GPU):
    python -u scripts/debug_probes/droid100_first_frames.py \
        --out /logs/droid100_first_frames/<run_id> --shard 3 --num_shards 16

Writes, under --out:
    frames/<task>/cam1.jpg cam2.jpg wrist.jpg panel.jpg     the three views + a captioned strip
    shard<NN>.json                                          one record per task, rewritten per task
    shard<NN>.log                                           this script's own narration

Sharding is ROUND-ROBIN (task i -> shard i % num_shards), so a suite whose later tasks happen to be
heavier does not land them all on one job.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import extract_from_obs
from realm.sim_config import set_sim_config

CONFIG_ROOT = "/app/realm/config"

#: Views saved per task, in panel order. Index into extract_from_obs()'s return tuple.
VIEWS = (("cam1", 0), ("cam2", 2), ("wrist", 4))

#: Saved size of a single view, and of one panel cell. Both are JPEG -- these are for eyeballing and
#: a 100-task pull at 1280x720 PNG is ~1 GB of scp for no extra information.
VIEW_SIZE = (960, 540)
CELL_SIZE = (640, 360)
JPEG_QUALITY = 92


def _say(msg, log_fh):
    print(msg, flush=True)
    log_fh.write(msg + "\n")
    log_fh.flush()


def _to_uint8(im):
    im = np.asarray(im)
    if im.dtype.kind == "f":
        im = (im * 255).clip(0, 255)
    return im.astype(np.uint8)[..., :3]


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(views, caption, subcaption):
    """One captioned strip: cam1 | cam2 | wrist, with the instruction and the flags underneath."""
    cw, ch = CELL_SIZE
    bar = 62
    canvas = Image.new("RGB", (cw * len(views), ch + bar), (24, 24, 28))
    for i, (_, im) in enumerate(views):
        canvas.paste(Image.fromarray(im).resize(CELL_SIZE, Image.BILINEAR), (i * cw, 0))
    d = ImageDraw.Draw(canvas)
    for i, (name, _) in enumerate(views):
        d.text((i * cw + 8, 6), name, fill=(255, 230, 120), font=_font(20))
    d.text((8, ch + 6), caption[:190], fill=(240, 240, 240), font=_font(19))
    d.text((8, ch + 32), subcaption[:260], fill=(170, 200, 240), font=_font(16))
    return canvas


def _vec(v):
    """Any torch/numpy/list pose component -> a plain list of rounded floats."""
    if v is None:
        return None
    v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    return [round(float(x), 5) for x in np.atleast_1d(v).ravel()]


def _obj_record(obj, authored_pos):
    """Live pose/AABB of one scene object plus how far it drifted from its authored position."""
    pos, _ = obj.get_position_orientation(frame="scene")
    pos = _vec(pos)
    rec = {
        "name": obj.name,
        "category": getattr(obj, "category", None),
        "pos_scene": pos,
        "aabb_center": _vec(obj.aabb_center),
        "aabb_extent": _vec(obj.aabb_extent),
    }
    if authored_pos is not None:
        rec["authored_pos"] = [round(float(x), 5) for x in authored_pos]
        rec["drift"] = [round(a - b, 5) for a, b in zip(pos, rec["authored_pos"])]
    return rec


def _relations(env):
    """Inside/OnTop of main object w.r.t. the target, plus t=0 rubric state.

    A `put`/`stack` task whose main object is ALREADY inside/on its target at t=0 starts solved; a
    `pick`/`remove` task whose main object is NOT on/in its source has nothing to remove. Both are
    authoring defects and both are cheap to read here.
    """
    out = {}
    if not env.main_objects or not env.target_objects:
        return out
    mo, to = env.main_objects[0], env.target_objects[0]
    for label, state in (("inside", og.object_states.Inside), ("on_top", og.object_states.OnTop)):
        try:
            out[label] = bool(mo.states[state].get_value(to))
        except Exception as exc:                      # a state a category does not support
            out[label] = f"ERROR: {type(exc).__name__}: {exc}"
    return out


def _flags(rec, task_type):
    """Terse defect flags, derived from the record. These are what makes the pile of 100 greppable."""
    flags = []
    for cam, stats in rec.get("view_stats", {}).items():
        if stats["mean"] < 4.0:
            flags.append(f"BLACK:{cam}")
        elif stats["mean"] > 245.0:
            flags.append(f"BLOWN:{cam}")
    for role in ("main", "target"):
        for o in rec.get("objects", {}).get(role, []):
            drift = o.get("drift")
            if drift is None:
                continue
            if drift[2] < -0.05:
                flags.append(f"FELL:{o['name']}({drift[2]:+.2f}m)")
            if max(abs(drift[0]), abs(drift[1])) > 0.10:
                flags.append(f"SLID:{o['name']}")
    rel = rec.get("relations", {})
    on_or_in = (rel.get("inside") is True) or (rel.get("on_top") is True)
    if task_type in ("put", "stack") and on_or_in:
        flags.append("ALREADY_SOLVED")
    if task_type in ("pick", "rotate") and rec.get("objects", {}).get("target") and not on_or_in:
        flags.append("NO_INITIAL_RELATION")
    prog = rec.get("task_progression") or {}
    solved = [k for k, v in prog.items() if v]
    if solved:
        flags.append("RUBRIC_TRUE:" + ",".join(solved))
    if rec.get("reach_distance") is not None and rec["reach_distance"] > 0.95:
        flags.append(f"FAR:{rec['reach_distance']:.2f}m")
    return flags


def run_task(task, suite, out_root, robot, rendering_mode, log_fh):
    """Build one task, render its three views at t=0, tear it down. Returns the record dict."""
    t0 = time.perf_counter()
    rec = {"task": task, "suite": suite, "status": "building"}
    frame_dir = os.path.join(out_root, "frames", task)
    os.makedirs(frame_dir, exist_ok=True)

    env = RealmEnvironmentDynamic(
        config_path=CONFIG_ROOT,
        task_cfg_path=f"{suite}/{task}/default.yaml",
        perturbations=["Default"],
        multi_view=True,
        robot=robot,
        rendering_mode=rendering_mode,
    )
    rec["build_s"] = round(time.perf_counter() - t0, 2)
    rec["instruction"] = env.instruction
    rec["task_type"] = env.task_type
    rec["scene"] = f"{env.scene_model}/{env.scene_part}"
    rec["robot_pos"] = _vec(env.robot_pos)
    rec["use_droid_with_base"] = bool(env.use_droid_with_base)
    # external_sensors is None when the env config declared none; here it is cam1 + cam2.
    rec["camera_pos"] = {s.name: _vec(s.get_position_orientation()[0])
                         for s in (env.omnigibson_env.external_sensors or {}).values()}

    obs, _ = env.reset()
    obs, _, _, _, _ = env.warmup(obs)
    # The single step whose observation is saved: hold the reset pose, gripper open.
    action = np.concatenate((env.warmup_ee_cmd() if env.ee_control else env.reset_qpos[:7],
                             np.atleast_1d(1.0)))
    # step() returns the LATCHED FRACTION of the rubric reached, not the per-stage dict -- the dict
    # is env.task_progression, mutated in place by recompute_task_progression().
    obs, progression_fraction, _, _, _ = env.step(action)
    rec["step_s"] = round(time.perf_counter() - t0, 2)

    extracted = extract_from_obs(obs, robot_name=robot)
    views, stats = [], {}
    for name, idx in VIEWS:
        im = _to_uint8(extracted[idx])
        views.append((name, im))
        stats[name] = {"shape": list(im.shape),
                       "mean": round(float(im.mean()), 3),
                       "std": round(float(im.std()), 3)}
        Image.fromarray(im).resize(VIEW_SIZE, Image.BILINEAR).save(
            os.path.join(frame_dir, f"{name}.jpg"), quality=JPEG_QUALITY)
    rec["view_stats"] = stats

    ee_pos, _ = env.get_ee_pose()
    rec["ee_pos"] = _vec(ee_pos)
    rec["gripper_state"] = round(float(extracted[6]), 4)
    rec["progression_fraction"] = round(float(progression_fraction), 4)
    rec["task_progression"] = {k: bool(v) for k, v in env.task_progression.items()}
    rec["relations"] = _relations(env)
    rec["self_collision"], rec["env_collision"] = (bool(x) for x in env.check_collisions())

    authored = {c["name"]: c.get("position") for c in env.cfg["objects"]}
    rec["objects"] = {
        "main": [_obj_record(o, authored.get(o.name)) for o in env.main_objects],
        "target": [_obj_record(o, authored.get(o.name)) for o in env.target_objects],
        "distractors": [_obj_record(o, authored.get(o.name)) for o in env.distractors],
    }
    if env.main_objects:
        mo_pos = np.asarray(rec["objects"]["main"][0]["aabb_center"], dtype=float)
        rec["reach_distance"] = round(float(np.linalg.norm(mo_pos - np.asarray(env.robot_pos))), 4)

    rec["flags"] = _flags(rec, env.task_type)
    rec["status"] = "ok"

    caption = f"{task}   [{env.task_type}]   \"{env.instruction}\""
    sub = (f"scene={rec['scene']}  reach={rec.get('reach_distance')}  "
           f"inside={rec['relations'].get('inside')} on_top={rec['relations'].get('on_top')}  "
           f"flags={','.join(rec['flags']) or 'none'}")
    _panel(views, caption, sub).save(os.path.join(frame_dir, "panel.jpg"), quality=JPEG_QUALITY)

    rec["total_s"] = round(time.perf_counter() - t0, 2)
    _say(f"[ok] {task}  {rec['total_s']}s  means="
         f"{ {k: v['mean'] for k, v in stats.items()} }  flags={rec['flags'] or 'none'}", log_fh)
    return rec


def teardown(log_fh):
    """Return the process to a state where the next og.Environment can be built.

    Mirrors this fork's own tests/test_envs.py: og.clear() straight after stepping, which closes the
    stage and relaunches a fresh simulator carrying the previous one's dt/viewer/device settings.
    NO explicit og.sim.stop() first -- Simulator._partial_clear() stops the physics itself, and
    stopping ahead of it is what OG's own tests do NOT do.

    This is the one operation that is expected to be flaky. Measured 2026-08-24 (job 195340) on
    realm_og391_v3.sif: og.clear() -> scene.clear() -> VisionSensor.remove() -> the replicator's
    annotator detach dies with `TypeError: Invalid NodeObj object in Py_Node in getAttributes`,
    i.e. a syntheticdata node that was already destroyed. Three vision sensors are live here (two
    external + the wrist), and og.clear() aborts before close_stage(), so the process cannot build
    anything afterwards.

    So a failure here is NOT swallowed: it propagates, main() stops the shard, and the launcher
    re-invokes this script for the tasks that have no record yet. That degrades one-boot-per-shard
    to one-boot-per-task without losing a task, which is why the resume path exists at all.
    """
    og.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="run directory; frames/ and shard*.json go here")
    p.add_argument("--suite", default="REALM_DROID100")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--tasks", default=None, help="comma-separated task dir names, overrides sharding")
    p.add_argument("--limit", type=int, default=None, help="stop after N tasks (smoke testing)")
    p.add_argument("--robot", default="DROID")
    p.add_argument("--rendering_mode", default="rt")
    args = p.parse_args()

    suite_dir = os.path.join(CONFIG_ROOT, "tasks", args.suite)
    all_tasks = sorted(d for d in os.listdir(suite_dir)
                       if os.path.isfile(os.path.join(suite_dir, d, "default.yaml")))
    if args.tasks:
        mine = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in mine if t not in all_tasks]
        assert not unknown, f"no such task(s) in {args.suite}: {unknown}"
    else:
        mine = [t for i, t in enumerate(all_tasks) if i % args.num_shards == args.shard]
    if args.limit:
        mine = mine[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    tag = f"shard{args.shard:02d}"
    json_path = os.path.join(args.out, f"{tag}.json")
    log_fh = open(os.path.join(args.out, f"{tag}.log"), "a")

    # RESUME. Anything already carrying a record -- ok OR error -- is done: a per-task exception is
    # a finding about that task, and re-running it in the next process would just reproduce it.
    # Only og.clear() failures leave tasks record-less, and those are what a relaunch picks up.
    summary = {"shard": args.shard, "num_shards": args.num_shards, "suite": args.suite,
               "robot": args.robot, "rendering_mode": args.rendering_mode, "tasks": mine,
               "records": [], "slurm_jobs": []}
    if os.path.isfile(json_path):
        try:
            with open(json_path) as fh:
                prev = json.load(fh)
            summary["records"] = prev.get("records", [])
            summary["slurm_jobs"] = prev.get("slurm_jobs", [])
        except (OSError, ValueError) as exc:
            _say(f"[{tag}] ignoring unreadable {json_path}: {exc}", log_fh)
    summary.pop("aborted_after", None)
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        summary["slurm_jobs"].append(job)
    done = {r["task"] for r in summary["records"]}
    todo = [t for t in mine if t not in done]

    _say(f"[{tag}] suite={args.suite} robot={args.robot} rendering={args.rendering_mode}", log_fh)
    _say(f"[{tag}] {len(mine)} of {len(all_tasks)} tasks; {len(done)} already recorded; "
         f"{len(todo)} to do: {todo}", log_fh)

    def flush_summary():
        summary["n_ok"] = sum(r["status"] == "ok" for r in summary["records"])
        summary["n_tasks"] = len(mine)
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    flush_summary()
    if not todo:
        _say(f"[{tag}] nothing left to do", log_fh)
        log_fh.close()
        return 0

    set_sim_config(robot=args.robot)
    for i, task in enumerate(todo):
        _say(f"\n===== [{tag}] {i + 1}/{len(todo)}  {task} =====", log_fh)
        try:
            rec = run_task(task, args.suite, args.out, args.robot, args.rendering_mode, log_fh)
        except Exception:
            rec = {"task": task, "suite": args.suite, "status": "error",
                   "traceback": traceback.format_exc()}
            _say(f"[FAIL] {task}\n{rec['traceback']}", log_fh)
        summary["records"].append(rec)
        flush_summary()
        try:
            teardown(log_fh)
        except Exception:
            # og.clear() aborts before close_stage(), so this process can no longer build an
            # environment -- carrying on would emit a run of identical failures that read as task
            # defects. Exit non-zero instead; the launcher relaunches for whatever has no record.
            _say(f"[{tag}] og.clear() FAILED after {task} -- ending this process, "
                 f"{len(todo) - i - 1} task(s) left for the relaunch\n{traceback.format_exc()}",
                 log_fh)
            summary["aborted_after"] = task
            flush_summary()
            log_fh.close()
            # Not og.shutdown(): the simulator is half-cleared and a clean shutdown can hang here.
            os._exit(2)

    n_ok = sum(r["status"] == "ok" for r in summary["records"])
    _say(f"\n[{tag}] DONE {n_ok}/{len(mine)} ok -> {json_path}", log_fh)
    log_fh.close()
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
