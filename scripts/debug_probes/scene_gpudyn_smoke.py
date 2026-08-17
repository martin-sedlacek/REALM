"""Smallest run that exercises every stage GPU dynamics breaks, so the port can be iterated on.

`gm.USE_GPU_DYNAMICS = True` is not a flag flip in this stack -- it is a systemic device port.
OmniGibson mixes USD/fabric reads (always CPU) with physics-view reads (on `og.sim.device`, which
becomes `cuda:0` under GPU dynamics) and normalises the device nowhere, so the failure walks up the
stack one frame at a time: fix one site, the next one fires. Iterating on
`curl_press_direction.py` costs a full sweep per iteration; this costs one env build.

Stages, in the order they break:

    1. scene load       -- Environment(...) -> og.sim.play() -> _non_physics_step() -> serialize()
    2. reset + warmup   -- the state dump/load round trip
    3. physics steps    -- ControllableObjectViewAPI.post_physics_step, the compute backend
    4. contact reads    -- RigidContactAPI index/mask tensors against a device-native matrix
    5. proprioception   -- th.cat over _proprio_obs

Each prints `STAGE_OK <n>` on success, so a log that stops after `STAGE_OK 3` names the stage
without needing the traceback. The traceback is printed too, with the failing frame's file:line
called out, because that is the thing the next fix needs.

    REALM_GPU_DYNAMICS=1 MODE=oglite ./scripts/clara/interactive/rr python -u \
        /app/scripts/debug_probes/scene_gpudyn_smoke.py --tag scene_gpudyn

Isaac exits 139 at teardown regardless of outcome: grep `GPUDYN_SMOKE`, never the exit code.
The JSON is written before anything is printed, for the same reason.
"""
import argparse
import json
import os
import traceback

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze")
ap.add_argument("--tag", default="scene_gpudyn")
ap.add_argument("--steps", type=int, default=10)
args = ap.parse_args()

RESULT = dict(tag=args.tag, robot=args.robot, stages={}, device=None, gpu_dynamics=None,
              failed_stage=None, error=None, frames=[])
JSON = os.path.join(args.out, f"{args.tag}_smoke.json")


def flush():
    """Write BEFORE printing. Isaac's teardown hang makes a time-limit kill routine, and a
    block-buffered tail dies with the process."""
    os.makedirs(args.out, exist_ok=True)
    with open(JSON, "w") as f:
        json.dump(RESULT, f, indent=2, default=str)


def stage(n, name, fn):
    if RESULT["failed_stage"] is not None:
        return None
    print(f"\n{'=' * 90}\nSTAGE {n}: {name}\n{'=' * 90}", flush=True)
    try:
        out = fn()
    except BaseException as e:                    # noqa: BLE001 -- the point is to report anything
        RESULT["failed_stage"] = n
        RESULT["error"] = f"{type(e).__name__}: {e}"
        # The frames are what the next fix needs, so pull them out rather than leaving them in a
        # wall of traceback text.
        RESULT["frames"] = [f"{fr.filename}:{fr.lineno} in {fr.name}  |  {fr.line}"
                            for fr in traceback.extract_tb(e.__traceback__)]
        flush()
        print(f"\nGPUDYN_SMOKE_FAIL stage={n} ({name})\n  {type(e).__name__}: {e}", flush=True)
        print("  frames (innermost last):", flush=True)
        for fr in RESULT["frames"]:
            print(f"    {fr}", flush=True)
        return None
    RESULT["stages"][name] = "OK"
    flush()
    print(f"STAGE_OK {n} {name}", flush=True)
    return out


import omnigibson as og                                                         # noqa: E402
from omnigibson.macros import gm                                                # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI                          # noqa: E402

from realm.sim_config import set_sim_config                                     # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic              # noqa: E402

set_sim_config(robot=args.robot)
print(f"[smoke] REALM_GPU_DYNAMICS={os.environ.get('REALM_GPU_DYNAMICS')} "
      f"gm.USE_GPU_DYNAMICS={gm.USE_GPU_DYNAMICS}", flush=True)
RESULT["gpu_dynamics"] = bool(gm.USE_GPU_DYNAMICS)
flush()

state = {}


def s1():
    state["env"] = RealmEnvironmentDynamic(
        config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
        multi_view=False, no_rendering=False, rendering_mode="rt", robot=args.robot,
    )
    RESULT["device"] = str(og.sim.device)
    print(f"  og.sim.device = {og.sim.device}", flush=True)
    # If the device is cpu here the run proves nothing about GPU dynamics -- say so loudly rather
    # than letting a green log be read as a pass. env_base.py:75 hardcodes "cpu" when the env config
    # omits a device, so this is a real way to get a meaningless pass.
    if RESULT["gpu_dynamics"] and str(og.sim.device) == "cpu":
        raise RuntimeError("gm.USE_GPU_DYNAMICS is True but og.sim.device is cpu -- this run would "
                           "not have exercised the GPU path at all")


def s2():
    env = state["env"]
    obs, _ = env.reset()
    state["obs"] = env.warmup(obs)[0]


def s3():
    env = state["env"]
    robot = env.robot
    cmd = np.asarray(env.reset_qpos[:7], dtype=np.float64)
    for i in range(args.steps):
        env.step(np.concatenate([cmd, [-1.0]]))
    q = robot.get_joint_positions()
    print(f"  joint positions on {q.device}: {q[:7]}", flush=True)
    state["robot"] = robot


def s4():
    robot = state["robot"]
    scene_idx = robot.scene.idx
    links = set(robot.links.values())
    rows = RigidContactAPI.get_contact_row_indices(scene_idx, links)
    cols = RigidContactAPI.get_contact_col_indices(scene_idx, links)
    print(f"  contact rows {tuple(rows.shape)} on {rows.device}, cols {tuple(cols.shape)} on "
          f"{cols.device}", flush=True)
    pairs = RigidContactAPI.get_contact_pairs(scene_idx=scene_idx, query_set=links,
                                              with_set=None, current_only=True)
    print(f"  {len(pairs)} contact pairs involving the robot", flush=True)
    # The masks and the batch query are separate code paths from the pair lookup.
    rm = RigidContactAPI.get_contact_row_mask(scene_idx, links)
    cm = RigidContactAPI.get_contact_col_mask(scene_idx, links)
    print(f"  row mask {tuple(rm.shape)} on {rm.device}, col mask {tuple(cm.shape)} on {cm.device}",
          flush=True)
    RESULT["stages"]["n_contact_pairs"] = len(pairs)


def s5():
    robot = state["robot"]
    proprio, _ = robot.get_proprioception()
    print(f"  proprioception {tuple(proprio.shape)} on {proprio.device}", flush=True)
    # Normalized joint positions go through the joint-limit properties, a separate device path.
    qn = robot.get_joint_positions(normalized=True)
    print(f"  normalized joint positions on {qn.device}, range "
          f"[{float(qn.min()):.3f}, {float(qn.max()):.3f}]", flush=True)


stage(1, "scene load", s1)
stage(2, "reset + warmup", s2)
stage(3, f"{args.steps} env steps", s3)
stage(4, "contact reads", s4)
stage(5, "proprioception + normalized limits", s5)

flush()
if RESULT["failed_stage"] is None:
    print(f"\nGPUDYN_SMOKE_OK device={RESULT['device']} gpu_dynamics={RESULT['gpu_dynamics']} "
          f"all 5 stages", flush=True)
else:
    print(f"\nGPUDYN_SMOKE_FAILED at stage {RESULT['failed_stage']}", flush=True)
print(f"json: {JSON}", flush=True)
og.shutdown()
