"""Persistent REALM/OmniGibson debug session for the robolab Franka + Robotiq 2F-85 asset.

Boots the sim + env ONCE (~5 min), then watches /dbg/inbox for *.py snippets, execs each in a
shared namespace, and writes captured stdout to /dbg/outbox/<name>.out. A probe costs ~1 s
instead of a fresh 5-minute boot.

Differs from tmp/dbg_session/pump.py (the 2026-08-10 velocity-fix session):
  - the robot config is read from $REALM_ROBOT (default DROID_robolab_v2), not hardcoded to DROID,
    so the same pump serves the stock asset for A/B without an edit.
  - it connects an InferenceClient to the pi0.5 server on $REALM_PORT, and exposes rollout(),
    a faithful reduction of realm/eval.py's loop. The known robolab gap (task_progression 0.0,
    stage REACH) only reproduces under the real policy, unlike the 3.9.1 gripper bug, which
    synthetic actions were enough for.

Shared namespace: env, robot, og, np, th, obs, client, rollout, extract_from_obs.
Touch /dbg/STOP to shut down cleanly.
"""

import glob
import io
import os
import sys
import time
import traceback
from queue import Queue

INBOX = "/dbg/inbox"
OUTBOX = "/dbg/outbox"
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)
for stale in ("/dbg/READY", "/dbg/BOOT_FAILED", "/dbg/STOP"):
    if os.path.exists(stale):
        os.remove(stale)

ROBOT = os.environ.get("REALM_ROBOT", "DROID_robolab_v2")
TASK_CFG = os.environ.get(
    "REALM_TASK_CFG", "REALM_DROID10/put_green_block_into_bowl/default.yaml"
)
PORT = int(os.environ.get("REALM_PORT", "8500"))
MODEL_TYPE = os.environ.get("REALM_MODEL_TYPE", "openpi")

print(f"[pump] robot={ROBOT} task={TASK_CFG} port={PORT}", flush=True)
print("[pump] importing omnigibson ...", flush=True)
try:
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot

    import omnigibson as og
    from omnigibson.macros import gm  # noqa: F401 -- omnigibson.macros.gm import has side effects

    from realm.sim_config import set_sim_config
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.inference import InferenceClient, extract_from_obs

    # Mirror eval.py: gm.* must be set before the env is constructed.
    set_sim_config(robot=ROBOT)

    print("[pump] creating env ...", flush=True)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=TASK_CFG,
        perturbations=["Default"],
        multi_view=False,
        no_rendering=False,
        rendering_mode="rt",
        robot=ROBOT,
    )
    print("[pump] env created; resetting ...", flush=True)
    obs, _ = env.reset()
    robot = env.robot
    print(f"[pump] reset done; robot={robot.name} prim={robot.prim_path}", flush=True)
except Exception:
    with open("/dbg/BOOT_FAILED", "w") as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
    sys.exit(1)

# The policy server is optional: a boot that survives without it still serves the kinematic and
# camera probes, and the client can be rebuilt from a snippet once the server is up.
client = None
try:
    client = InferenceClient(MODEL_TYPE, host="127.0.0.1", port=PORT)
    print(f"[pump] policy client connected on 127.0.0.1:{PORT}", flush=True)
except Exception:
    traceback.print_exc()
    print("[pump] WARNING: no policy client; rollout() unavailable until reconnected", flush=True)


def rollout(max_steps=300, horizon=8, render_on_demand=True, max_render_interval=16,
            n_pre_obs_renders=2, reset=True, verbose=True):
    """Closed-loop rollout against the pi0.5 server. A reduction of realm/eval.py's loop.

    Drops the video recorder and the CSV report; keeps the action-chunk buffer, the gripper
    binarisation, render-on-demand and the task-progression terminal countdown, because those are
    what decide whether the policy commits. Returns a dict of per-step traces.
    """
    global obs
    if client is None:
        raise RuntimeError("no policy client -- is the pi0.5 server up on this node?")

    if reset:
        obs, _ = env.reset()
        obs, _, _, _, _ = env.warmup(obs)
        client.reset()

    action_buffer = Queue()
    qpos, actions, ee_poses, progressions = [], [], [], []
    t, task_progression, terminal_steps = 0, 0.0, 15
    steps_since_render = 0
    t0 = time.perf_counter()

    while t < max_steps and terminal_steps > 0:
        (base_im, base_depth, base_im_second, base_depth_second,
         wrist_im, robot_state, gripper_state) = extract_from_obs(obs, robot_name=env.robot.name)

        ee_pos, ee_rot = env.get_ee_pose()
        ee_poses.append(np.asarray(ee_pos.cpu() if hasattr(ee_pos, "cpu") else ee_pos))

        if action_buffer.empty():
            _ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
            _ee_rot = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.array(ee_rot)
            _ee_pose_world = np.concatenate([_ee_pos, Rot.from_quat(_ee_rot).as_euler("xyz")])
            cartesian_position = env._world2robot(_ee_pose_world).astype(np.float32)

            chunk = client.infer(
                env.instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                use_base_im_second=(env.task_type == "open_close_drawer"
                                    if hasattr(env, "task_type") else False),
                ee_control=env.ee_control,
                cartesian_position=cartesian_position,
            )
            if chunk.ndim == 2:
                for a in chunk[:horizon]:
                    action_buffer.put(np.squeeze(a))
            else:
                action_buffer.put(chunk)

        qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))
        action = action_buffer.get()
        actions.append(action)

        new_action = action.copy()
        new_action[-1] = 1 if action[-1] > 0.5 else -1

        if render_on_demand:
            need_render = action_buffer.empty() or (steps_since_render + 1) >= max_render_interval
            with og.sim.render_on_step(need_render):
                obs, curr_progression, terminated, truncated, info = env.step(
                    new_action, n_render_iterations=n_pre_obs_renders if need_render else 1
                )
            steps_since_render = 0 if need_render else steps_since_render + 1
        else:
            obs, curr_progression, terminated, truncated, info = env.step(new_action)

        progressions.append(float(curr_progression))
        if curr_progression > task_progression:
            task_progression = curr_progression
            if verbose:
                print(f"[rollout] t={t} progression -> {task_progression:.3f}", flush=True)
        if task_progression >= 1.0:
            terminal_steps -= 1
        t += 1

    wall = time.perf_counter() - t0
    out = {
        "steps": t,
        "task_progression": task_progression,
        "qpos": np.stack(qpos),
        "actions": np.stack(actions),
        "ee_poses": np.stack(ee_poses),
        "progressions": np.array(progressions),
        "ms_per_step": 1000.0 * wall / max(t, 1),
    }
    if verbose:
        print(f"[rollout] {t} steps in {wall:.1f}s ({out['ms_per_step']:.0f} ms/step), "
              f"task_progression={task_progression:.3f}", flush=True)
    return out


with open("/dbg/READY", "w") as f:
    f.write("ready\n")
print("[pump] READY -- drop snippets in /dbg/inbox", flush=True)

seen = set()
while not os.path.exists("/dbg/STOP"):
    for path in sorted(glob.glob(f"{INBOX}/*.py")):
        if path in seen:
            continue
        seen.add(path)
        name = os.path.basename(path)[:-3]
        buf = io.StringIO()
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        try:
            exec(compile(open(path).read(), path, "exec"), globals())
        except Exception:
            traceback.print_exc(file=buf)
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
        # Write to a temp name first so a reader never sees a half-written file.
        tmp = f"{OUTBOX}/.{name}.partial"
        with open(tmp, "w") as f:
            f.write(buf.getvalue())
        os.replace(tmp, f"{OUTBOX}/{name}.out")
        print(f"[pump] ran {name} ({len(buf.getvalue())} bytes)", flush=True)
    time.sleep(0.4)

print("[pump] STOP seen, shutting down", flush=True)
og.shutdown()
