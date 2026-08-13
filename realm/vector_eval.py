"""Vectorized REALM evaluation: N rollouts in one simulator, stepped together.

`realm/eval.py:evaluate()` is single-env -- one rollout at a time, one Isaac boot amortised over
`repeats` sequential rollouts. This runs `num_envs` rollouts concurrently inside one
`RealmVectorEnvironment`, in waves of `num_envs`, and writes the same artifacts
(`reports/*.csv`, `qpos/`, `actions/`, `videos/` parquets) so downstream tooling does not care which
path produced them.

Deliberately NOT batched inference: the policy is called once per member per chunk boundary, in a
loop. Batching is a separate change and would hide desync bugs behind a fixed-shape batch.

**Members desync.** Rollouts end at different times -- one succeeds at step 180, another runs to
`max_steps`. `og.sim.step()` advances every scene regardless, so a finished member cannot simply
stop. It is marked inactive, its result is finalised immediately, and it keeps receiving its last
action as a hold command while the others run on. Nothing about an inactive member is recorded.

Metric definitions, termination (15 terminal steps after task_progression hits 1.0), the gripper
sign convention and the drops correction are copied from `evaluate()` so the two paths produce
comparable numbers.
"""
import datetime
import time
from queue import Queue

import numpy as np
import omnigibson as og
from scipy.spatial.transform import Rotation as Rot

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.inference import InferenceClient, extract_from_obs
from realm.realm_logging import VideoRecorder, append_trajectory, append_video, save_results
from realm.sim_config import set_sim_config

TERMINAL_STEPS = 15
CONTROL_DT = 1.0 / 15.0


class _Member:
    """Per-rollout bookkeeping for one vector-env member."""

    def __init__(self, env, run_id, log_dir, task, perturbation, no_record):
        self.env = env
        self.run_id = run_id
        self.active = True
        self.buf = Queue()
        self.qpos, self.actions, self.ee_poses = [], [], []
        self.task_progression = 0.0
        self.tp_timestamps = []
        self.terminal_steps = TERMINAL_STEPS
        self.collisions_self = self.collisions_env = self.drops = 0
        self._self_col_active = self._env_col_active = self._was_grasping = False
        self.last_action = None
        self.steps = 0
        self.no_record = no_record
        self.recorder = None
        if not no_record:
            ts = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            self.recorder = VideoRecorder(log_dir, ts, run_id, task, perturbation)

    # -- metrics, identical in definition to realm/eval.py ------------------------------------
    def observe(self, obs, robot_name, obs_is_fresh=True):
        env = self.env
        base_im, _, base_im_second, _, wrist_im, robot_state, gripper_state = extract_from_obs(
            obs, robot_name=robot_name)

        ee_pos, ee_rot = env.get_ee_pose()
        self.ee_poses.append(ee_pos)

        is_self_col, is_env_col = env.check_collisions()
        if is_self_col and not self._self_col_active:
            self.collisions_self += 1
        self._self_col_active = is_self_col
        if is_env_col and not self._env_col_active:
            self.collisions_env += 1
        self._env_col_active = is_env_col

        is_grasping = env.check_grasp_condition(obs)
        if self._was_grasping and not is_grasping:
            placed = False
            if getattr(env, "task_type", None) in ("put", "stack") and len(env.target_objects) > 0:
                mo, target = env.main_objects[0], env.target_objects[0]
                placed = (mo.states[og.object_states.Inside].get_value(target)
                          or mo.states[og.object_states.OnTop].get_value(target))
            if not placed:
                self.drops += 1
        self._was_grasping = is_grasping

        # Under render_on_demand `obs` only carries a new frame on render steps; recording the
        # blind steps in between would pad the mp4 with duplicates of the last rendered frame.
        if self.recorder is not None and obs_is_fresh:
            self.recorder.add_frame(base_im, wrist_im, base_im_second)
        self.qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))
        return base_im, base_im_second, wrist_im, robot_state, gripper_state, ee_pos, ee_rot

    def note_progression(self, curr, t):
        if curr > self.task_progression:
            self.task_progression = curr
            self.tp_timestamps.append(t)
        if self.task_progression >= 1.0:
            self.terminal_steps -= 1
        self.steps += 1
        return self.terminal_steps <= 0

    def result(self, task, perturbation, model_type):
        env = self.env
        qpos_arr = np.stack(self.qpos)
        joints = qpos_arr[:, :7]
        if len(joints) > 4:
            vel = np.diff(joints, axis=0) / CONTROL_DT
            acc = np.diff(vel, axis=0) / CONTROL_DT
            jerk = np.diff(acc, axis=0) / CONTROL_DT
            joint_vel_var = np.mean(np.var(vel, axis=0) * len(vel))
            joint_acc_var = np.mean(np.var(acc, axis=0) * len(acc))
            joint_jerk = np.mean(np.linalg.norm(jerk, axis=1))
            joint_path = np.sum(np.linalg.norm(np.diff(joints, axis=0), axis=1))
        else:
            joint_vel_var = joint_acc_var = joint_jerk = joint_path = 0.0

        ee_arr = np.stack(self.ee_poses)
        if len(ee_arr) > 4:
            cvel = np.diff(ee_arr, axis=0) / CONTROL_DT
            cacc = np.diff(cvel, axis=0) / CONTROL_DT
            cjerk = np.diff(cacc, axis=0) / CONTROL_DT
            cart_jerk = np.mean(np.linalg.norm(cjerk, axis=1))
            cart_path = np.sum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))
        else:
            cart_jerk = cart_path = 0.0

        stage = "SUCCESS"
        if env.task_progression is not None:
            for name, done in env.task_progression.items():
                if not done:
                    stage = name
                    break
        else:
            stage = "N/A"

        drops = self.drops
        if self.task_progression == 1.0 and getattr(env, "task_type", None) in ("put", "stack"):
            drops = max(0, drops - 1)

        entry = {
            "run_id": self.run_id, "task": task, "perturbation": perturbation,
            "instruction": env.instruction, "model": model_type, "real2sim": "Simulated",
            "env": "REALM", "task_progression": self.task_progression,
            "task_progression_timestamps": self.tp_timestamps, "stage": stage,
            "binary_SR": 1.0 if self.task_progression == 1.0 else 0.0,
            "joint_vel_var": joint_vel_var, "joint_acc_var": joint_acc_var,
            "joint_jerk": joint_jerk, "joint_path_length": joint_path,
            "cart_path_length": cart_path, "cart_jerk": cart_jerk,
            "collisions_self": self.collisions_self, "collisions_env": self.collisions_env,
            "object_drops": drops,
        }
        entry["qpos"] = qpos_arr.tolist()
        entry["actions"] = np.stack(self.actions).tolist()
        return entry


def evaluate_vectorized(
        num_envs=4, task_id=0, perturbation_id=0, repeats=25, max_steps=500, horizon=8,
        model_type="openpi", model_name="model", port=8000, host="127.0.0.1",
        log_dir="/logs", rendering_mode="rt", robot="DROID", multi_view=False,
        no_record=False, task_cfg_path=None,
        render_on_demand=True, n_pre_obs_renders=2, max_render_interval=8,
):
    start = time.perf_counter()
    set_sim_config(robot=robot)
    task = SUPPORTED_TASKS[task_id] if task_cfg_path is None else task_cfg_path.split("/")[-2]
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]
    if task_cfg_path is None:
        task_cfg_path = f"REALM_DROID10/{task}/default.yaml"

    print(f"[vec_eval] building {num_envs} environments...", flush=True)
    vec_env = RealmVectorEnvironment(
        num_envs, task_cfg_path=task_cfg_path, perturbations=[perturbation],
        robot=robot, rendering_mode=rendering_mode, multi_view=multi_view,
    )
    print(f"[vec_eval] {num_envs} envs ready at {time.perf_counter() - start:.1f}s", flush=True)

    # One client per member. Inference is still strictly sequential -- this only avoids sharing one
    # websocket's state across concurrent rollouts.
    clients = [InferenceClient(model_type, port, host) for _ in range(num_envs)]

    results, results_filename = [], None
    run_id = 0
    wave = 0
    while run_id < repeats:
        n_record = min(num_envs, repeats - run_id)
        wave += 1
        print(f"[vec_eval] wave {wave}: rollouts {run_id}..{run_id + n_record - 1} "
                    f"({n_record} of {num_envs} members recorded)", flush=True)

        step_results = vec_env.warmup()
        members = [
            _Member(vec_env.envs[i], run_id + i, log_dir, task, perturbation, no_record)
            if i < n_record else None
            for i in range(num_envs)
        ]
        for c in clients:
            c.reset()

        t = 0
        steps_since_render = 0
        obs_is_fresh = True   # warmup() rendered every step
        wave_start = time.perf_counter()
        while t < max_steps and any(m is not None and m.active for m in members):
            actions = []
            for i, m in enumerate(members):
                if m is None or not m.active:
                    # Inactive members still have to be given something: og.sim.step() advances
                    # their scene either way. Hold the last commanded action rather than zeros,
                    # which would drive the arm toward the zero pose.
                    env_i = vec_env.envs[i]
                    hold = (m.last_action if (m is not None and m.last_action is not None)
                            else env_i.warmup_action(0, env_i.warmup_ee_cmd()))
                    actions.append(hold)
                    continue

                obs = step_results[i][0]
                (base_im, base_im_second, wrist_im, robot_state,
                 gripper_state, ee_pos, ee_rot) = m.observe(
                    obs, vec_env.envs[i].robot.name, obs_is_fresh=obs_is_fresh)

                if m.buf.empty():
                    env = vec_env.envs[i]
                    _p = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
                    _r = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.array(ee_rot)
                    cartesian_position = env._world2robot(
                        np.concatenate([_p, Rot.from_quat(_r).as_euler("xyz")])).astype(np.float32)
                    chunk = clients[i].infer(
                        env.instruction, base_im, base_im_second, wrist_im, robot_state,
                        gripper_state,
                        use_base_im_second=(getattr(env, "task_type", None) == "open_close_drawer"),
                        ee_control=env.ee_control, cartesian_position=cartesian_position,
                    )
                    if chunk.ndim == 2:
                        for a in chunk[:horizon]:
                            m.buf.put(np.squeeze(a))
                    else:
                        m.buf.put(chunk)

                raw = m.buf.get()
                m.actions.append(raw)
                act = raw.copy()
                # (1,0) -> (1,-1); same convention as realm/eval.py
                act[-1] = 1 if raw[-1] > 0.5 else -1
                m.last_action = act
                actions.append(act)

            if render_on_demand:
                # og.sim.render_on_step() is GLOBAL -- one flag for every scene -- so the decision
                # has to be the OR across active members: if ANY of them needs fresh images next
                # iteration, the whole batch renders. In practice members stay in phase, because
                # each active member pops exactly one action per step and they all refill on the
                # same boundary, so this costs no more renders than the single-env path.
                # max_render_interval bounds how far the renderer may lag physics.
                need_render = any(m is not None and m.active and m.buf.empty() for m in members)
                need_render = need_render or (steps_since_render + 1) >= max_render_interval
                with og.sim.render_on_step(need_render):
                    step_results = vec_env.step(
                        actions,
                        n_render_iterations=n_pre_obs_renders if need_render else 1,
                    )
                steps_since_render = 0 if need_render else steps_since_render + 1
                obs_is_fresh = need_render
            else:
                step_results = vec_env.step(actions)
                obs_is_fresh = True

            for i, m in enumerate(members):
                if m is None or not m.active:
                    continue
                if m.note_progression(step_results[i][1], t):
                    m.active = False
                    print(f"[vec_eval]   member {i} (run {m.run_id}) finished at step {t}, "
                                f"TP={m.task_progression}", flush=True)
            t += 1

        print(f"[vec_eval] wave {wave} stepped {t} times in "
                    f"{time.perf_counter() - wave_start:.1f}s", flush=True)

        for m in members:
            if m is None:
                continue
            entry = m.result(task, perturbation, model_type)
            if not m.no_record:
                video_bytes = m.recorder.get_video_bytes()
                entry["video"] = video_bytes
                append_video(log_dir, task, perturbation, m.run_id, video_bytes)
                m.recorder.cleanup()
            append_trajectory(log_dir, task, perturbation, m.run_id,
                              np.stack(m.qpos), np.stack(m.actions))
            results.append(entry)
            print(f"[vec_eval]   run {m.run_id}: SR={entry['binary_SR']} "
                        f"TP={entry['task_progression']} stage={entry['stage']} "
                        f"steps={m.steps} col_env={entry['collisions_env']}", flush=True)

        results_filename = save_results(results, log_dir + "/reports", task, perturbation,
                                        filename=results_filename)
        run_id += n_record

    sr = float(np.mean([r["binary_SR"] for r in results]))
    tp = float(np.mean([r["task_progression"] for r in results]))
    print(f"[vec_eval] DONE {len(results)} rollouts in "
                f"{time.perf_counter() - start:.1f}s -- SR={sr:.3f} TP={tp:.3f}", flush=True)
    save_results(results, log_dir + "/reports", task, perturbation)
    return results
