from realm.config.shared import (  # noqa: F401
    CONTROL_DT,
    CONTROL_HZ,
    GRIPPER_OPEN_ABOVE_HALF,
    GRIPPER_OPEN_BELOW_HALF,
    PLACEMENT_TASK_TYPES,
    SHORT_TRAJECTORY_SAMPLES,
    TERMINAL_STEPS,
)

from queue import Queue
from typing import Any, NamedTuple

import numpy as np
import omnigibson as og
from scipy.spatial.transform import Rotation

from realm.environments.task_progression import DRAWER_TASK_TYPES
from realm.inference import extract_from_obs
from realm.realm_logging import append_trajectory, append_video






def wants_base_im_second(task_type, base_im_second):

    return task_type in DRAWER_TASK_TYPES and base_im_second is not None


# Policy and REALM gripper conventions differ.


def resolve_task(task_id, task_cfg_path, supported_tasks, name_includes_config):

    if task_cfg_path is None:
        task = supported_tasks[task_id]
        return task, f"REALM_DROID10/{task}/default.yaml"

    task = task_cfg_path.split("/")[-2]
    if name_includes_config:
        config_name = task_cfg_path.split("/")[-1].replace(".yaml", "").replace(".cfg", "")
        if config_name != "default":
            task = f"{task}_{config_name}"
    return task, task_cfg_path


def gripper_is_inverted(model_type):

    if model_type in GRIPPER_OPEN_ABOVE_HALF:
        return False
    if model_type in GRIPPER_OPEN_BELOW_HALF:
        return True
    raise NotImplementedError()


def binarize_gripper(gripper_value, inverted=False):
    is_open = gripper_value < 0.5 if inverted else gripper_value > 0.5
    return 1 if is_open else -1


def enqueue_action_chunk(buffer, chunk, horizon):
    if len(chunk.shape) == 2:
        for action in chunk[:horizon]:
            buffer.put(np.squeeze(action))
    elif len(chunk.shape) < 2:
        buffer.put(chunk)
    else:
        assert len(chunk.shape) <= 2, (
            f"Unsupported number of dimensions in action chunk with shape: {chunk.shape}. "
            "The chunk is expected to be 2D."
        )


def robot_frame_ee_pose(env, ee_pos, ee_quat):

    position = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
    quaternion = ee_quat.cpu().numpy() if hasattr(ee_quat, "cpu") else np.array(ee_quat)
    world_pose = np.concatenate([position, Rotation.from_quat(quaternion).as_euler("xyz")])
    return env._world2robot(world_pose).astype(np.float32)


def is_placed_on_target(env):
    if getattr(env, "task_type", None) not in PLACEMENT_TASK_TYPES or len(env.target_objects) == 0:
        return False
    main_object, target = env.main_objects[0], env.target_objects[0]
    placed = bool(main_object.states[og.object_states.Inside].get_value(target)
                  or main_object.states[og.object_states.OnTop].get_value(target))
    # A bidirectional task (see RealmEnvironmentDynamic) is placed either way round, so releasing
    # the target onto the main object is a completed placement, not a dropped object.
    if not placed and getattr(env, "bidirectional", False):
        placed = bool(target.states[og.object_states.OnTop].get_value(main_object))
    return placed


class RenderSchedule:
    """Which control steps render, in render-on-demand mode.

    Inference only runs at action-chunk boundaries, so the cameras only have to be rendered on the
    step whose observation feeds the next inference; every other control step runs physics only. OG
    3.9.1 provides this natively through the `og.sim.render_on_step()` context manager, and
    `og.sim.step()` with rendering off still runs the full physics substeps plus
    `_non_physics_step()`. Before 3.9.1 it needed OG-lite (`gm.RENDER_ON_STEP` + `env.render_obs()`).

    `max_render_interval` bounds how far the renderer may lag physics. Letting it drift arbitrarily
    far was a source of instability on the pre-3.9.1 OG-lite path.

    `n_pre_obs_renders` is how many render passes a rendering step makes before its observation is
    read. The extra passes flush the pipeline: after a run of blind steps the scene has moved, and
    one render() does not fully propagate that before the sensors are read -- OmniGibson's own
    `Simulator.step()` notes that a stage change "will take two `_sim_context.step(render=True)` for
    the result to propagate to the rendering", so two is the documented minimum. 2 means one in-step
    render plus one explicit `og.sim.render()`. It was 3, inherited unmeasured from the pre-3.9.1
    OG-lite path; the third pass cost ~14 ms per render step (1.9% of stepping time) with no
    evidence it was needed. 2 has never been verified as sufficient -- only that 3 was unjustified.

    `obs_is_fresh` says whether the observation the most recent step returned carries a new camera
    frame. It starts True because both entry points warm up with rendering on every step.
    """

    def __init__(self, max_render_interval, n_pre_obs_renders):
        self.max_render_interval = max_render_interval
        self.n_pre_obs_renders = n_pre_obs_renders
        self.obs_is_fresh = True
        self._steps_since_render = 0

    def schedule(self, needs_fresh_obs):
        """Advance the schedule one control step; returns (render, n_render_iterations).

        Renders iff the NEXT iteration needs fresh images -- i.e. an action buffer just ran dry, so
        inference runs next -- or the drift fallback is due.
        """
        render = needs_fresh_obs or (self._steps_since_render + 1) >= self.max_render_interval
        self._steps_since_render = 0 if render else self._steps_since_render + 1
        self.obs_is_fresh = render
        return render, self.n_pre_obs_renders if render else 1


class RolloutMetrics:


    def __init__(self):
        self.qpos = []
        self.actions = []
        self.ee_positions = []
        self.task_progression = 0.0
        self.progression_timestamps = []
        self.terminal_steps = TERMINAL_STEPS
        self.collisions_self = 0
        self.collisions_env = 0
        self.drops = 0
        self.steps = 0
        self._self_collision_active = False
        self._env_collision_active = False
        self._was_grasping = False

    def record_step(self, env, obs, robot_state, gripper_state):
        ee_pos, ee_quat = env.get_ee_pose()
        self.ee_positions.append(ee_pos)

        is_self_collision, is_env_collision = env.check_collisions()
        if is_self_collision and not self._self_collision_active:
            self.collisions_self += 1
        self._self_collision_active = is_self_collision
        if is_env_collision and not self._env_collision_active:
            self.collisions_env += 1
        self._env_collision_active = is_env_collision

        is_grasping = env.check_grasp_condition(obs)
        if self._was_grasping and not is_grasping and not is_placed_on_target(env):
            self.drops += 1
        self._was_grasping = is_grasping

        self.qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))
        return ee_pos, ee_quat

    def record_action(self, action):
        self.actions.append(action)

    def record_progression(self, task_progression, step):
        if task_progression > self.task_progression:
            self.task_progression = task_progression
            self.progression_timestamps.append(step)
        if self.task_progression >= 1.0:
            self.terminal_steps -= 1
        self.steps += 1

    @property
    def is_finished(self):
        return self.terminal_steps <= 0


class PolicyObservation(NamedTuple):
    base_im: Any
    base_im_second: Any
    wrist_im: Any
    robot_state: Any
    gripper_state: Any
    ee_pos: Any
    ee_quat: Any


class Rollout:


    def __init__(self, env, run_id, recorder=None, gripper_inverted=False):
        self.env = env
        self.run_id = run_id
        self.recorder = recorder
        self.gripper_inverted = gripper_inverted
        self.metrics = RolloutMetrics()
        self.action_buffer = Queue()
        self.last_command = None
        self.active = True

    def observe(self, obs, obs_is_fresh):

        base_im, _, base_im_second, _, wrist_im, robot_state, gripper_state = extract_from_obs(
            obs, robot_name=self.env.robot.name)
        ee_pos, ee_quat = self.metrics.record_step(self.env, obs, robot_state, gripper_state)

        if self.recorder is not None and obs_is_fresh:
            self.recorder.add_frame(base_im, wrist_im, base_im_second)

        return PolicyObservation(base_im, base_im_second, wrist_im, robot_state, gripper_state,
                                 ee_pos, ee_quat)

    def act(self, observation, client, horizon):
        if self.action_buffer.empty():
            env = self.env
            chunk = client.infer(
                env.instruction, observation.base_im, observation.base_im_second,
                observation.wrist_im, observation.robot_state, observation.gripper_state,
                use_base_im_second=wants_base_im_second(getattr(env, "task_type", None),
                                                        observation.base_im_second),
                ee_control=env.ee_control,
                cartesian_position=robot_frame_ee_pose(env, observation.ee_pos,
                                                       observation.ee_quat),
            )
            enqueue_action_chunk(self.action_buffer, chunk, horizon)

        action = self.action_buffer.get()
        self.metrics.record_action(action)
        command = action.copy()
        command[-1] = binarize_gripper(action[-1], self.gripper_inverted)
        self.last_command = command
        return command

    def record_progression(self, task_progression, step):
        self.metrics.record_progression(task_progression, step)
        if self.metrics.is_finished:
            self.active = False
        return self.active

    def needs_fresh_obs(self):
        return self.action_buffer.empty()


def joint_space_metrics(qpos):
    joints = qpos[:, :7]
    if len(joints) <= SHORT_TRAJECTORY_SAMPLES:
        return {"joint_vel_var": 0.0, "joint_acc_var": 0.0,
                "joint_jerk": 0.0, "joint_path_length": 0.0}

    velocity = np.diff(joints, axis=0) / CONTROL_DT
    acceleration = np.diff(velocity, axis=0) / CONTROL_DT
    jerk = np.diff(acceleration, axis=0) / CONTROL_DT
    return {
        "joint_vel_var": np.mean(np.var(velocity, axis=0) * len(velocity)),
        "joint_acc_var": np.mean(np.var(acceleration, axis=0) * len(acceleration)),
        "joint_jerk": np.mean(np.linalg.norm(jerk, axis=1)),
        "joint_path_length": np.sum(np.linalg.norm(np.diff(joints, axis=0), axis=1)),
    }


def cartesian_metrics(ee_positions):
    positions = np.stack(ee_positions)
    if len(positions) <= SHORT_TRAJECTORY_SAMPLES:
        return {"cart_jerk": 0.0, "cart_path_length": 0.0}

    velocity = np.diff(positions, axis=0) / CONTROL_DT
    acceleration = np.diff(velocity, axis=0) / CONTROL_DT
    jerk = np.diff(acceleration, axis=0) / CONTROL_DT
    return {
        "cart_jerk": np.mean(np.linalg.norm(jerk, axis=1)),
        "cart_path_length": np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
    }


def first_incomplete_stage(env):
    if env.task_progression is None:
        return "N/A"
    for stage, is_completed in env.task_progression.items():
        if not is_completed:
            return stage
    return "SUCCESS"


def corrected_drops(metrics, env):
    if metrics.task_progression == 1.0 and getattr(env, "task_type", None) in PLACEMENT_TASK_TYPES:
        return max(0, metrics.drops - 1)
    return metrics.drops


def build_result_entry(rollout, task, perturbation, model_type):

    env, metrics = rollout.env, rollout.metrics
    qpos = np.stack(metrics.qpos)
    joint = joint_space_metrics(qpos)
    cartesian = cartesian_metrics(metrics.ee_positions)

    return {
        "run_id": rollout.run_id,
        "task": task,
        "perturbation": perturbation,
        "instruction": env.instruction,
        "model": model_type,
        "real2sim": "Simulated",
        "env": "REALM",
        "task_progression": metrics.task_progression,
        "task_progression_timestamps": metrics.progression_timestamps,
        "stage": first_incomplete_stage(env),
        "binary_SR": 1.0 if metrics.task_progression == 1.0 else 0.0,
        "joint_vel_var": joint["joint_vel_var"],
        "joint_acc_var": joint["joint_acc_var"],
        "joint_jerk": joint["joint_jerk"],
        "joint_path_length": joint["joint_path_length"],
        "cart_path_length": cartesian["cart_path_length"],
        "cart_jerk": cartesian["cart_jerk"],
        "collisions_self": metrics.collisions_self,
        "collisions_env": metrics.collisions_env,
        "object_drops": corrected_drops(metrics, env),
        "qpos": qpos.tolist(),
        "actions": np.stack(metrics.actions).tolist(),
    }


def write_rollout_artifacts(rollout, entry, log_dir, task, perturbation):
    """Write one rollout's video and trajectory parquets, and attach the video bytes to `entry`.

    The layout is frozen -- downstream tooling and tests/test_vector_integrity.py look for
    `reports/{task}_{perturbation}.csv` and `{qpos,actions,videos}/{task}.parquet`. The report
    itself is written by the caller, which rewrites it in full after every rollout.
    """
    metrics = rollout.metrics
    if rollout.recorder is not None:
        video_bytes = rollout.recorder.get_video_bytes()
        entry["video"] = video_bytes
        append_video(log_dir, task, perturbation, rollout.run_id, video_bytes)
        rollout.recorder.cleanup()
    append_trajectory(log_dir, task, perturbation, rollout.run_id,
                      np.stack(metrics.qpos), np.stack(metrics.actions))
