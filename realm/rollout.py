"""What REALM's two evaluation entry points must agree on.

`realm/eval.py` runs rollouts one at a time in a single environment; `realm/vector_eval.py` runs
`num_envs` of them concurrently in one simulator. A metric or a convention that drifts between the
two makes their numbers incomparable, which defeats the point of having a second path, so
everything shared lives here: what a rollout measures, which control steps render, what a result row
looks like, and where artifacts land.

Where the two paths legitimately differ, the difference is an argument to something in this module
rather than a second copy-adapted body -- see `resolve_task`'s `name_includes_config` and
`Rollout`'s `gripper_inverted`.
"""
from queue import Queue
from typing import Any, NamedTuple

import numpy as np
import omnigibson as og
from scipy.spatial.transform import Rotation

from realm.environments.task_progression import DRAWER_TASK_TYPES
from realm.inference import extract_from_obs
from realm.realm_logging import append_trajectory, append_video

CONTROL_HZ = 15.0
CONTROL_DT = 1.0 / CONTROL_HZ

#: Control steps a rollout keeps running after `task_progression` first reaches 1.0.
TERMINAL_STEPS = 15

#: A trajectory of this many samples or fewer reports zero for every derived metric, rather than a
#: velocity/acceleration/jerk taken over one or two points.
SHORT_TRAJECTORY_SAMPLES = 4

#: Task types that end with the object somewhere rather than in the gripper, so a release over the
#: target is a placement rather than a drop.
PLACEMENT_TASK_TYPES = ("put", "stack")


def wants_base_im_second(task_type, base_im_second):
    """Whether this step sends the SECOND exterior camera to the policy instead of the first.

    True only for the drawer tasks, and only when a second camera exists (no ``--multi-view``
    means ``base_im_second`` is None, and the openpi path would crash resizing None rather than
    falling back). CAVEAT: this is the repaired reading of a branch that was dead since the
    project's first commit -- the intent (drawer tasks use camera 2) is clear, the justification
    is not, and it has never been run.
    """
    return task_type in DRAWER_TASK_TYPES and base_im_second is not None


#: Policies trained on DROID emit the gripper as (open, closed) = (1, 0); molmoact emits (0, 1).
#: REALM's gripper controller expects (1, -1).
GRIPPER_OPEN_ABOVE_HALF = ("debug", "openpi", "GR00T", "GR00T_N16", "dreamzero")
GRIPPER_OPEN_BELOW_HALF = ("molmoact",)


def resolve_task(task_id, task_cfg_path, supported_tasks, name_includes_config):
    """Resolve a task selection into (artifact name, task config path).

    The artifact name is the basename every artifact of the run is filed under:
    ``reports/{name}_{perturbation}.csv`` and ``{qpos,actions,videos}/{name}.parquet``.

    Args:
        task_id: index into `supported_tasks`; used only when `task_cfg_path` is None.
        task_cfg_path: explicit config path relative to the config root, e.g.
            "REALM_DROID10/pick_spoon/no_distractors.yaml". Its parent directory names the task.
        supported_tasks: the task table, i.e. `realm.eval.SUPPORTED_TASKS`.
        name_includes_config: append `_{config stem}` for a config file other than default.yaml, so
            two configs of one task do not overwrite each other's artifacts. True on the single-env
            path only -- the vector path has never done this, and turning it on would rename the
            artifacts every vector run has produced so far.
    """
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
    """Whether `model_type` emits the gripper as (0, 1) rather than (1, 0).

    Raises for a policy REALM has no mapping for, rather than defaulting to either convention: a
    wrong sign here inverts the gripper for a whole evaluation, which reads as a policy failure
    rather than a harness bug. `model_type` comes from the CLI verbatim and is never inferred from
    `--model_name`; `realm.inference.InferenceClient` decides what to do with it.
    """
    if model_type in GRIPPER_OPEN_ABOVE_HALF:
        return False
    if model_type in GRIPPER_OPEN_BELOW_HALF:
        return True
    raise NotImplementedError()


def binarize_gripper(gripper_value, inverted=False):
    """Map a policy's continuous gripper output onto REALM's (1, -1) open/closed convention.

    The gripper is the last element of the action the policy already emits, and is overwritten in
    place rather than appended as an extra dimension.
    """
    is_open = gripper_value < 0.5 if inverted else gripper_value > 0.5
    return 1 if is_open else -1


def enqueue_action_chunk(buffer, chunk, horizon):
    """Queue up to `horizon` actions from one policy prediction.

    A chunk is (chunk_length, action_dim); a policy predicting a single action may return it 1D.
    Anything with more dimensions is a bug in the client, and trips the assert below.
    """
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
    """The end-effector pose as (x, y, z, roll, pitch, yaw) in the robot's frame, float32.

    `env.get_ee_pose()` reports the world frame; policies that condition on cartesian state
    (DreamZero) want it relative to the robot base.
    """
    position = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
    quaternion = ee_quat.cpu().numpy() if hasattr(ee_quat, "cpu") else np.array(ee_quat)
    world_pose = np.concatenate([position, Rotation.from_quat(quaternion).as_euler("xyz")])
    return env._world2robot(world_pose).astype(np.float32)


def is_placed_on_target(env):
    """Whether the task's main object is currently on or inside its target.

    Releasing over the target is a placement; releasing anywhere else is a drop. Only placement
    tasks have a target to check.
    """
    if getattr(env, "task_type", None) not in PLACEMENT_TASK_TYPES or len(env.target_objects) == 0:
        return False
    main_object, target = env.main_objects[0], env.target_objects[0]
    return bool(main_object.states[og.object_states.Inside].get_value(target)
                or main_object.states[og.object_states.OnTop].get_value(target))


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
    """Everything one rollout accumulates, step by step.

    Collisions are counted as EDGES: a contact that persists for many steps counts once. A drop is
    a grasp that ended with the object neither on nor inside its target.

    Every quantity here reads physics (object poses, contacts) or proprioception, never camera data,
    and proprioception stays fresh on a blind step -- so metrics are accumulated on every control
    step in both rendering modes. The pre-3.9.1 OG-lite path had to carry values forward across
    blind steps; this one does not.
    """

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
        """Accumulate one control step's proprioception, collisions and grasp transitions.

        Returns the end-effector pose (position, quaternion) it read, which the caller also needs
        for the policy's cartesian observation.
        """
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
        """Record what the policy produced, before REALM's gripper convention is applied to it."""
        self.actions.append(action)

    def record_progression(self, task_progression, step):
        """Fold in this step's task progression, and count the step."""
        if task_progression > self.task_progression:
            self.task_progression = task_progression
            self.progression_timestamps.append(step)
        if self.task_progression >= 1.0:
            self.terminal_steps -= 1
        self.steps += 1

    @property
    def is_finished(self):
        """Whether the task has now been complete for `TERMINAL_STEPS` control steps."""
        return self.terminal_steps <= 0


class PolicyObservation(NamedTuple):
    """What one control step read out of an observation: the policy's inputs, plus the eef pose."""

    base_im: Any
    base_im_second: Any
    wrist_im: Any
    robot_state: Any
    gripper_state: Any
    ee_pos: Any
    ee_quat: Any


class Rollout:
    """One rollout in flight: its action buffer, its metrics and its video recorder.

    The single-env path holds one of these and steps it to completion; the vector path holds
    `num_envs` and steps them together. `active` goes False once the rollout is over -- which the
    vector path needs, because `og.sim.step()` advances every scene regardless and a finished member
    cannot simply stop.

    `gripper_inverted` selects the policy's gripper convention; see `binarize_gripper`.
    """

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
        """Take this control step's metrics and video frame out of `obs`.

        `obs_is_fresh` says whether `obs` carries a newly rendered camera frame. Under
        render-on-demand it does not on every step, and recording the blind steps in between would
        pad the mp4 with duplicates of the last rendered frame.
        """
        base_im, _, base_im_second, _, wrist_im, robot_state, gripper_state = extract_from_obs(
            obs, robot_name=self.env.robot.name)
        ee_pos, ee_quat = self.metrics.record_step(self.env, obs, robot_state, gripper_state)

        if self.recorder is not None and obs_is_fresh:
            self.recorder.add_frame(base_im, wrist_im, base_im_second)

        return PolicyObservation(base_im, base_im_second, wrist_im, robot_state, gripper_state,
                                 ee_pos, ee_quat)

    def act(self, observation, client, horizon):
        """Pop the next action, running inference first if the action buffer ran dry.

        Returns the command to send to the simulator: the policy's action with REALM's gripper
        convention applied to its last element.
        """
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
        """Fold in this step's task progression. Returns whether the rollout is still active."""
        self.metrics.record_progression(task_progression, step)
        if self.metrics.is_finished:
            self.active = False
        return self.active

    def needs_fresh_obs(self):
        """Whether this rollout's next step feeds inference, and so needs a rendered observation."""
        return self.action_buffer.empty()


def joint_space_metrics(qpos):
    """Smoothness and path metrics over the arm's joint trajectory.

    `qpos` is the (N, 8) stack of seven arm joints plus the gripper state; only the arm joints
    enter these.
    """
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
    """Smoothness and path metrics over the end-effector's world-frame path."""
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
    """The first task stage still unmet; "SUCCESS" if none are, "N/A" if the task has no rubric."""
    if env.task_progression is None:
        return "N/A"
    for stage, is_completed in env.task_progression.items():
        if not is_completed:
            return stage
    return "SUCCESS"


def corrected_drops(metrics, env):
    """Drop count, less the release that completed a placement task.

    A completed put/stack ends in a release, which `RolloutMetrics.record_step` may already have
    counted as a drop. Floored at zero.
    """
    if metrics.task_progression == 1.0 and getattr(env, "task_type", None) in PLACEMENT_TASK_TYPES:
        return max(0, metrics.drops - 1)
    return metrics.drops


def build_result_entry(rollout, task, perturbation, model_type):
    """One row of the run report, plus the trajectory arrays the parquets carry.

    Key order is the CSV column order: `realm_logging.save_results` takes its fieldnames from this
    dict, and downstream tooling reads the report by column.
    """
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
