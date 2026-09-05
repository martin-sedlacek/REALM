"""Where a rollout's 0-1 task progression comes from.

Both evaluation paths (realm/eval.py, realm/vector_eval.py) step the environment, get the rubric's
progression back from `env.step()`, and hand a value to `Rollout.record_progression()`. A scorer
sits between those two: it receives the rubric value, the post-step observation and whether that
observation carries a freshly rendered frame, and returns the value to record.

    RubricScorer     the default. Returns the rubric value untouched. Zero behavioural change: the
                     default path records exactly what it always did, in the same order, and
                     appends no columns.
    RobometerScorer  `--robometer`. Sends the exterior frames the policy has seen so far, plus the
                     instruction, to a Robometer reward-model server and records ITS progress
                     estimate instead. The rubric is still computed by `env.step()` (it is not
                     optional there) and is kept alongside for comparison.

WHAT `--robometer` CHANGES, precisely
------------------------------------
- `task_progression` is Robometer's running-max progress estimate, sampled at every action-chunk
  boundary that has a fresh frame (see `RobometerScorer.score`), not the rubric's stage fraction.
  Each query sends, per scored camera (default: the exterior view the policy sees AND the wrist
  camera), the rollout's frames so far, downscaled and linspace-subsampled to `max_frames` (16,
  the model's training clip length; first and current frame always kept). The cameras' raw
  scores are fused (`max` by default) into one raw score per query.
- The raw score is CALIBRATED per task before it is recorded: calibrated = clip((raw - floor) /
  (ceiling - floor), 0, 1) with floor/ceiling from realm/config/robometer_calibration.yaml, bound
  by `configure(task)`. Raw Robometer scores plateau at task-dependent levels (a finished banana
  placement reads ~0.7, a finished block-in-bowl ~0.8, neither ever 1.0), so without this the 0-1
  scale means nothing across tasks. A task without an entry is passed through raw and warned about.
- Success is calibrated `task_progression >= success_threshold` (default 1.0, i.e. raw reached the
  task's ceiling). `binary_SR`, the TERMINAL_STEPS countdown and the placement drop correction all
  follow that threshold.
- The report gains columns: `scorer`, `success_threshold`, `rubric_task_progression` (the rubric's
  own max, for the same rollout), `robometer_success_prob` (the model's success head at the last
  query, blank if the checkpoint has none), `robometer_queries`, the per-query trajectory
  `robometer_query_steps` / `robometer_progress_trace` / `robometer_success_trace` (RAW outputs,
  not calibrated and not the running max, so the calibration can be re-fitted from reports), the
  camera setup (`robometer_cameras`, `robometer_fusion`, one `robometer_progress_trace_<camera>`
  per camera with that camera's unfused raw scores) and the calibration used: `robometer_raw_max`,
  `robometer_floor`, `robometer_ceiling`, `robometer_calibrated`.
- `stage` still comes from the rubric -- it is the first incomplete rubric stage, and there is no
  Robometer equivalent.

Rubric-scored and Robometer-scored rows are NOT comparable and must not be mixed in one report;
`realm.eval.evaluate` refuses to `--resume` across the two. Use a distinct `--experiment_name`.

The scorer never touches the simulator and draws no random numbers, so it cannot move a
rubric-scored number: the rubric path does not import it, and under `--robometer` the only
observable change to the environment is wall-clock time.
"""
import numpy as np
from PIL import Image

from realm.inference import extract_from_obs
from realm.robometer_calibration import (
    DEFAULT_CALIBRATION_PATH,
    TaskCalibration,
    calibration_for,
    load_calibration,
)
from realm.rollout import wants_base_im_second

#: Calibrated progress at or above which a Robometer-scored rollout counts as a success. Calibrated
#: 1.0 means "raw score reached the task's ceiling" (realm/config/robometer_calibration.yaml), so the
#: threshold is the per-task ceiling itself; lower it to accept a fraction of the ceiling.
DEFAULT_ROBOMETER_SUCCESS_THRESHOLD = 1.0
#: Longest side, in pixels, of the frames sent to the server. The 1280x720 exterior render is far
#: above what the model's vision encoder keeps, and a whole clip travels with every query.
DEFAULT_ROBOMETER_FRAME_SIZE = 256
#: Frames per query. Robometer-4B was trained on clips linspace-subsampled to 16 frames, and the
#: server's npy endpoint does NOT subsample -- every frame sent goes through the vision tower. A
#: causal clip that keeps every chunk-boundary frame reaches 60+ frames on a 500-step rollout,
#: which is both off-distribution and, measured 2026-09-05, a CUDA OOM on a shared 32 GB card. The
#: first and the current frame are always kept, so the reward is still the current frame's.
DEFAULT_ROBOMETER_MAX_FRAMES = 16
DEFAULT_ROBOMETER_PORT = 8010
#: Cameras scored per query and how their raw scores are fused into one. "base" is the exterior
#: view the policy is shown (the second exterior camera on the drawer tasks under --multi-view),
#: "wrist" the gripper camera. The wrist sees the grasp and the object up close, which the exterior
#: view often occludes, so both are scored by default; `max` takes the more confident view.
DEFAULT_ROBOMETER_CAMERAS = ("base", "wrist")
DEFAULT_ROBOMETER_FUSION = "max"
FUSIONS = {"max": max, "min": min, "mean": lambda xs: sum(xs) / len(xs)}


class RubricScorer:
    """Pass the rubric's progression straight through. The default, and the benchmark's scorer."""

    name = "rubric"
    success_threshold = 1.0

    def configure(self, task):
        """Called once per evaluation with the resolved task name; the rubric needs nothing."""

    def score(self, items):
        """items: [(rollout, rubric_progression, obs, obs_is_fresh)] -> [progression to record]."""
        return [rubric_progression for _, rubric_progression, _, _ in items]

    def result_columns(self, rollout):
        return {}


def downscale_frame(frame, longest_side):
    """Shrink an HxWx3 uint8 frame so its longer side is `longest_side`; never upscales."""
    frame = np.ascontiguousarray(np.asarray(frame)[..., :3])
    h, w = frame.shape[:2]
    scale = longest_side / max(h, w)
    if scale >= 1.0:
        return frame.astype(np.uint8, copy=False)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(frame.astype(np.uint8, copy=False)).resize(size, Image.BILINEAR))


def camera_frames(env, obs, cameras):
    """{camera: HxWx3 frame} for the requested camera names. "base" is the exterior image the
    policy is shown for this env (the second camera on the drawer tasks under --multi-view, the
    first otherwise), "wrist" the gripper camera; anything else raises."""
    base_im, _, base_im_second, _, wrist_im, _, _ = extract_from_obs(obs, robot_name=env.robot.name)
    if wants_base_im_second(getattr(env, "task_type", None), base_im_second):
        base_im = base_im_second
    available = {"base": base_im, "wrist": wrist_im}
    unknown = [c for c in cameras if c not in available]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; choose from {sorted(available)}")
    return {c: available[c] for c in cameras}


def exterior_frame(env, obs):
    """The exterior camera image the policy is shown for this env."""
    return camera_frames(env, obs, ("base",))["base"]


class RobometerScorer:
    """Task progression from a Robometer server instead of the rubric.

    `client` is anything with robometer_client.RobometerClient's `progress_batch` and
    `wait_until_healthy`; the default is a real client, and the constructor waits for the server
    to report healthy so an eval does not boot Isaac only to fail on its first query.
    """

    name = "robometer"

    def __init__(self, host="127.0.0.1", port=DEFAULT_ROBOMETER_PORT, *,
                 success_threshold=DEFAULT_ROBOMETER_SUCCESS_THRESHOLD,
                 frame_size=DEFAULT_ROBOMETER_FRAME_SIZE, max_frames=DEFAULT_ROBOMETER_MAX_FRAMES,
                 cameras=DEFAULT_ROBOMETER_CAMERAS, fusion=DEFAULT_ROBOMETER_FUSION,
                 calibration=DEFAULT_CALIBRATION_PATH, timeout_s=120.0, wait_s=600.0, client=None):
        """`calibration` is the per-task raw->0-1 table: a path to a YAML in the format of
        realm/config/robometer_calibration.yaml (the default), an already-loaded dict, or {} for
        no calibration at all (raw scores, success unreachable). The task is bound by configure().
        `cameras` are scored each query and their raw scores combined by `fusion` (max/min/mean)
        BEFORE calibration; the calibration table must have been fitted with the same setting."""
        if client is None:
            from robometer_client import RobometerClient
            client = RobometerClient(host=host, port=port, timeout_s=timeout_s)
        self.client = client
        self.success_threshold = float(success_threshold)
        self.frame_size = int(frame_size)
        self.max_frames = int(max_frames)
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("at least one camera is required")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(FUSIONS)}, got {fusion!r}")
        self.fusion = fusion
        self._fuse = FUSIONS[fusion]
        self.calibration_table = (load_calibration(calibration) if isinstance(calibration, str)
                                  else dict(calibration or {}))
        self.task_calibration = TaskCalibration("<unconfigured>", 0.0, 1.0, False)
        self.client.wait_until_healthy(timeout_s=wait_s)

    def configure(self, task):
        """Bind the evaluation's task so raw scores are mapped through its calibration entry."""
        self.task_calibration = calibration_for(task, self.calibration_table)
        cal = self.task_calibration
        if cal.calibrated:
            print(f"[robometer] task {task!r}: calibration floor={cal.floor:g} ceiling={cal.ceiling:g}"
                  f"{'' if cal.task == task else f' (entry {cal.task!r})'}; success at calibrated "
                  f">= {self.success_threshold:g}", flush=True)
        else:
            print(f"[robometer] WARNING task {task!r} has no entry in the calibration table: "
                  f"task_progression is the RAW score and success (>= {self.success_threshold:g}) "
                  f"is effectively unreachable. Add it to realm/config/robometer_calibration.yaml.",
                  flush=True)

    def score(self, items):
        """Query the server for every rollout whose post-step observation is a fresh frame AND whose
        action buffer has just run dry -- i.e. once per action chunk, on the frame the policy is
        about to see. Under --render_on_demand those are the only fresh frames anyway; under
        --no-render_on_demand this keeps the query count at one per chunk instead of one per step,
        each carrying a clip that grows with the rollout.

        Rollouts not queried this step return their latest estimate, which
        `record_progression`'s running max then leaves unchanged. Everything is done in one
        round trip so the vector path pays one request per wave step, not one per member.

        The clips a rollout accumulates (one per camera) keep every chunk-boundary frame; what is
        SENT is each clip linspace-subsampled to `max_frames` with the first and current frame kept
        (robometer_client.subsample_frames), so a query stays the size the model was trained on
        however long the rollout runs. All cameras of all rollouts due a query go in ONE request,
        ordered rollout-major, camera-minor; the per-camera raw scores are fused by `fusion` and the
        fused raw is what gets calibrated and recorded.
        """
        from robometer_client import subsample_frames

        queried, clips, tasks = [], [], []
        for rollout, rubric_progression, obs, obs_is_fresh in items:
            metrics = rollout.metrics
            metrics.rubric_task_progression = max(metrics.rubric_task_progression,
                                                  float(rubric_progression))
            if obs_is_fresh and rollout.needs_fresh_obs():
                frames = camera_frames(rollout.env, obs, self.cameras)
                for cam in self.cameras:
                    rollout.clips.setdefault(cam, []).append(downscale_frame(frames[cam], self.frame_size))
                    clips.append(subsample_frames(np.stack(rollout.clips[cam]), self.max_frames))
                    tasks.append(rollout.env.instruction)
                queried.append(rollout)

        if queried:
            results = self.client.progress_batch(clips, tasks)
            n_cam = len(self.cameras)
            for i, rollout in enumerate(queried):
                per_cam = dict(zip(self.cameras, results[i * n_cam:(i + 1) * n_cam]))
                raw = self._fuse([r.reward for r in per_cam.values()])
                succ = [r.success_prob for r in per_cam.values()]
                fused_succ = None if any(s is None for s in succ) else self._fuse(succ)
                metrics = rollout.metrics
                # The recorded progression is CALIBRATED (per-task fused raw -> 0-1); the traces
                # keep the raw scores so the calibration can be re-fitted from reports later.
                metrics.scorer_progress = self.task_calibration.apply(raw)
                metrics.scorer_success_prob = fused_succ
                metrics.scorer_queries += 1
                # metrics.steps counts record_progression calls so far, i.e. the current step index.
                metrics.scorer_query_steps.append(metrics.steps)
                metrics.scorer_progress_trace.append(raw)
                metrics.scorer_success_trace.append(fused_succ)
                for cam, r in per_cam.items():
                    metrics.scorer_camera_traces.setdefault(cam, []).append(r.reward)

        return [rollout.metrics.scorer_progress for rollout, _, _, _ in items]

    def result_columns(self, rollout):
        metrics = rollout.metrics
        cols = {
            "scorer": self.name,
            "success_threshold": self.success_threshold,
            "robometer_cameras": "+".join(self.cameras),
            "robometer_fusion": self.fusion,
            "rubric_task_progression": metrics.rubric_task_progression,
            "robometer_success_prob": ("" if metrics.scorer_success_prob is None
                                       else metrics.scorer_success_prob),
            "robometer_queries": metrics.scorer_queries,
            "robometer_query_steps": list(metrics.scorer_query_steps),
            "robometer_progress_trace": list(metrics.scorer_progress_trace),
            "robometer_success_trace": list(metrics.scorer_success_trace),
            "robometer_raw_max": (max(metrics.scorer_progress_trace)
                                  if metrics.scorer_progress_trace else 0.0),
            "robometer_floor": self.task_calibration.floor,
            "robometer_ceiling": self.task_calibration.ceiling,
            "robometer_calibrated": self.task_calibration.calibrated,
        }
        for cam in self.cameras:
            cols[f"robometer_progress_trace_{cam}"] = list(metrics.scorer_camera_traces.get(cam, []))
        return cols
