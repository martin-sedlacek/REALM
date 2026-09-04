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
- Success is `task_progression >= success_threshold` (default 0.9) rather than "every rubric stage
  done": a learned estimate rarely reaches 1.0 exactly. `binary_SR`, the TERMINAL_STEPS countdown
  and the placement drop correction all follow that threshold.
- The report gains columns: `scorer`, `success_threshold`, `rubric_task_progression` (the rubric's
  own max, for the same rollout), `robometer_success_prob` (the model's success head at the last
  query, blank if the checkpoint has none), `robometer_queries`, and the per-query trajectory
  `robometer_query_steps` / `robometer_progress_trace` / `robometer_success_trace` (raw outputs,
  not the running max, so the estimate can be inspected step by step).
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
from realm.rollout import wants_base_im_second

#: Progress at or above which a Robometer-scored rollout counts as a success.
DEFAULT_ROBOMETER_SUCCESS_THRESHOLD = 0.9
#: Longest side, in pixels, of the frames sent to the server. The 1280x720 exterior render is far
#: above what the model's vision encoder keeps, and a whole clip travels with every query.
DEFAULT_ROBOMETER_FRAME_SIZE = 256
DEFAULT_ROBOMETER_PORT = 8010


class RubricScorer:
    """Pass the rubric's progression straight through. The default, and the benchmark's scorer."""

    name = "rubric"
    success_threshold = 1.0

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


def exterior_frame(env, obs):
    """The exterior camera image the policy is shown for this env -- the second camera on the drawer
    tasks under --multi-view, the first otherwise -- so the scorer watches the same view."""
    base_im, _, base_im_second, _, _, _, _ = extract_from_obs(obs, robot_name=env.robot.name)
    if wants_base_im_second(getattr(env, "task_type", None), base_im_second):
        return base_im_second
    return base_im


class RobometerScorer:
    """Task progression from a Robometer server instead of the rubric.

    `client` is anything with robometer_client.RobometerClient's `progress_batch` and
    `wait_until_healthy`; the default is a real client, and the constructor waits for the server
    to report healthy so an eval does not boot Isaac only to fail on its first query.
    """

    name = "robometer"

    def __init__(self, host="127.0.0.1", port=DEFAULT_ROBOMETER_PORT, *,
                 success_threshold=DEFAULT_ROBOMETER_SUCCESS_THRESHOLD,
                 frame_size=DEFAULT_ROBOMETER_FRAME_SIZE, timeout_s=120.0, wait_s=600.0,
                 client=None):
        if client is None:
            from robometer_client import RobometerClient
            client = RobometerClient(host=host, port=port, timeout_s=timeout_s)
        self.client = client
        self.success_threshold = float(success_threshold)
        self.frame_size = int(frame_size)
        self.client.wait_until_healthy(timeout_s=wait_s)

    def score(self, items):
        """Query the server for every rollout whose post-step observation is a fresh frame AND whose
        action buffer has just run dry -- i.e. once per action chunk, on the frame the policy is
        about to see. Under --render_on_demand those are the only fresh frames anyway; under
        --no-render_on_demand this keeps the query count at one per chunk instead of one per step,
        each carrying a clip that grows with the rollout.

        Rollouts not queried this step return their latest estimate, which
        `record_progression`'s running max then leaves unchanged. Everything is done in one
        round trip so the vector path pays one request per wave step, not one per member.
        """
        queried, clips, tasks = [], [], []
        for rollout, rubric_progression, obs, obs_is_fresh in items:
            metrics = rollout.metrics
            metrics.rubric_task_progression = max(metrics.rubric_task_progression,
                                                  float(rubric_progression))
            if obs_is_fresh and rollout.needs_fresh_obs():
                rollout.clip.append(downscale_frame(exterior_frame(rollout.env, obs),
                                                    self.frame_size))
                queried.append(rollout)
                clips.append(np.stack(rollout.clip))
                tasks.append(rollout.env.instruction)

        if queried:
            for rollout, result in zip(queried, self.client.progress_batch(clips, tasks)):
                metrics = rollout.metrics
                metrics.scorer_progress = result.reward
                metrics.scorer_success_prob = result.success_prob
                metrics.scorer_queries += 1
                # metrics.steps counts record_progression calls so far, i.e. the current step index.
                metrics.scorer_query_steps.append(metrics.steps)
                metrics.scorer_progress_trace.append(result.reward)
                metrics.scorer_success_trace.append(result.success_prob)

        return [rollout.metrics.scorer_progress for rollout, _, _, _ in items]

    def result_columns(self, rollout):
        metrics = rollout.metrics
        return {
            "scorer": self.name,
            "success_threshold": self.success_threshold,
            "rubric_task_progression": metrics.rubric_task_progression,
            "robometer_success_prob": ("" if metrics.scorer_success_prob is None
                                       else metrics.scorer_success_prob),
            "robometer_queries": metrics.scorer_queries,
            "robometer_query_steps": list(metrics.scorer_query_steps),
            "robometer_progress_trace": list(metrics.scorer_progress_trace),
            "robometer_success_trace": list(metrics.scorer_success_trace),
        }
