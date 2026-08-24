# Logs, outputs and the viewer

## What a run produces

A run writes into `<log_dir>/<experiment_name>/<model_name>/<run_id>/`, with four kinds of artifact:

```
reports/    per-cell CSV -- the metrics
qpos/       joint trajectories
actions/    the actions the policy emitted
videos/     recorded rollouts
```

`--log_dir` defaults to `/app/logs` for the single-env script and `/logs` for the vectorized one.
Both are container paths; on the host they land wherever the log directory is bound from.

Files are keyed `<task>_<perturbation>` — for example `pick_spoon_VSB-NOBJ`. That naming is what the
integrity tests and the viewer rely on, so do not rename them.

## The report schema

One row per rollout. Columns, in order:

| Column | Meaning |
|---|---|
| `run_id` | the run this rollout belongs to |
| `task` | task identifier |
| `perturbation` | perturbation identifier |
| `instruction` | the instruction actually given — **not** always the task's default, since the semantic perturbations rewrite it |
| `model` | the model type |
| `real2sim` | always `Simulated` here |
| `env` | always `REALM` |
| `task_progression` | fraction of the ladder reached, 0.0–1.0 |
| `task_progression_timestamps` | when each stage was reached |
| `stage` | the **first ladder stage not completed** — i.e. where the rollout stopped — or `SUCCESS` if all completed, or `N/A` if no ladder. **It is the failure point, not the furthest stage reached.** |
| `binary_SR` | 1.0 if `task_progression` reached 1.0, else 0.0 |
| `joint_vel_var`, `joint_acc_var`, `joint_jerk` | joint-space smoothness |
| `joint_path_length` | total joint-space distance travelled |
| `cart_path_length`, `cart_jerk` | the same in Cartesian space |
| `collisions_self` | self-collision events |
| `collisions_env` | collisions with the environment |
| `object_drops` | dropped objects |

`object_drops` has one adjustment worth knowing: a successful `put` or `stack` necessarily involves
releasing the object, so one drop is subtracted when the task succeeded.

`instruction` is the column to check first when a semantic perturbation looks like it did nothing —
if it matches the task default, the perturbation was a no-op for that rollout.

> **Read `stage` in the right direction.** It is produced by walking the ladder and stopping at the
> first incomplete stage, so `stage = GRASP` means the rollout **did not** grasp. Use
> `task_progression` for "how far did it get"; use `stage` for "where did it fail". Getting this
> backwards inverts every failure-mode breakdown built on it.

## Reading results

The reports are plain CSV; anything that reads CSV will do.

There is also a Streamlit dashboard, in a **separate repository** —
<https://github.com/martin-sedlacek/REALM_toolkit>:

```sh
git clone https://github.com/martin-sedlacek/REALM_toolkit
cd REALM_toolkit
uv sync
REALM_LOGS=/path/to/your/logs uv run streamlit run realm_viewer/dashboard.py
```

> **The viewer wants `REALM_LOGS`, which is exactly the variable the run harness deliberately
> ignores** (it reads `REALM_LOGS` instead — see [Installation](Installation)). Both are
> correct in their own repository; they are different tools that happen to have collided on a name.
> Set `REALM_LOGS` for the viewer and do not expect it to affect a run.

## ⚠ Vector runs: check before trusting `binary_SR`

Vectorized results recorded **before the per-environment rubric fix** are contaminated. The
environments in a wave shared one progression dictionary, which made progression an **OR across
members** — so success by any one environment was recorded for all of them, inflating success rate.

**The tell is `task_progression_timestamps` being identical across members of a wave.** If you see
that, `task_progression`, `stage` and `binary_SR` in that file are invalid. The `qpos/`, `actions/`
and `videos/` artifacts from the same runs are fine.

A specific published-looking figure from that period — a success rate of 0.960 over 25 rollouts — is
**explicitly retracted** in the project's own vector-env notes. Do not cite it.

## ⚠ Videos are sparse by default

`--render_on_demand` is **on by default**, which means the simulator only renders on steps whose
observation feeds inference. Recorded video therefore contains roughly **one frame per action
chunk** — on the order of 39 frames from a 300-step rollout, not 300.

If a video looks like a stuttering slideshow, that is why, and it is expected. Pass
`--no-render_on_demand` when the video itself matters. Older documentation blames `--no_record` for
missing video; that is no longer the common cause.

## Disk

Videos dominate. Rough guidance rather than a promise: a full 10 × 16 sweep at 25 repeats produces a
lot of video, and `--no_record` is the lever if you only need metrics. **Measure it on your own
sweep** before planning capacity — the numbers depend on rendering mode, resolution and rollout
length, and any figure quoted here would be a guess.

## See also

- [Running evaluations](Running-Evaluations)
- [Cluster and parallel runs](Cluster-and-Parallel-Runs)
- [Known issues and gotchas](Known-Issues-and-Gotchas)
