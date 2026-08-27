# Logging

Each run writes into `<log_dir>/<experiment_name>/<model_name>/<run_id>/`, with four kinds of artifact:

```
reports/    per-cell CSV -- the metrics
qpos/       joint trajectories
actions/    the actions the policy emitted
videos/     recorded rollouts
```

`--log_dir` defaults to `/app/logs` for the single-env script and `/logs` for the vectorized one.
Both are container paths; on the host they land wherever the log directory is bound from.

Files are keyed `<task>_<perturbation>` — for example `pick_spoon_VSB-NOBJ`.

## The experiment report schema

These are plain CSV files with the structure of one row per rollout and the following columns, in order:

| Column | Meaning                                                                                                           |
|---|-------------------------------------------------------------------------------------------------------------------|
| `run_id` | numerical identifier within the experiment                                                                        |
| `task` | task identifier of the run                                                                                        |
| `perturbation` | perturbation identifier of the run                                                                                |
| `instruction` | the instruction actually given — **not** always the task's default, since the semantic perturbations rewrite it   |
| `model` | the model type, which dictates dsitinct server/client pipelines - e.g., openpi, dreamzer, etc.                    |
| `real2sim` | always `Simulated`                                                                                                |
| `env` | always `REALM`                                                                                                    |
| `task_progression` | score betweem 0 and 1 based on the fraction of equally weighted task stages completed                             |
| `task_progression_timestamps` | when each stage was reached                                                                                       |
| `stage` | the **stage where the rollout terminated** — this will either be the failure stage or a `SUCCESS` if all completed |
| `binary_SR` | 1.0 if `task_progression` was 1.0, else 0.0                                                                       |
| `joint_vel_var`, `joint_acc_var`, `joint_jerk` | joint-space smoothness, lwoer is better                                                                           |
| `joint_path_length` | total joint-space distance travelled                                                                              |
| `cart_path_length`, `cart_jerk` | total Cartesian space distance travelled                                                                          |
| `collisions_self` | # of robot self-collisions                                                                                        |
| `collisions_env` | # of collisions with the environment                                                                              |
| `object_drops` | # of time objects got dropped                                                                                     |

`object_drops` has one adjustment worth knowing: a successful `put` or `stack` necessarily involves
releasing the object, so one drop is subtracted when the task succeeded.

`instruction` is the column to check first when a semantic perturbation looks like it did nothing —
if it matches the task default, the perturbation was a no-op for that rollout.

## Results Dashboard

The repository includes a lightweight results dashboard, which you can run with:

```sh
uv sync --locked
REALM_LOGS=/path/to/your/logs \
  uv run streamlit run tooling/realm_viewer/dashboard.py --server.port 8501
```

`REALM_LOGS` defaults to `logs/`. The dashboard runs locally and does not need a GPU or the REALM container.

Use the sidebar to search and select experiments. Selecting a folder selects the runs below it.
Task and perturbation filters apply to the plots, videos and exported CSV.

The dashboard shows success rate, task progression, failure stages, completion time, smoothness,
path length, collisions and drops. The Causal Insights section provides statistical comparisons.
Its output is a signal to inspect, not proof of causality.

If a run contains `metadata.json`, Experiment Status checks that every requested task,
perturbation and repeat is present. Without metadata, the dashboard can display the results but
cannot determine whether the experiment is complete.

You can download the filtered rows as CSV, generate a PDF report, or inspect the aggregated table.
Videos are shown six per page. **Unpack All Parquet Videos** creates MP4 files without changing the
original Parquet files.

Do not combine runs made with different task configs, robots, render settings or benchmark versions.

## Video Logging

### ⚠ Saved videos have sparse frames by default

`--render_on_demand` option skips the rendering step unless the RGB observations are being sent to the policy server for inference. 
This option is **enabled by default**, but can be disabled manually.

`--no-render_on_demand` flag can be passed to render every frame at the expense of simulation speed.

### ⚠ Disk Space Bottleneck

`--no_record` flag does not save any videos, which can help you recover a noticeable speedup if you run on systems where 
disk writes are a bottleneck, such as lustre - in our own experience.
