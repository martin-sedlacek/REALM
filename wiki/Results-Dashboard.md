# Results Dashboard

The results dashboard reads completed REALM runs. It compares models, checks missing evaluations,
plots metrics and displays rollout videos. It does not run the simulator or change report CSV files.

## Running the dashboard

From the repository root:

```sh
uv sync --locked
REALM_LOGS=/path/to/logs \
  uv run streamlit run tooling/realm_viewer/dashboard.py --server.port 8501
```

`REALM_LOGS` defaults to `logs/`. The dashboard runs locally and does not need a GPU or the REALM
container.

## Log layout

The dashboard expects the normal REALM layout:

```text
logs/
└── <experiment>/
    └── <model>/
        └── <run_id>/
            ├── reports/
            ├── videos/
            ├── qpos/
            └── actions/
```

A folder is treated as a run when it contains reports or videos. CSV files from selected runs are
combined into one table. Keep the `<experiment>/<model>/<run_id>` hierarchy because the model name
is read from the path.

## Selecting results

Use the left sidebar to search and select experiments. Selecting a folder selects the runs below it.
Task and perturbation filters apply to plots, videos and CSV export.

If `metadata.json` is present, **Experiment Status** checks that every requested task,
perturbation and repeat is in the reports. Without metadata, the dashboard can show the results but
cannot know whether the experiment is complete.

## Plots

The dashboard includes:

- success rate by task and perturbation;
- task progression;
- failure stage frequency;
- completion time and stage timestamps;
- joint and Cartesian smoothness;
- path length, collisions and drops;
- statistical comparisons under **Causal Insights**.

`stage` is the first incomplete task stage. For example, `GRASP` means the rollout failed to grasp.
Use `task_progression` to see how much of the task was completed.

The Causal Insights section is an analysis helper. Its thresholds are shown in the dashboard. Treat
the output as a signal to inspect, not proof of causality.

## Exporting results

- **Download CSV** saves the filtered table.
- **Generate PDF Report** creates a summary of the selected runs.
- **Aggregated Reports** shows the combined rows directly.

Record the selected runs and filters when using an export in a report or paper.

## Videos

MP4 videos are shown six per page. If a run contains Parquet videos, use **Unpack All Parquet
Videos** to create MP4 files. The Parquet files are kept unchanged.

Videos can have sparse frames because `--render_on_demand` is enabled during evaluation by default.
See [Logging](Logging) for details.

## Notes

- Do not combine results made with different task configs, robots, render settings or benchmark
  versions.
- Check the `instruction` column when reviewing semantic perturbations.
- Old vectorized reports with identical `task_progression_timestamps` across environments contain
  the shared-progression bug and should not be used for success rates.
- The sidebar task filters currently list the canonical REALM_DROID10 tasks. Other task families can
  be loaded, but their names are not shown as first-class filters.

## See also

- [Logging](Logging)
- [Running evaluations](Running-Evaluations)
