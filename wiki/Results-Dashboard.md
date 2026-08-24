# Results dashboard

The REALM results dashboard is a local Streamlit application for browsing completed runs, comparing
models, checking matrix completeness, inspecting videos, and exporting analysis. It reads existing
artifacts only; it does not launch evaluations or modify report CSV files.

## Start the dashboard

From the repository root:

```sh
uv sync --locked
uv run streamlit run tooling/realm_viewer/dashboard.py
```

The default input is the repository's `logs/` directory. Point it at another result tree with:

```sh
REALM_LOGS=/path/to/logs \
  uv run streamlit run tooling/realm_viewer/dashboard.py --server.port 8501
```

Streamlit prints the local URL. The dashboard runs in the host-side uv environment and does not need
the simulation container or a GPU.

## Expected directory layout

The browser recursively discovers a run when a directory contains `reports/`, or contains MP4 or
Parquet video files under `videos/`. Standard evaluation output has this shape:

```text
logs/
└── <experiment>/
    └── <model>/
        └── <run_id>/
            ├── reports/*.csv
            ├── videos/*.{mp4,parquet}
            ├── qpos/
            └── actions/
```

Each selected run's report CSVs are concatenated. The model label is inferred from the second path
component below `REALM_LOGS`, so preserve the standard `<experiment>/<model>/<run_id>` hierarchy when
comparing models.

## Select and filter runs

The sidebar presents a searchable directory tree:

- expand folders to inspect experiments and runs;
- select a folder to recursively select its discovered runs;
- select several run IDs or models to aggregate them in one comparison;
- use task and perturbation filters to limit plots, exported CSV, and videos.

Changing the selected runs resets video pagination. Report and filesystem discovery are cached for a
short interval; after copying fresh results, wait a few seconds or reload the Streamlit page if they
do not appear immediately.

## Experiment completeness

When an experiment-level `metadata.json` is available, **Experiment Status** compares its requested
task IDs, perturbation IDs, and repeats against the selected reports. Missing matrix cells are listed
explicitly. This check is stronger than the presence of a run directory, but it still relies on the
metadata matching the command that produced the artifacts.

Without metadata, the dashboard can visualize reports but cannot prove that the intended sweep is
complete.

## Analysis views

The dashboard derives the following views from the selected report rows:

| Section | What it answers |
|---|---|
| Success rate per task/perturbation | Which skills or perturbations change binary completion? |
| Task progression | How far does the policy advance when it does not fully succeed? |
| Failure-stage frequency | Which first incomplete rubric stage explains failures? |
| Time to completion | How quickly do successful rollouts finish? |
| Stage timesteps | When are progression milestones reached? |
| Motion metrics | How smooth and efficient are joint and Cartesian trajectories? |
| Collisions and drops | Which runs exhibit interaction or reliability failures? |
| Causal insights | Which model/task/perturbation differences pass the dashboard's statistical screens? |

Model colors and marker shapes remain consistent within the current selection. Success-rate plots
use success/failure counts rather than treating a displayed mean as uncertainty-free. Expand
**How this analysis works — methodology & thresholds** before interpreting Causal Insights; those
findings are exploratory diagnostics, not causal identification from a randomized experiment.

`stage` is the first incomplete ladder stage, not the furthest completed stage. A row labeled
`GRASP` therefore means the grasp requirement was not reached. Use `task_progression` to answer how
far a rollout progressed.

## Reports and exports

- **Download CSV** exports the currently filtered aggregate.
- **Generate PDF Report** produces a presentation-oriented summary of the selected unfiltered runs.
- **Aggregated Reports** exposes the underlying combined table for direct inspection.

Record the selected paths, filters, repository commit, and input artifact checksum when using an
export in a publication. The dashboard is a view over mutable local logs; the PDF alone is not full
experiment provenance.

## Videos

MP4 videos are displayed six per page and follow the active task and perturbation filters. If a run
contains only Parquet-packed video, use **Unpack All Parquet Videos** to create viewable MP4 files in
the run's video directory. This is the one dashboard operation that writes derived files; it does not
alter report data or the original Parquet file.

Render-on-demand is enabled by default during evaluation, so videos may contain roughly one frame per
action chunk. Sparse playback is not evidence that the dashboard dropped frames. See
[Logs, outputs and the viewer](Logs-Outputs-and-Viewer).

## Interpretation safeguards

- Do not combine runs with different task definitions, renderer settings, robot configurations, or
  benchmark versions merely because their columns align.
- Inspect `instruction` when evaluating semantic perturbations; a no-op rewrite can otherwise look
  like robustness.
- Check for identical `task_progression_timestamps` across vector members in old artifacts; this is
  the signature of the historical shared-rubric bug.
- Treat missing values and missing matrix cells separately from failures.
- The task and perturbation filter lists currently target the canonical REALM_DROID10 benchmark.
  Reports from other task families can still be loaded, but unsupported names will not appear as
  first-class sidebar filters without extending `dashboard_utils.py`.

## Troubleshooting

- **No experiments appear:** point `REALM_LOGS` at the directory above the experiment folders and
  verify each run contains `reports/` or supported videos.
- **Plots are empty:** select at least one discovered run and clear task/perturbation filters that do
  not match its CSV values.
- **Completeness is unavailable:** add or recover the experiment's `metadata.json`; reports alone do
  not encode the intended full matrix.
- **Videos do not appear:** unpack Parquet videos, or verify MP4 filenames contain the task and
  perturbation tokens used by the filters.
- **Fresh files are missing:** reload after the short Streamlit cache interval.

## See also

- [Logs, outputs and the viewer](Logs-Outputs-and-Viewer)
- [Running evaluations](Running-Evaluations)
- [Tasks and perturbations](Tasks-and-Perturbations)
