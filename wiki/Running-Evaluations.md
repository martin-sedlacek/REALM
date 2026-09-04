# Running evaluations

The complete flag surface, and what the knobs actually do.

## The two entry points

| Script | Use |
|---|---|
| `examples/02_evaluate.py` | one environment per process |
| `examples/04_vector_evaluate.py` | N environments in one process, rollouts run in waves |

There is also `examples/01_pi0_eval.py`, which takes **no flags at all** — it is a hardcoded
demonstration (task 1, no perturbation, one repeat) and silently ignores anything you pass it.

> `realm/eval.py` is a **library module**. It has no `__main__` and no argument parser. A script
> invoking `python realm/eval.py --...` is stale.

## Container execution

Everything runs inside the release container. Set `REALM_SIF` and `REALM_DATA_PATH`, then open it
with `./scripts/run_apptainer.sh`; see [Quick start](Quick-Start). On a managed cluster, invoke the
launcher only after entering an allocated GPU node using your site's scheduler instructions.

The release image contains OmniGibson 3.9.1 with REALM's patch set already applied, so no bind is
needed. Binding a host OmniGibson checkout over the installed package is a development workflow for
testing a change without rebuilding, not part of a normal evaluation.

Relevant optional environment variables are:
`REALM_INCREMENTAL_CONTACT_CACHE`, `REALM_PROXIMITY_GATE`, `REALM_GPU_DYNAMICS`,
`OMNIGIBSON_HEADLESS` (defaults to `1`).

> **`REALM_GPU_DYNAMICS=1` segfaults at the first reset.** It is passed through for debugging, not
> because it works.

> **Two container traps:**
> - It uses `apptainer run`, never `exec` — `exec` skips the runscript that activates the conda
>   environment, and you land on a Python with no `omnigibson`.
> - **Never wrap the command in `bash -lc`.** A *login* shell re-sources your host `~/.bashrc`,
>   which can shadow the container's environment. Use `bash -c`, or just call `python` directly.

## Flags — `examples/02_evaluate.py`

**Required:** `--model_name`, `--model_type`, `--port`, `--experiment_name`.

| Flag | Type | Default |
|---|---|---|
| `--task_id` | int | `0` |
| `--perturbation_id` | int | `0` |
| `--task_cfg_path` | str | none — overrides `--task_id` when given |
| `--repeats` | int | `5` |
| `--max_steps` | int | `500` |
| `--horizon` | int | `8` |
| `--model_type` | str | **required** |
| `--model_name` | str | **required** |
| `--port` | int | **required** |
| `--host` | str | `127.0.0.1` |
| `--experiment_name` | str | **required** |
| `--run_id` | str | none |
| `--log_dir` | str | none → `/app/logs` |
| `--robot` | str | `DROID` |
| `--rendering_mode` | str | none → `rt`; one of `pt`, `rt`, `r` |
| `--multi-view` | flag | off — **dash, not underscore** |
| `--resume` | flag | off |
| `--no_record` | flag | off |
| `--no_render` | flag | off |
| `--render_on_demand` / `--no-render_on_demand` | flag | **on** |
| `--robometer` | flag | off — score `task_progression` with a [Robometer](Robometer) server instead of the rubric |
| `--robometer_host` | str | `127.0.0.1` |
| `--robometer_port` | int | `8010` — keep it distinct from `--port` |
| `--robometer_success_threshold` | float | `0.9` — Robometer progress at or above which a rollout is a success |
| `--robometer_frame_size` | int | `256` — longest side of the frames sent to the server |

Note the inconsistent separators: `--multi-view` has a dash, `--no_record` and `--no_render` have
underscores, and the negation of `--render_on_demand` is spelled `--no-render_on_demand`. These are
as-is in the source.

### `examples/04_vector_evaluate.py` — the differences

Same required flags. Adds `--num_envs` (default `4`), `--n_pre_obs_renders` (`2`) and
`--max_render_interval` (`8`). Its `--repeats` defaults to `25` rather than `5`, and its `--log_dir`
defaults to `/logs` rather than `/app/logs`. The `--robometer*` flags are identical on both.

**It has no `--resume` and no `--no_render`.** Do not copy a single-env command line onto it.

## Scoring with Robometer (`--robometer`)

By default `task_progression` is the rubric: the fraction of a task's stages whose predicates hold,
computed from privileged simulator state (`realm/environments/task_progression.py`). `--robometer`
swaps that for a learned estimate from a [Robometer](Robometer) reward model watching the same
exterior camera the policy sees. The rubric is still computed, and kept in a side column.

What changes, concretely:

- **Cadence.** Once per action chunk: at every step whose observation is a fresh frame *and* whose
  action buffer has just run dry, the frames seen so far (downscaled to `--robometer_frame_size`)
  plus the current instruction go to the server in one request. Between queries the last estimate
  stands. The recorded value is the running maximum, as it is for the rubric.
- **Success.** `binary_SR`, the 15-step terminal countdown and the placement drop correction all
  trigger at `task_progression >= --robometer_success_threshold` (default `0.9`), not at exactly
  `1.0`.
- **Report columns.** `scorer`, `success_threshold`, `rubric_task_progression`,
  `robometer_success_prob`, `robometer_queries` and the per-query traces
  (`robometer_query_steps`, `robometer_progress_trace`, `robometer_success_trace`) are appended;
  see [Logging](Logging). `stage` is still the rubric's first incomplete stage.
- **Failure mode.** The scorer is built before Isaac boots and waits for the server's `/health`, so
  a dead server fails the run in seconds. A server that dies mid-run raises on the next query.

The server is a separate process on its own GPU: `./scripts/run_robometer_server.sh`, then point
`--robometer_host` / `--robometer_port` at it. **Robometer-scored and rubric-scored rows are not
comparable.** Give a Robometer run its own `--experiment_name`; `--resume` refuses to append rows
scored one way to a report scored the other.

In the vectorized path the whole wave is scored in one request per step, so cost grows with steps,
not with `num_envs`.

## `--render_on_demand` is on by default, and it costs you video

This is the flag most likely to surprise you. With it on, rendering happens only on the steps whose
observation actually feeds inference — physics runs on the rest. That roughly halves median step
time, which is why it defaults on.

The consequence: **the recorded video drops to roughly one frame per action chunk.** A 300-step
rollout yields on the order of 39 frames instead of 300. If the video matters — for a figure, or for
eyeballing a failure — pass `--no-render_on_demand`.

It also means results are not step-for-step comparable against baselines recorded before it existed.

## Model types

`--model_type` accepts exactly three values that construct:

| Value | Behaviour |
|---|---|
| `openpi` | websocket client to an openpi policy server |
| `dreamzero` | the DreamZero client |
| `debug` | returns a constant action; **no server needed** |

Anything else raises `NotImplementedError` at construction. You may see `GR00T`, `GR00T_N16` and
`molmoact` branches further down the inference path — those objects can never be constructed on this
branch. Older documentation offering `pi0`, `pi0_FAST` or `hamster` does not apply to this branch
either.

> **This has a consequence for reproducing the paper.** The published results table is π₀, π₀-FAST
> and GR00T N1.5. **None of those three can be constructed on this branch** — only `openpi`,
> `dreamzero` and `debug` can. The `openpi` client is the route to a π-family policy, but it is a
> different client from the one those numbers were produced with. If you are trying to reproduce the
> paper specifically rather than evaluate your own policy, follow [Reproducibility](Reproducibility)
> and use `v0.1.1`.

`debug` is what the integrity tests use, and it is the right choice for checking that the simulation
and logging path work before you involve a policy.

### The policy server

`--host` and `--port` are the only wiring. There is no URL flag, no TLS flag, no timeout flag.

The client **blocks forever, retrying every 5 seconds**, if nothing is listening — it does not fail
fast. That is why every batch launcher does a socket preflight and aborts rather than starting an
eval against a dead port. Do the same if you are writing your own launcher.

## Rendering modes

`--rendering_mode` takes `rt` (default), `pt`, or `r`.

| Mode | What it sets |
|---|---|
| `rt` | ray-traced lighting, the default path |
| `pt` | path tracing, with a fixed sample count and the Optix denoiser on |
| `r` | ray-traced lighting with reflections, indirect diffuse, shadows, ambient occlusion and DLSS **frame generation** disabled; translucency enabled; sampled lighting pinned to 1 spp. DLSS itself is not disabled — it is switched to its performance mode |

`r` is the cheap mode and `pt` the expensive one, but **treat any speed multiplier you see quoted for
these as unsourced** — the ones in older docs were never measured. More importantly, `r` changes what
the policy sees, so switching to it for speed is a change to the experiment, not just to its cost.

`gm.ENABLE_HQ_RENDERING` is unconditionally **off**. OmniGibson 3.9.1 asserts that high-quality
isosurface rendering runs at 60 FPS or better; REALM renders at 5–30 Hz, so enabling it aborts at
environment creation. Older documentation claiming it is on for some modes is wrong.

`--no_render` disables rendering entirely and zeroes the camera observations. It cannot be combined
with `--multi-view`; the environment asserts on it.

## Frequencies

The DROID control and rendering frequency is fixed:

| Robot | Sim step / rendering |
|---|---|
| `DROID*` | 15 Hz |

Physics always runs at **120 Hz**. At 15 Hz that is 8 physics substeps per environment step.

`ENABLE_TRANSITION_RULES` is off (it triggers an upstream state bug on collision) and
`ENABLE_OBJECT_STATES` is on, because `push_switch` needs the toggle state.

## What counts as success

Rollouts are scored on a progression ladder, not pass/fail — see
[Tasks and perturbations](Tasks-and-Perturbations). `binary_SR` is 1.0 only when progression reaches
1.0.

Once a rollout reaches full progression, **15 further "settling" steps run regardless**, so the
recorded trajectory extends past the success moment.

## Resume

If a single-environment run is interrupted, pass `--resume` with its existing `--run_id`. The run ID
is the timestamp folder inside the experiment's log directory.

```sh
OMNIGIBSON_HEADLESS=1 python /app/examples/02_evaluate.py \
    ...same arguments as the original run... \
    --run_id 20240101_120000 \
    --resume
```

Keep every other argument identical to the original run. Completed repeats are skipped. The
vectorized entry point does not support `--resume`.

## See also

- [Tasks and perturbations](Tasks-and-Perturbations)
- [Reproducibility](Reproducibility)
- [Logs, outputs and the viewer](Logging)
- [Cluster and parallel runs](Cluster-and-Parallel-Runs)
