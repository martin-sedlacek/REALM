# Running evaluations

The complete flag surface, and what the knobs actually do.

## The two entry points

| Script | Use |
|---|---|
| `examples/02_evaluate.py` | one environment per process |
| `examples/04_vector_evaluate.py` | N environments in one process, rollouts run in waves |

There is also `examples/01_pi0_eval.py`, which takes **no flags at all** — it is a hardcoded
demonstration (task 1, no perturbation, one repeat) and silently ignores anything you pass it.

> `realm/eval.py` is a **library module**. It has no `__main__` and no argument parser. If you find a
> script invoking `python realm/eval.py --...`, that script is stale — see
> [Known issues](Known-Issues-and-Gotchas).

## `rr` and `MODE`

Everything runs inside the container. `scripts/clara/interactive/rr` puts you there. It takes **no
flags of its own** — everything after `rr` is the in-container command, and configuration is by
environment variable.

**`rr` starts the container wherever it is invoked.** It does not allocate and it does not `srun`, so
it has to be reached through one:

```sh
MODE=stock srun --jobid=<ID> --overlap \
  ./scripts/clara/interactive/rr python -u examples/02_evaluate.py --task_id 0 ...
```

Run bare on a login node you get a container with no GPU. The `go` wrapper does the `srun` for you
and adds logging — see [Cluster and parallel runs](Cluster-and-Parallel-Runs).

`MODE` selects which OmniGibson the run sees. **`stock` is the default.**

| `MODE` | What it binds |
|---|---|
| `stock` | the image's own OmniGibson 3.9.1. Nothing bound. **Default.** |
| `oglite` | the host OG-lite fork bound over the image's package — the whole fork |
| `stockfix` | the image's own OmniGibson, plus individually-bound patched files from the stock-patch directory |

`stockfix` exists because a rebuilt image cannot currently be produced (see
[Installation](Installation)). Both build recipes apply the same patches, so `stockfix` is intended
to behave identically to a rebuilt image — but **check that the stock-patch directory is current**
before relying on it. It is produced by `scripts/clara/interactive/make_stock_patch.sh`, and `rr`
binds whatever it finds there; if the directory is stale, you silently run without the newer fixes.

Other environment variables `rr` passes through, only when you set them:
`REALM_INCREMENTAL_CONTACT_CACHE`, `REALM_PROXIMITY_GATE`, `REALM_GPU_DYNAMICS`,
`OMNIGIBSON_HEADLESS` (defaults to `1`).

> **`REALM_GPU_DYNAMICS=1` segfaults at the first reset.** It is passed through because an
> investigation needed it, not because it works — see
> [Performance and scaling](Performance-and-Scaling).

> **Two container traps, both encoded in `rr`'s own comments:**
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

Note the inconsistent separators: `--multi-view` has a dash, `--no_record` and `--no_render` have
underscores, and the negation of `--render_on_demand` is spelled `--no-render_on_demand`. These are
as-is in the source.

### `examples/04_vector_evaluate.py` — the differences

Same required flags. Adds `--num_envs` (default `4`), `--n_pre_obs_renders` (`2`) and
`--max_render_interval` (`8`). Its `--repeats` defaults to `25` rather than `5`, and its `--log_dir`
defaults to `/logs` rather than `/app/logs`.

**It has no `--resume` and no `--no_render`.** Do not copy a single-env command line onto it.

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
> paper specifically rather than evaluate your own policy, start from the paper's own release rather
> than from this branch, and ask before assuming the two are interchangeable.

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

Set per robot family, with physics fixed:

| Robot | Sim step / rendering |
|---|---|
| `WidowX` | 5 Hz |
| `UR5*` | 30 Hz |
| everything else, including all `DROID*` | 15 Hz |

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

`--run_id <id> --resume` picks an existing run report back up rather than starting over. Useful when
a sweep cell died partway. The sweep drivers rely on this, and additionally skip cells whose outputs
already exist.

## See also

- [Tasks and perturbations](Tasks-and-Perturbations)
- [Robots and configs](Robots-and-Configs)
- [Logs, outputs and the viewer](Logs-Outputs-and-Viewer)
- [Cluster and parallel runs](Cluster-and-Parallel-Runs)
- [Known issues and gotchas](Known-Issues-and-Gotchas)
