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

## Robots

`--robot` names a file in `realm/config/robots/` (without `.yaml`). The values with a RobotDefinition
registered for OmniGibson 3.9.1 are:

| `--robot` | Asset | Arm DOF | Controller | Notes |
|---|---|---|---|---|
| `DROID` | `droid.usd` (bare Franka + Robotiq) | 7 | `CustomJointController` | the paper's robot |
| `DROID_mounted` | `droid_mounted.usd` (with base column) | 7 | `CustomJointController` | |
| `DROID_ee_control`, `DROID_ee_delta_control`, `DROID_mounted_ee_control` | as above | 7 | `DroidEndEffectorController` | Cartesian actions |
| `DROID_default_pd_control`, `DROID_polaris_control`, `DROID_no_wrist_cam` | `droid.usd` | 7 | joint PD variants | |
| `YAM` | `yam.usd` (YAMLab arm, bare) | 6 | stock `JointController`, YAMLab `high_pd` gains | see below |
| `YAM_base_pd_control` | `yam.usd` | 6 | stock `JointController`, YAMLab `base` gains | |

`UR5*` and `WidowX` configs exist but have no registered definition on 3.9.1 and do not load.

**YAM.** Ported from [YAMLab](https://github.com/ARISE-Initiative/yamlab): the spec is
`realm/robots/yam.py`, the definition `realm/robots/definitions/yam/yam.yaml`, the asset
`realm/robots/yam/yam.usd` (rebuilt from YAMLab's export by `scripts/build_yam_usd.py`; see
`realm/robots/yam/PROVENANCE`). Actions are `[6 absolute joint targets, gripper]`, observations
`proprio[:6]` + the `left_finger` position as the gripper state (`-0.0475` open, `0.0` closed), one
wrist camera under `link_6`. The eef frame is a massless `eef_link` 14.3 cm out along the `link_6`
flange axis (the midpoint of YAMLab's fingertip keypoints); `get_ee_pose` and the Cartesian metrics
report that point. The arm base is spawned `mount_height` (0.863891 m, the DROID column
height) above the scene's robot pose so the exterior cameras frame the workspace as for DROID; the
value is a config key, not a measurement. Only the `debug` model type has been exercised with a
6-DOF state: `openpi`/`dreamzero` policy servers must accept a 6-entry `observation/joint_position`.

> **Verified so far (2026-09-04, RTX 5090, `debug` model, 90 steps, `--no-render_on_demand`; task 0
> single-view and task 1 `--multi-view`):** the definition loads, `assert_proprio_layout` and
> `assert_wrist_camera` pass (DOF order is joint1..6, left_finger, right_finger), all four artifacts
> are written with 7-wide qpos/action rows, the wrist camera renders the tabletop the right way up, and
> with task 1's camera placement the folded arm is visible at table height in the first exterior view.
> **Not yet checked:** any motion -- the debug model holds the zero pose, so the arm never moved and
> the gripper never closed; a real policy run is still owed. Task 0's exterior extrinsics
> (`ep_001042_cam1`) leave the zero-pose arm ~10 deg outside the frustum, so only the camera mount
> shows at the frame edge there; a raised `reset_joint_pos` is the knob if that matters.

## Frequencies

The control and rendering frequency is fixed per robot in `realm/sim_config.py`:

| Robot | Sim step / rendering |
|---|---|
| `DROID*` | 15 Hz |
| `YAM*` | 30 Hz (YAMLab: 120 Hz physics, decimation 4) |

Physics always runs at **120 Hz**. At 15 Hz that is 8 physics substeps per environment step; at
30 Hz, 4.

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
