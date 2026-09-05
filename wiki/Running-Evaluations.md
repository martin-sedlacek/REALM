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
| `debug` | **no server needed.** DROID: the historical constant all-zero action (Franka zero pose, gripper closed). YAM robots: a true no-op -- holds the current joints, gripper open |

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
| `YAM_bimanual` | `yam_bimanual.usd` (two YAM arms on a shared mount) | 2 x 6 | stock `JointController` x2 + `MultiFingerGripperController` x2, `high_pd` gains | 14-D action, two wrist cameras, YAMLab top camera as the exterior view; see below |
| `YAM_crank_bimanual` | `yam_crank_bimanual.usd` (the same workstation with I2RT's crank gripper) | 2 x 6 | as `YAM_bimanual` | ABC-project gripper: inverted finger sign, steeper wrist camera, ABC home pose; see below |

`UR5*` and `WidowX` configs exist but have no registered definition on 3.9.1 and do not load.

The state of the YAM port -- what is verified, what is not, the controller-gap analysis and the plan -- is kept
in [YAM port status](YAM-Port-Status).

**YAM.** Ported from [YAMLab](https://github.com/ARISE-Initiative/yamlab): the spec is
`realm/robots/yam.py`, the definition `realm/robots/definitions/yam/yam.yaml`, the asset
`realm/robots/yam/yam.usd` (rebuilt from YAMLab's export by `scripts/build_yam_usd.py`; see
`realm/robots/yam/PROVENANCE`). Actions are `[6 absolute joint targets, gripper]`, observations
`proprio[:6]` + the `left_finger` position as the gripper state (`-0.0475` open, `0.0` closed), one
wrist camera under `link_6` rendered at 960x720: 4:3 like the D405 calibration and YAMLab's own 640x480
frames, so both FOVs (78.6 x 63.1 deg) match; the recorder letterboxes it next to the 16:9 exterior views. The eef frame is a massless `eef_link` 14.3 cm out along the `link_6`
flange axis (the midpoint of YAMLab's fingertip keypoints); `get_ee_pose` and the Cartesian metrics
report that point. The arm base is spawned `mount_height` (0.863891 m, the DROID column
height) above the scene's robot pose so the exterior cameras frame the workspace as for DROID; the
value is a config key, not a measurement. On top of that, every YAM config carries a REALM-only
`spawn_offset` (`pos: [0.30, 0.0, 0.0]`, `yaw_deg: 0.0`): a rigid shift of the whole robot in its own
frame, 0.30 m straight toward the workspace, because the YAM's reach is well short of
the Franka's. Objects stay where the scene places them; the robot-frame cameras (the YAM_bimanual top
camera, the task extrinsics) and the EE-control transforms move with the robot. The value was chosen by
eye in the GUI on 2026-09-05; `scripts/yam_placement_gui.py` nudges the robot with the keyboard and prints
the block to paste if it needs retuning. Every YAM asset also carries YAMLab's aluminium gate as the
visual-only link `frame` (see YAM_bimanual below); the single arm stands centred on its front cross bar.
Only the `debug` model type has been exercised with a
6-DOF state: `openpi`/`dreamzero` policy servers must accept a 6-entry `observation/joint_position`.

> **Verified so far (2026-09-04, RTX 5090, `debug` model, 90 steps, `--no-render_on_demand`; task 0
> single-view and task 1 `--multi-view`):** the definition loads, `assert_proprio_layout` and
> `assert_wrist_camera` pass (DOF order is joint1..6, left_finger, right_finger), all four artifacts
> are written with 7-wide qpos/action rows, the wrist camera renders the tabletop the right way up, and
> with task 1's camera placement the folded arm is visible at table height in the first exterior view.
> **Motion:** verified 2026-09-05 through the bimanual asset, which shares the arm (see YAM_bimanual below).
> **Corrected 2026-09-05:** "the gripper never closed" in that run was a bug, not the debug model.
> OmniGibson's stock binary `MultiFingerGripperController` sends open to the joints' UPPER limit and
> close to the LOWER one, and on the YAM fingers the upper limit (0.0) is closed and the lower (-0.0475)
> open, so every gripper command was inverted. Both single-arm configs now set
> `open_qpos: [-0.0475, -0.0475]` / `closed_qpos: [0.0, 0.0]` (verified on the bimanual asset, which
> shares the gripper block: a close command now reaches the closed limit). Task 0's exterior extrinsics
> (`ep_001042_cam1`) leave the zero-pose arm ~10 deg outside the frustum, so only the camera mount
> shows at the frame edge there; a raised `reset_joint_pos` is the knob if that matters.

**YAM_bimanual.** YAMLab's bimanual workstation as ONE OmniGibson robot (spec
`realm/robots/yam.py::YamBimanualRobot`, definition `realm/robots/definitions/yam_bimanual/`, asset
`realm/robots/yam/yam_bimanual.usd` composed from the single-arm file by
`scripts/build_yam_bimanual_usd.py`). The two arms sit 0.61 m apart in y on a geometry-free
`base_link` at their midpoint, which is the robot frame; the midpoint is spawned `mount_height` above the
scene's robot pose and shifted by the same `spawn_offset` as the single arm (0.30 m forward). The arms
stand on YAMLab's aluminium-extrusion frame, carried as the visual-only link `frame` (no collision, so it
never counts as an environment collision): the mesh is YAMLab's `gate_visual`, with the part below the
arm plates stretched in z so the feet reach the floor at `mount_height` while the posts and top bar above
the plates keep their real dimensions. The crank variant's finger collision shapes are primitive capsules
and boxes with `purpose = guide` authored in the USD: OmniGibson only hides collision geometry of the gprim
types it classifies, and a `Capsule` is not one of them. Links and joints carry the arm as a prefix (`left_link_6`,
`right_joint1`, `left_left_finger`), and the arms collide with each other but not with themselves
(`self_collisions: true` + every within-arm pair filtered, matching YAMLab's per-arm articulations with
self-collisions off).

* **Action** (14-D, YAMLab's `YamActionLayout`): `[left_arm(6), left_gripper, right_arm(6),
  right_gripper]` -- absolute joint targets, gripper columns 6 and 13 binarised by `realm.rollout`.
* **State**: `robot_state` is `[left joints(6), right joints(6)]` and `gripper_state` a 2-vector
  `[left, right]` in (0 open, 1 closed), both looked up by joint name in the articulation DOF order
  (`ROBOT_OBS_PROFILES["YAM_bimanual"].dof_order`, asserted against the loaded robot at construction).
  The qpos parquet rows are therefore 14 wide: `[12 joints, 2 grippers]`.
* **Cameras**: both wrist cameras (`<arm>_link_6/wrist_camera`, the right one at YAMLab's separately
  calibrated offset) come through as `PolicyObservation.wrist_im` (left) and `.wrist_im_second` (right);
  the recorder tiles them as a 2x2 grid `top | second exterior (or black) / left wrist | right wrist`.
  The fixed **top camera** is NOT in the USD: the robot config's REALM-only `exterior_camera` key places
  `external_sensor0` at YAMLab's `cameras.top` pose relative to the mount frame (0.17 m behind, 0.94 m
  above the arm bases, looking forward and 60 degrees down, 78.4 degrees horizontal FOV) in place of the
  task's `cam1` extrinsics, so V-VIEW and the recorder treat it like any exterior view. `--multi-view`
  adds the task's second exterior camera as usual.
* **Scoring**: grasp detection checks each arm's finger pair separately and reports a grasp when
  either arm holds the object; `get_ee_pose` and the Cartesian metrics report the LEFT (default) arm's
  eef frame. The ten REALM tasks are single-arm tasks -- a bimanual task set is not part of this port.
* **Inference**: `--model_type yamlab` speaks YAMLab's LeRobot contract over openpi's websocket
  protocol: `observation.state` (14: `left_joint1..6, left_finger, right_joint1..6, right_finger`, fingers
  in metres), `observation.images.top_rgb` / `left_rgb` / `right_rgb` at REALM's native 1280x720, `prompt`;
  it expects `{"actions": (n, 14)}` absolute targets in the same layout and converts the finger targets to
  REALM's open-fraction gripper value (`realm/inference/yamlab.py`). `tests/yamlab_sweep_server.py` is a
  reference server that validates the contract and answers with a joint sweep; start your own policy
  server with the same protocol and point `--port/--host` at it. `debug` also works (holds the reset pose with the grippers open). The single-arm adapters (`openpi`, `dreamzero`) send a 12-entry joint state and ignore the