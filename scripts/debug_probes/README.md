# Interactive debug probes

Booting the OG 3.9.1 sim costs ~4-5 minutes. `pump.py` pays that once: it creates the env and robot,
connects a policy client, then watches `/dbg/inbox` for `*.py` snippets, execs each in a shared
namespace and writes captured stdout to `/dbg/outbox/<name>.out`. A probe then costs ~1 s instead of
a fresh boot.

## Running it

Take a long interactive allocation and start the policy server and the pump as separate steps:

```sh
salloc --no-shell -p l40s --gres=gpu:L40S:1 --cpus-per-task=32 --mem=120G -t 12:00:00 -J og391-debug

# policy server, on the host (openpi is not in the container)
srun --jobid=<ID> --overlap bash -c '
  cd ~/projects/openpi
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  uv run scripts/serve_policy.py --port=8500 policy:checkpoint \
    --policy.config=pi05_full_droid_finetune \
    --policy.dir=~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos' &

# pump, in the container. Run from the repo root. Paths come from the harness's own resolver rather
# than being spelled out here -- and do NOT substitute $REALM_ROOT / $REALM_SIF straight from your
# shell: the profile exports both, naming the pre-port 1.1.1 tree and image. See
# scripts/clara/lib/paths.sh, which overwrites them with the og391 values.
source scripts/clara/lib/paths.sh
ROOT=$REALM_ROOT
mkdir -p $ROOT/tmp/dbg/{inbox,outbox} $ROOT/tmp/dbg_tmp
cp scripts/debug_probes/pump.py $ROOT/tmp/dbg/
srun --jobid=<ID> --overlap apptainer run --userns --nv --writable-tmpfs \
  --bind $ROOT:/app --bind $REALM_DATA:/data --bind $REALM_APPDATA:/cache \
  --bind $ROOT/tmp/dbg:/dbg --bind $ROOT/tmp/dbg_tmp:/tmp \
  --env TMPDIR=/tmp --env OMNIGIBSON_HEADLESS=1 --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 --env REALM_ROBOT=DROID_robolab --env REALM_PORT=8500 \
  $REALM_SIF python /dbg/pump.py &
```

Then drop probes into `tmp/dbg/inbox/` and read `tmp/dbg/outbox/`. The namespace exposes
`env`, `robot`, `obs`, `client`, `rollout()`, `extract_from_obs`, `og`, `np`, `th`.

`REALM_ROBOT` selects `realm/config/robots/<name>.yaml`, so the same pump serves the stock asset for
an A/B without an edit.

## Traps, all of which cost real time on 2026-08-11

- **Boot the stock asset alongside the new one, from the start.** A second pump on the same
  allocation is one more `srun --overlap`, and it is what killed three wrong hypotheses (the missing
  arm in the exterior view, the post-warmup EE rise, the wrist extrinsics) in minutes each.
- **Confirm you have exactly ONE pump before trusting any output.** `squeue -j <ID> -s` should show
  one `apptainer` step. The pump only checks `/dbg/STOP` *between* probes, so a `touch STOP` during a
  long probe is missed; if you then delete STOP and start a second pump, both glob the same inbox and
  race to write the outbox. That produced three probes' worth of contradictory results, read as stale
  bytecode and then as a consumed contact buffer. Both wrong.
- **Print a code-identity block first.** Whether the fix you just made is actually in the live source
  (`inspect.getsource`), what the config values resolved to, what the controller overrides are. Stale
  code then shows up in the output instead of being inferred three probes later.
- **Wrap every probe body in a function.** Snippets exec in the pump's own `globals()`, so a bare
  `for name, ... in ...` clobbers the pump's `name` and misnames its output file.
- **Keep the debug dir on Lustre.** `/tmp` is node-local, so a login-node scratch dir is invisible on
  the compute node and the bind fails with `mount source doesn't exist`.
- **Watch the videos early.** `scripts/videos_parquet_to_mp4.py` extracts them in seconds (the
  parquet holds complete MP4 bytes). Two bugs that survived three measurement-driven sessions were
  each obvious in a single frame.

## Probes here

- `verify_gripper_mapping.py` — holds each binary gripper command and reads back the physical jaw gap
  *and* the `gripper_state` handed to the policy, refusing to run rollouts if either is inverted.
  **Run this before any batch on a new gripper.** Both failure modes it checks are silent, and one of
  them cost a full 10-repeat batch. Note its precondition: the finger link origins must be on the
  pads, or you are measuring the linkage rather than the jaw — see
  `scripts/fix_robolab_link_origins.py`.

- `ee_ik_smoke.py` -- the bottom-up check that dm_robotics end-effector control still works at all.
  **No GPU and no Isaac**: it only needs the container, so it runs on a login node in ~30 s. Imports
  `dm_control` / `dm_robotics.moma.effectors`, builds `RobotIKSolver`, prints the MJCF joint order
  (which is what the OG arm DOFs must match), and drives a closed loop to a reachable target to prove
  the QP actually solves rather than returning zeros. Run this before blaming the env for an EE bug.
- `ee_env_probe.py` -- boots one env under an EE-control config and dumps the things that fail
  silently: the OG joints `dof_idx` selects vs the MJCF chain the IK solves on
  (`JOINT_MAPPING_MATCH`), the frame the controller's eef pose is in vs `env._world2robot`
  (`HEIGHT_OFFSET_CONSISTENT` -- see below), `og.sim.device`, the resolved wrist-camera key, and 8
  real steps of the debug action with the pose error per step. `REALM_ROBOT=<config>`.
- `ee_press_compliance.py` -- drives the **OPEN** gripper straight down onto the table under EE
  control and records it, to test whether the 2F-85 fingertips curl inward when they catch on a
  surface. `REALM_GRIP` selects the jaw state and defaults to `open`; it was hardcoded CLOSED until
  2026-08-14, and a closed jaw braces the two pads against each other along the linkage's stiff axis,
  which geometrically forbids the inward curl -- every press number taken before that flip is about a
  different experiment. The descent is adaptive (it lands when the arm's tracking lag exceeds
  `REALM_SHORT_TH`, then overtravels `REALM_OVERTRAVEL` past the ACHIEVED landing height) and
  external_sensor0 is re-aimed every step, perpendicular to the closing plane and clamped to stay
  `REALM_CAM_MIN_ABOVE` above the table top. Before descending it **levels the tips**: it rolls the
  hand about the horizontal axis perpendicular to the closing axis until the two pad origins sit at
  the same world z. Without that, the arm stalls against the first tip it lands and the second never
  reaches the surface -- measured 2026-08-14, `pivL` read `+0.00 deg` at every depth out to 200 mm of
  overtravel on both assets while `pivR` did all the moving. Everything is reported PER PAD, pivot
  angles in DEGREES, and the descent stops early if the arm's tracking lag stops growing, which is
  the test for "the arm is the limiter, not the gripper".
  Picks its own descent column (nearest table-height surface in front of the robot, then the point
  on it furthest from every object standing on it), verifies the commanded orientation round-trips
  before descending, and logs the pad links in the `panda_link8` frame so arm motion is removed.
  `REALM_ROBOT=<config>`, writes to `$REALM_OUT` (default `/logs/ee_press`), and carries the signed
  **tip-vs-heel** direction test. **Superseded as the compliance-MAGNITUDE test by
  `gripper_squeeze_compliance.py`**: pressing with the jaws SHUT loads the four-bar along its stiff
  axis, which is why it measured ~0.13 mm on both assets. That marking does NOT extend to the
  *direction* question -- "do the tips curl inward when pressed onto a surface" is a press question
  and a squeeze cannot answer it, which is what the tip/heel block is for. One thing to check before
  reading any of its numbers as "a press N mm past contact": **where the arm actually stopped.** It
  descends and it presses -- both this probe's own control run and `curl_A` (job 191032) reach the
  table -- but it STALLS there, and every further commanded millimetre then moves `panda_link8` by
  microns (117 mm of command -> 0.2 mm of motion in `curl_A`). Two consequences: the commanded depth
  is not the achieved one, and a "hover" pose computed from geometry can already be in contact, which
  silently makes the rest reference a LOADED pose. Log the achieved eef z and the pad world z against
  the table top, which this probe prints, and treat `pads went -N mm BELOW the surface` (a negative
  number) as the signal that the contact is not where the arithmetic thinks it is.
- `curl_press_direction.py` -- the SIGNED press probe: WHICH WAY the fingertips rotate under a press,
  not how far. Default `--load tip` keeps the arm at `reset_qpos` under joint control and ramps a
  200 kg pinned object UP into ONE fingertip, 0.5 mm per step, until the contact view reports contact
  and then `--tip-past` steps further -- no IK, no arm motion, contact force measured rather than
  inferred. Each fingertip in turn, per mimic `nf`/`dr` rung, with an unloaded reference per rung (the
  object parked 1.3 m away) and a release phase to tell an elastic violation from a snap-through. Two
  hull-free observables, both signed `+ = INWARD`: pad-origin separation (link poses) and pad rotation
  about the closing-plane normal (link orientations); the sign convention is derived in the module
  docstring from the frame rather than read off a joint value. `--load ee` is the table-press load
  case and is documented broken on this build. Grep `CURL_VERDICT` / `CURL_PROBE_OK`.
- `gripper_squeeze_compliance.py` -- the squeeze counterpart, and the probe that actually answers the
  compliance question. Closes the jaws on the task cube under **joint control** -- no IK anywhere: the
  arm holds `reset_qpos[:7]` for the whole run and the OBJECT is teleported to the midpoint between
  the pads (gravity disabled, one face normal put exactly on the closing axis) instead of the hand
  being driven to the object. Squeezes twice, once with the object free and once with it pinned heavy
  so it cannot recoil, then restores gravity to see whether it is held. Records a gripper close-up
  from a repositioned `external_sensor0`; the wrist camera looks along the fingers and hides bending.
  `--robot <config> --out <dir>`. Pair with `gripper_squeeze_analyse.py`, which rebuilds one uniform
  table across several runs from their npz files. Results in
  `~/runbook/streams/realm_og391_port.md`.
- `gripper_squeeze_analyse.py` -- the cross-run table. **Read its module docstring before quoting any
  millimetre from either script**: it carries the two measurement traps (the jaw gap must be
  self-calibrated per asset, and the flex-vs-unloaded-linkage estimator has a ~0.5 mm error bar
  because the drive slews the whole jaw in one 15 Hz step and leaves the reference curve sparse).
- `xflat_run_chain.sh <jobid>` -- the whole `droid_robolab_xflat` measurement chain in one
  allocation: runtime mass properties, then the curl at the AUTHORED `nf=1000`, each paired with a
  `DROID_robolab_v2` control taken in the same session with identical flags. **`MODE=stock`
  throughout** -- the image's own OmniGibson with no loader patch bound over it, because the point of
  the flattened asset is that it needs none. The control is what proves the loader is still broken
  and the asset alone is doing the work; without it a passing run cannot be told from a quietly
  patched image. Reuses `inertia_runtime_realm.py` (branch `inertia-diff`) verbatim, so the two
  routes' numbers are the same quantity in the same convention. Build the asset first with
  `scripts/make_xflat_gripper_usd.py`.

### Gripper traps (2026-08-14)

- **`collision_boundary_points_world` is ~120 mm off the pad LINK ORIGINS on robolab v2.** The two
  pad origins are exactly symmetric about the flange axis (the closing axis and the finger long axis
  both come out exactly axis-aligned in the `panda_link8` frame); the hull points are not -- their
  centroid sits ~123 mm off along the closing axis. Same ~120 mm the squeeze probe records as
  `hull_off` between the task cube's hull centre and its own pose, so it is not specific to the cube.
  Measured consequence: within ONE rung the hull-based tip-to-tip separation and the origin-based pad
  separation moved in OPPOSITE directions (+5.2 mm vs -2.8 mm). Hull extents along an axis are still
  fine as SIZES (a translation offset cancels), and `gap_hull` is still fine as a self-calibrated
  relative measure, but do not build a signed displacement observable on hull points here. Link poses
  and link orientations are unaffected.
- **A link-origin separation is not a jaw gap, and it is not comparable across assets.** At full
  closure robolab v2's inner-finger origins sit 33.0 mm apart and stock droid_mounted's 7.1 mm, so
  only the *change* means anything. Calibrate: subtract each asset's own value at full unloaded
  closure, where the pads touch and the gap is zero by definition. Cross-check against the finger's
  convex-hull extremes along the closing axis (also constant-offset, -24.0 mm on robolab) and against
  a known object width. On robolab both then read 83.18 mm open, against an 85 mm nominal 2F-85
  stroke. On stock they disagree (73.0 vs 87.2 mm) and only the hull measure is validated.
- **The stock asset has a mimic joint too.** `left_inner_finger_prismatic_joint` is
  `PhysxMimicJointAPI`-coupled to the right one, with kp/kd forced to 0 like any mimic DOF. "Stock has
  no mimic joints" is wrong; what it does not have is a kinematic four-bar -- its two revolute finger
  joints are driven independently, from the *measured* prismatic positions, by REALM's
  `droid_gripper_controller` override. So "deviation from the linkage's unloaded relation" has no
  meaning on stock: fitting one gives a 0.217 rad unloaded residual, and the 0.09-0.13 rad of "flex"
  that falls out of it is pure model error. Use the prismatic joints (1 nm of movement under load).
- **OmniGibson overwrites the asset's authored drive gains.** `controller_base` defaults
  `isaac_kp/isaac_kd` to 1e7/1e5 for any POSITION controller whose config does not name them, and
  `robot.update_controller_mode()` pushes that into the joint on **every** `og.sim.play()`. robolab's
  `finger_joint` authors stiffness 100 / damping 0.0002 in the USD, so the sim runs it 10^5 times
  stiffer than the asset asks; only `maxForce` (16.5) survives. A gain poked straight onto the joint
  is therefore wiped by the next stop/play -- set it in the `gripper_0` controller config instead.

### EE-control traps (2026-08-14)

- **`robot._controllers[name]` is a `(group_key, controller_idx)` tuple in OG 3.9.1**, not a
  controller. Resolve through `ControllerView` (`get_mode`, `get_dof_idx`, ...) or
  `ControllerView._controller_groups[group_key]` for the instance. Reading an attribute off the
  tuple is what had kept EE control dead since the port.
- **`height_offset` is a property of the ASSET, not of the controller.** It converts a command in
  the DROID arm-base frame into the robot-prim frame the controller compares against. 0.87 is right
  for `droid_mounted.usd` (panda_link0 sits 0.86444 above the prim) and catastrophically wrong for
  `droid_robolab_v2.usd` (panda_link0 IS the prim). Copying an EE arm block between assets without
  revisiting it silently moves the target by 0.87 m -- and
  `tests/test_vector_integrity.py` reports PASS through it.


### The mimic-constraint / max_effort sweep (2026-08-14)

`gripper_squeeze_compliance.py` gained a **sweep mode** and two knobs that no REALM config touches.
Full results in `~/runbook/streams/realm_og391_port.md`; the short version and the traps:

- `--rungs "name=nf/dr/onf/odr/me/spi/kp/kd/ccs/ccd,..."` runs the whole OPEN -> unloaded calibration
  sweep -> free close -> squeeze A -> squeeze B cycle **once per configuration in a single process**.
  Every rung takes its own jaw-gap zero and its own reference kinematics. **Repeat a rung** (`def0`
  and `def1` with identical settings) -- that pair is the error bar, and it came out at 0.047 mm.
- **Rungs are CUMULATIVE.** `-` means "leave what the previous rung left", not "the authored value".
  An effort rung placed after an nf rung measures both unless every field is restated.
- **`--mimic-joints`** selects which mimic joints `nf`/`dr` apply to (default: the four inner ones).
- Companions: `mimic_table.py` (cross-run table in degrees, with the unloaded slop beside the loaded
  flex) and `mimic_contact_sheet.py` (`--diff` / `--sbs`, for the visibility question).

Traps worth not re-hitting:

- **`physxMimicJoint:<inst>:naturalFrequency` and `:dampingRatio` are not in this build's
  `PhysxMimicJointAPI` schema** (isaacsim 5.1.0 / omni.physx 107.3.26: `Usd.SchemaRegistry` lists only
  `gearing`, `offset`, `referenceJoint`, `referenceJointAxis`, and `_physxSchema.so` lacks the string).
  They are read anyway, as **custom attributes by literal token** -- `omni/physx/bindings/_physx*.so`
  exports `MIMIC_JOINT_ATTRIBUTE_NAME_NATURAL_FREQUENCY_ROT{X,Y,Z}`. Do not conclude from the schema
  that they are inert; a runtime write changes the physics and is reversible.
- **The mimic instance token is not the joint's `physics:axis`.** All six gripper joints author axis Z,
  yet the four inner ones use `PhysxMimicJointAPI:rotX` and `right_outer_knuckle_joint` uses `rotZ`.
- **Every USD write must be inside `with og.sim.editing_usd():`** or OmniGibson aborts the run
  (`simulator.py:1651`). That context is also what syncs the edit into Fabric.
- **`naturalFrequency` is what holds the four-bar together, not a spring behind a rigid pad.** The
  followers have no drive at all, so lowering it lets the kinematic relation fail. Below nf~=10 the
  fingers splay, the object tumbles out and the jaws close through it.
- **The jaw-gap estimator stops being valid once the fingers rotate.** Its own validation against the
  30.000 mm cube drifts from -0.19 mm at the default to +2.3 mm at nf=100 and +17 mm at nf=10, and
  `past object width` then goes negative -- the jaw reading WIDER than the object it holds. Use the
  follower-deflection columns, which need no hull geometry.
- **Judge visibility on squeeze A, never squeeze B.** On a 200 kg pinned cube two runs with IDENTICAL
  settings already differ by more than any rung differs from the default (7.19/255), so squeeze B
  cannot discriminate. Squeeze A (free 27 g cube) is also the realistic grasp.
- **Read the `pads` column on squeeze B only.** On squeeze A even the default ends with one pad in
  contact, because a free 27 g cube gets pushed onto one pad and rides there.
- `physxMaterial:compliantContactStiffness` IS in this build's schema, but
  `geom_prim.get_applied_physics_material()` returns nothing for either pad link, so the probe cannot
  reach the pad material that way. Unfinished lead, not a closed door.
