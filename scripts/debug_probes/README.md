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

# pump, in the container
ROOT=~/projects/REALM_og391
mkdir -p $ROOT/tmp/dbg/{inbox,outbox} $ROOT/tmp/dbg_tmp
cp scripts/debug_probes/pump.py $ROOT/tmp/dbg/
srun --jobid=<ID> --overlap apptainer run --userns --nv --writable-tmpfs \
  --bind $ROOT:/app --bind $ROOT/data/datasets:/data --bind $ROOT/data/cache:/cache \
  --bind $ROOT/tmp/dbg:/dbg --bind $ROOT/tmp/dbg_tmp:/tmp \
  --env TMPDIR=/tmp --env OMNIGIBSON_HEADLESS=1 --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 --env REALM_ROBOT=DROID_robolab --env REALM_PORT=8500 \
  $ROOT/realm_og391.sif python /dbg/pump.py &
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
- `ee_press_compliance.py` -- drives the CLOSED gripper straight down into the table under EE
  control and records the external view, to test whether the robolab 2F-85 fingers are compliant.
  Picks its own descent column (nearest table-height surface in front of the robot, then the point
  on it furthest from every object standing on it), verifies the commanded orientation round-trips
  before descending, and logs the pad links in the `panda_link8` frame so arm motion is removed.
  `REALM_ROBOT=<config>`, writes to `$REALM_OUT` (default `/logs/ee_press`).

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

