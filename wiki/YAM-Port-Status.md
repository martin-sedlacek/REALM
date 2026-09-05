# YAM port: where we left off (2026-09-05)

Handoff for the `add-yam` branch. Everything below is committed on `add-yam` (13 commits on top of
`v1.0.0`, `2add807`); the operator reference is [Running evaluations → Robots](Running-Evaluations#robots).

## What exists

Three robots, all `--robot` values, all sharing `realm/robots/yam.py` as the single source of numbers:

| `--robot` | What | Asset |
|---|---|---|
| `YAM` (+ `YAM_base_pd_control`) | single YAMLab arm | `realm/robots/yam/yam.usd` |
| `YAM_bimanual` | YAMLab's two-arm workstation | `yam_bimanual.usd` |
| `YAM_crank_bimanual` | the same workstation with I2RT's crank gripper from ABC's MuJoCo model | `yam_crank_bimanual.usd` (from `yam_crank.usd`) |

Build chain (host, needs `pxr` + numpy; the scratch venv used so far is not part of the repo):

```sh
python scripts/build_yam_usd.py --source <yamlab>/yamlab/robot/yam/arm/yam.usd          # yam.usd
python scripts/build_yam_crank_usd.py --mjcf <abc>/assets/put_bottles/assets/i2rt_yam/yam.xml   # yam_crank.usd
python scripts/build_yam_bimanual_usd.py                  # yam_bimanual.usd
python scripts/build_yam_bimanual_usd.py --variant crank  # yam_crank_bimanual.usd
```

Sources: YAMLab at `ARISE-Initiative/yamlab` commit `ec0455d`; ABC at commit `6bc6586` (`abc.bot`). Both were
sibling checkouts on the laptop (`/home/elmo/sedlam/yamlab`, `/home/elmo/sedlam/abc`); `realm/robots/yam/PROVENANCE`
carries every input hash.

Every asset carries YAMLab's aluminium gate frame as a visual-only link, stands on the floor at `mount_height`
(the DROID column height) and is shifted 0.30 m toward the workspace by the `spawn_offset` config key. Wrist
cameras render 960x720 (4:3, the D405 calibration aspect; 78.6 x 63.1 deg). The crank camera has a 2 cm near
plane (the finger bases are 3 cm from the lens). The `debug` policy on the YAMs is a true no-op (hold joints,
grippers open); on DROID it is the historical zero action.

## Verified

- **GPU (Clara L40S, `realm_og391_v3.sif`, jobs 2045xx–2046xx; laptop RTX 5090 Docker):** `YAM` and
  `YAM_bimanual` load, all four artifacts are written, `tests/test_yam_bimanual_motion.py` drives each of the
  14 action columns to its joint exactly, both grippers open/close, `--model_type yamlab` against the reference
  sweep server moves both arms. A `DROID_mounted` rollout on this branch is **bit-for-bit identical to main**.
- **Visually in the GUI (laptop):** frame, spawn offset, 4:3 wrist views, crank gripper rendering with the
  arm's materials, no collision blobs, fingers no longer clipped by the near plane.
- **Host tier-1:** ruff clean; `tests/test_yam_robot.py` (30 cases) pins definitions, configs, obs profiles,
  action layouts, USD structure (with pxr) and the debug adapter; `run_suite --only local` PASS.

## Not verified

- `YAM_crank_bimanual` has **not run on a GPU** since it was created. First check:
  `tests/test_yam_bimanual_motion.py --robot YAM_crank_bimanual` (each action column moves exactly its joint;
  gripper phases reach normalised 0 open / 1 closed — note the inverted finger sign).
- No trained policy has driven any YAM robot. `--model_type yamlab` speaks YAMLab's LeRobot contract
  (fingers in metres, `-0.0475` open); an ABC-trained policy (state = `q / 0.0475`, 1 open; DiT, 224x168
  4:3 images resized-with-pad to 224x224) needs its own adapter.
- The exterior cameras are REALM's 1280x720 at the task extrinsics (the bimanual top camera is at YAMLab's
  pose but REALM's resolution); ABC's top camera is also 58 deg vertical at 4:3.

## Control: the gap between commanded and measured joints (the open thread)

ABC's `states_actions.bin` holds 14 states + 14 actions per 30 Hz step, actions being absolute joint targets, so
it is the same commanded-vs-measured pairing REALM used to tune DROID. Fitted on the 6 real and 4 sim
put-bottles episodes of the preview set (first-order model `s[t+1] - s[t] = alpha (a[t] - s[t])`, both arms
pooled; a pure-delay term did not improve the fit):

| joint | real: tau (63% rise) | real mean gap | ABC sim: tau | ABC sim mean gap |
|---|---|---|---|---|
| j1 (DM4340) | 115 ms (3.4 steps) | 0.02 rad | 59 ms | 0.007 rad |
| j2 | 143 ms (4.3) | 0.04 | 73 ms | 0.014 |
| j3 | 123 ms (3.7) | 0.04 | 59 ms | 0.012 |
| j4 (DM4310) | 226 ms (6.8) | 0.05 | ~30 ms (overdamped, kp 20 / kv 0.5 in ABC's task file) | 0.006 |
| j5 | 210 ms (6.3) | 0.05 | 118 ms | 0.019 |
| j6 | 204 ms (6.1) | 0.04 | 106 ms | 0.017 |
| gripper | ~1 s (dominated by grasp clamping on the bottle) | 0.04 (norm.) | — | — |

Takeaways: the real controller is lag-dominated with no steady-state offset (fraction of "still but > 0.05 rad
off" steps < 2%); the wrist motors are ~2x slower than the shoulder motors; ABC's own sim tracks 2–3x tighter
than the real robot; and REALM's YAM configs use YAMLab's stiff `high_pd` gains (800/50 shoulder+elbow, 30/5
wrist), tighter still. The gripper state in the data almost never reaches fully closed — grasps stop at 0.4–0.6
(bottle width) — so a closed-gripper comparison against the sim has to be made holding an object.

**Plan.** (1) On Clara, run step responses on `YAM_crank_bimanual` for the two gain sets already in the repo
(`high_pd`, and the MJCF/`base` set 40/2.5 – 10/1 – 100/10, which are the DM motors' MIT-mode gains) and
measure tau per joint with the `test_yam_bimanual_motion.py` machinery; (2) pick/fit `isaac_kp`/`isaac_kd` per
actuator group so tau matches the real column above (target ~120–140 ms shoulder, ~210 ms wrist); if PD alone
cannot reproduce the lag without adding steady-state error, add a command delay of 2–3 control steps (the
DROID controller has no such mechanism today); (3) replay a real episode's actions in the sim and compare the
resulting joint trajectory with the recorded states as the acceptance test. Keep DROID bit-identical throughout.

## Data

- `logs/abc_preview/` on the laptop (also rsynced to Clara, see the handoff message): the public preview tar
  (155 MB), the extracted `train/` + `val/` episodes, `wrist_snaps/` (wrist frames + contact sheets, `closed/` =
  tightest grasp per arm) and a README describing the formats.
- The full ABC-130k (raw MCAPs at native camera resolution) is gated on Hugging Face: accept the terms, `hf auth
  login`, then `uv run export_hf_task.py --task put_the_plastic_bottles_in_the_bin --split train --max-episodes 1`
  in the ABC checkout.

## Useful commands

```sh
# GUI look at a robot (Docker on the laptop; Apptainer on Clara per scripts/run_apptainer.sh)
python /app/examples/04_vector_evaluate.py --num_envs 1 --repeats 1 --max_steps 9999 --task_id 0 --perturbation_id 0 \
    --robot YAM_crank_bimanual --model_type debug --model_name yam_view --port 0 --experiment_name yam_placement \
    --log_dir /app/tmp/yam_logs --no_record --no-render_on_demand
# nudge the robot with the keyboard and print a spawn_offset block
OMNIGIBSON_HEADLESS=0 python /app/scripts/yam_placement_gui.py --robot YAM_crank_bimanual --task_id 0
# host tier-1
uv run ruff check realm examples tests scripts
uv run python -m pytest -q tests/test_perturbation_task_types.py tests/test_cell_classification.py \
    tests/test_robot_base_column.py tests/test_robot_definition_parity.py tests/test_yam_robot.py
```

OmniGibson gotchas met along the way (all encoded in CLAUDE.md's "Add a robot"): links must be direct children
of the root and cameras direct children of a link; the eef link is made invisible (use a massless frame);
collision meshes must be direct children of their link (CoM rule); collision prims of types outside
`{Sphere, Cube, Cone, Cylinder, Mesh}` render unless `purpose = guide` is authored; the stock binary gripper
controller maps open to the UPPER limit (name `open_qpos`/`closed_qpos`); multi-arm DOF order is breadth-first.
