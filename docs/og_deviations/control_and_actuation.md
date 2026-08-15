# OmniGibson 3.9.1 vs Isaac Lab 2.2.0 — control and actuation

Read-only audit of everything in OmniGibson's control/actuation path that changes behaviour away
from raw Isaac Sim / Isaac Lab, or away from what the asset itself authored. **Nothing here is
fixed. Flagging only.**

## Scope and method

Audited tree: `/mnt/home_lustre/sedlam56/projects/OG-lite_og391` (a fork of BEHAVIOR-1K OmniGibson
v3.9.1). Files: `omnigibson/controllers/*`, `omnigibson/robots/robot.py`,
`omnigibson/prims/joint_prim.py`, plus every other site in the tree that writes joint drive state,
control modes, or actuation limits (found by grepping for `set_gains`, `set_control_type`,
`DriveAPI`, `drive:*:physics:*`, `set_max_efforts`, `set_max_velocities`, `armature`,
`jointFriction`) — which pulled in `omnigibson/objects/dataset_object.py` and
`omnigibson/simulator.py`.

Reference stack: Isaac Lab **2.2.0**, paths in-SIF absolute under
`/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif`, rooted at
`/workspace/isaaclab/source/isaaclab/isaaclab/` (`VERSION` = 2.2.0).
`/mnt/home_lustre/sedlam56/projects/RoboLab/robolab/robots/droid.py` is the worked example (same
Franka + Robotiq 2F-85 asset family REALM uses).

**Attribution is clean.** Commit `25c73e1` replaced the whole OmniGibson subtree with a verbatim
copy of the BEHAVIOR-1K v3.9.1 tag. Since then, `git log 25c73e1..HEAD -- omnigibson/controllers/
omnigibson/robots/ omnigibson/prims/joint_prim.py` returns exactly two commits, both touching only
`robots/robot.py`: `eaba43e` and `a1ee0d2`, and only one of their hunks changes behaviour
(row 17b). **Everything else in this document is stock upstream OmniGibson 3.9.1, not OG-lite.**

Line numbers cite the OG-lite tree at `6d04cc9`. Where OG-lite's insert shifted a line relative to
stock, the stock number is given too.

**Status.** The Isaac Lab side is a complete, verified extraction. The OmniGibson side covers what
this project measured or read directly and is **not exhaustive**. Sibling chapters:
[`rigid_bodies_and_collision.md`](rigid_bodies_and_collision.md),
[`simulator_and_scene.md`](simulator_and_scene.md),
[`transforms_and_assets.md`](transforms_and_assets.md).

---

## The headline difference

**Isaac Lab, when a config value is `None`, keeps what the USD authored. OmniGibson substitutes a
hardcoded default.** That single difference accounts for most of what follows.

Isaac Lab states it as a decision, verbatim at `actuator_base.py:165-168`:

```python
# resolve usd, actuator configuration values
# case 1: if usd_value == actuator_cfg_value: all good,
# case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
# case 3: if actuator_cfg_value is None: we use usd_value
```

The `usd_value` is read straight out of PhysX (`articulation.py:1542-1544`, `:1558-1559`) and passed
to the actuator (`articulation.py:1667-1681`). Resolution is `_parse_joint_parameter`
(`actuator_base.py:317`, `:331`, raising at `:350-351` if both are `None`).

OmniGibson's equivalent is `controller_base.py:204-207` falling back to `m.DEFAULT_ISAAC_KP = 1e7` and
`m.DEFAULT_ISAAC_KD = 1e5` (`:16-17`).

**Measured consequence** on REALM's Robotiq 2F-85: the USD authors `finger_joint` stiffness **100** /
damping **0.0002**; OmniGibson runs it at **1e7 / 1e5**. In matched units that is **~1745× stiffer**
and about seven orders more damped. Isaac Lab, given `stiffness=None, damping=None`, ran the same
joint at the authored values (5729.578 / 0.011459 in per-radian runtime units). See trap 3 below
before quoting any ratio from this chapter — an earlier "1e5× stiffer" figure in this project was a
degrees/radians error and is **retracted**.

## Table

Kind: **[I]** deviation from Isaac / Isaac Lab defaults · **[A]** override of an authored asset
value · **[C]** constraint imposed on the asset.

| # | kind | site (`file:line`) | what OmniGibson does | raw Isaac / Isaac Lab, or what the asset authored | silent? | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 1 | A, I | `controllers/controller_base.py:16-17`, applied `:204-207` | For `ControlType.POSITION`, substitutes `DEFAULT_ISAAC_KP = 1e7` / `DEFAULT_ISAAC_KD = 1e5` into the PhysX drive whenever the controller config names no `isaac_kp`/`isaac_kd` | `ImplicitActuatorCfg(stiffness=None, damping=None)` keeps the USD-authored gains (`actuator_base.py:165-168`, `:317-331`) — RoboLab does exactly this for the same gripper, `robolab/robots/droid.py:117-120` | **yes** — no log line, no warning | REALM Robotiq `finger_joint`: USD authors stiffness **100**, damping **0.0002**; runtime reads back **1e7 / 1e5**. In matched units **~1745× stiffer**, ~7 orders more damped (see trap 3 — the older "1e5×" figure was a unit error) | no |
| 2 | I | `controllers/controller_base.py:213` | For `ControlType.VELOCITY`, the default damping is `m.DEFAULT_ISAAC_KP` — i.e. **1e7**, the *stiffness* constant, not `DEFAULT_ISAAC_KD`. Near-certain upstream copy-paste bug | OG 1.1.1 used `default_kd = 1e5` here (`git show 0a38eb4:omnigibson/objects/controllable_object.py`, `:262-267` + `:826-831`). Isaac Lab substitutes nothing | **yes** | 100× more drive damping on every velocity-mode controller than 1.1.1. Affects the default base/gripper joint controllers, `DifferentialDriveController`, `HolonomicBaseJointController`. **Zero REALM impact** — no REALM controller is velocity-mode | no |
| 3 | I | `controllers/controller_base.py:476-482` | A POSITION controller writes the position target **and** a zero **velocity** target, every step (`enabled_controls * 0`) | Isaac Lab does the same (`articulation.py:213-220`) — and `_data.joint_vel_target` is sticky, nothing clears it on reset. The deviation is row 1's `kd`, not this | yes | With row 1's `kd = 1e5`, the drive contributes `−1e5·q̇` of pure braking. This is the term the runbook measured as "kd 1e5 → 40 removes the damping that was braking the jaw" | no |
| 4 | A, I | `robots/robot.py:632-651` re-called from `simulator.py:1373-1382` | Every `play()` after a `stop()` re-runs `update_controller_mode()`, re-writing the config gains, then `robot.reset()` and `robot.keep_still()` (`:1387-1388`) | Isaac Lab applies gains **once**, in `_initialize_impl` on the timeline PLAY event, guarded by `_is_initialized` (`asset_base.py:287-326`); **`reset()` does not re-apply them** (`articulation.py:172-182`; implicit `reset` is a no-op, `actuator_pd.py:111-113`) | **yes** | A runtime gain change survives `env.reset()` in Isaac Lab but is **wiped by any stop/play** in OmniGibson. Gains must go in the controller config, not onto the joint | no |
| 5 | A, I | `controllers/joint_controller.py:426`; `controllers/osc_controller.py:576-577` | `use_impedances=True` (or any OSC controller) makes `control_type` **EFFORT**. `controller_base.py:216-221` then *forbids* naming `isaac_kp`/`isaac_kd`, and `joint_prim.py:193-196` writes **kp = kd = 0** to the drive | Isaac Lab's effort-style actuators (`IdealPDActuator` etc.) do not zero the implicit drive as a side effect; you choose `ImplicitActuatorCfg` vs an explicit model per joint group | **yes** — and there is no config that avoids it | REALM's `IndividualJointPDController` (`realm/robots/droid_joint_controller.py:269-270`) returns EFFORT, so all 7 `panda_joint*` drives run at kp = kd = 0. RoboLab's arm runs kp 400 / kd 80 (`droid.py:105-106,114-115`) | no |
| 6 | I | `controllers/joint_controller.py:59` + `:358-364`; `controllers/osc_controller.py:94` + `:518-523` | **Coriolis / centrifugal compensation is ON by default** (`use_cc_compensation=True`) whenever impedances are in use | Isaac Lab adds **no** bias term anywhere in the actuator path — `Articulation.write_data_to_sim` (`articulation.py:184-220`) and `_apply_actuator_model` (`:1801-1838`) contain none, and the string `coriolis` does not appear anywhere under `isaaclab/`. Gravity compensation exists only in opt-in task-space controllers defaulting to `False` (`joint_impedance.py:32`, `operational_space_cfg.py:41`) | partly — the kwarg exists but defaults True, and nothing logs it | Active on every REALM run ever recorded (`DROID_robolab_v2.yaml:80` `use_cc_compensation: true`). **REALM's `use_gravity_compensation` / `use_cc_compensation` flags have no Isaac Lab counterpart** — they are an OmniGibson-side addition | no |
| 7 | I | `robots/robot.py:496-511` | Disables gravity on every link **not** rigidly fixed to the base, for all controllable objects | `disable_gravity` is a per-config spawn flag only (`schemas_cfg.py:77`), never applied globally; `disable_gravity()` is never called as a method. **But** RoboLab's own DROID sets `disable_gravity=True` for the *whole* robot (`droid.py:71`), which is stricter | yes | **Measured no-op here** — REALM already reports `disableGravity=True` on all nine gripper links. Deviation from Isaac defaults, **not** from the REALM/RoboLab reference — see benign note B4 | no |
| 8 | A, I | `prims/joint_prim.py:359-370` (`DEFAULT_MAX_EFFORT = 100.0`, `:21`) | `max_effort` getter returns **100.0** whenever the raw limit exceeds `INF_EFFORT_THRESHOLD = 1e10` | Isaac Lab reads back from the simulator deliberately (`articulation.py:1887`); `effort_limit=None` keeps the USD/PhysX value verbatim | **yes** | **This is not only a display sentinel — it becomes a real clamp**; see prose R8. Inert on REALM's current assets (all authored limits finite: arm 87/12 N·m, `finger_joint` 16.5 N·m) | no |
| 9 | A, I | `prims/joint_prim.py:330-344` (`:19-20`) | `max_velocity` getter returns **15.0 rad/s** (revolute) / **1.0 m/s** (prismatic) whenever the raw limit exceeds `INF_VEL_THRESHOLD = 1e5` | as row 8 | **yes** | Same clamp path as R8, on the velocity axis. Inert on REALM's DROID: `finger_joint` authors `physxJoint:maxJointVelocity = 500`, and no REALM controller is velocity-mode | no |
| 10 | A | `prims/joint_prim.py:460-513` (`DEFAULT_MAX_POS = 1000.0`, `:18`) | `lower_limit` / `upper_limit` return **∓1000.0** when the raw limit exceeds `INF_POS_THRESHOLD = 1e5` **or when `lower == upper`** | Isaac Lab reports the real limits | yes | The `lower == upper` branch is the sharp edge: a joint deliberately authored **locked** (identical limits) is reported to controllers as free over ±1000 rad. Not exercised by REALM's assets (checked: no locked revolute in the DROID chain) | no |
| 11 | I | `robots/robot.py:2454-2471` | `control_limits` is a `@cached_property`. Its own `TODO` at `:2454` says it is never invalidated when a joint limit changes | Isaac Lab re-reads limits from `ArticulationData` each step | yes | A runtime `joint.max_effort = x` never reaches any controller's clip or action-space scaling | no |
| 12 | I | `controllers/controller_base.py:167-168` | `_clip_lo` / `_clip_hi` are snapshotted **at controller construction** | as row 11 | yes | Even `del robot.control_limits` (which `robot.py:1296` does for holonomic bases) does not reach an already-built controller — that is why `:1302-1303` has to `reload_controllers()` outright | no |
| 13 | A | `robots/robot.py:1286-1293` | For holonomic-base robots only, **writes** the base joints' `max_velocity` to `MAX_LINEAR_VELOCITY = 1.5` / `MAX_ANGULAR_VELOCITY = π` (`:83-84`) and `max_effort` to `MAX_EFFORT = 1000.0` (`:85`), overwriting the asset | Isaac Lab writes limits only when the cfg names them | **yes** — comments call the asset values "too large"/"too small" without saying so at runtime | Not REALM: DROID is fixed-base, `is_holonomic_base` is False | no |
| 14 | A | `objects/dataset_object.py:318-330` | For **every rigid dataset object**, rewrites every prismatic/revolute drive: `type → "acceleration"`, `stiffness → 0.0`, `damping → 5.0` / `0.05` (`utils/constants.py:74-75`), `targetPosition → 0.0`, `targetVelocity → 0.0` | Isaac Lab leaves a spawned USD's drives alone unless a cfg names them | **yes** | Every drawer, door and cabinet in every REALM scene runs OmniGibson's damping, not the asset's, and on an `acceleration` drive (mass-normalised) rather than whatever the asset chose | no |
| 15 | A | `objects/dataset_object.py:238-243` | For **every dataset object**, writes `physxJoint:jointFriction` to `0.3` (prismatic) / `0.2` (revolute) (`utils/constants.py:70-71`) — ungated, applies whatever the `dataset_name` | Isaac Lab writes friction only when `ActuatorBaseCfg.friction` is set | **yes** | Same population as row 14. This is the only place OmniGibson writes joint friction at all (see benign B2) | no |
| 16 | I | `controllers/multi_finger_gripper_controller.py:230-242` | In `binary` mode the control is the joint **limit** itself — `should_open ? upper : lower` — not a graded target | a raw Isaac drive is commanded whatever target you write | partly — documented in the docstring, but the consequence is not | With row 1's kp = 1e7, "close" means "slam the target to the joint stop and hold it there with 1e7". REALM: `finger_joint` target ≥ 0 → **upper** limit 0.7854 rad = jaws shut (`DROID_robolab_v2.yaml:84-97`) | no |
| 17a | C | `robots/robot.py:674-677` (stock 3.9.1: `:658`) | Asserts that every DOF **not** claimed by a controller carries no DriveAPI | Isaac Lab only logs — `omni.log.warn` for unclaimed joints (`articulation.py:1725-1731`); they keep their authored drive untouched | **no** — it is a hard load failure | Forces REALM's converter to strip DriveAPI from the four mimic joints (`scripts/convert_robolab_gripper_usd.py`, `strip_mimic_drives`) for the asset to load at all | no |
| 17b | C | `robots/robot.py:658-672` | OG-lite adds an opt-in escape hatch: `REALM_ALLOW_DRIVEN_UNUSED_DOFS=1` downgrades 17a to a `log.warning` | — | no — it warns | Default path is byte-identical to upstream. Exists for one measurement only | **yes** (`a1ee0d2`) |
| 18 | C | `robots/robot.py:624-628` | Asserts every DOF a controller **does** claim **is** driven (has DriveAPI) | Isaac Lab matches actuators to joints by regex and tolerates undriven joints | no — hard failure | Together with 17a this is an **exact partition**: the set of DriveAPI joints must equal the set of controller-claimed joints. Neither a spare drive nor a missing one is loadable | no |
| 19 | C | `robots/robot.py:642-645` (`assert dof in unused_dofs`) | A DOF may be claimed by **at most one** controller | Isaac Lab asserts the same (an actuator group may not double-claim) | no | — | no |
| 20 | C | `prims/joint_prim.py:130-146` | Infers a joint's control type from its `(kp, kd)` at init and asserts all DOFs of one joint agree; a multi-DOF joint with mixed gains fails to load | Isaac Lab does not infer control mode from gains at all | no — hard failure | — | no |
| 21 | C | `prims/joint_prim.py:338, 355, 367, 381, 393, 406, 418, 431, 470, 490, 504, 524, 536, 550, 565` | `assert self.is_single_dof` on every drive/limit property. Reading or writing gains, effort, velocity, limits or axis on a **spherical or D6** joint raises | Isaac Lab handles multi-DOF joints in its articulation views | no — hard failure | Any asset that models a wrist or shoulder as a D6/spherical joint cannot be used as a robot here | no |
| 22 | I | `controllers/multi_finger_gripper_controller.py:250-257` | For `velocity`/`effort` motor types, zeroes any command that would push a finger further past a position limit already within `limit_tolerance` (default `0.001`, `robot.py:3477`) | no equivalent in Isaac Lab | **yes** | Not REALM (its gripper is position-mode) | no |
| 23 | I | `controllers/multi_finger_gripper_controller.py:117-119` | In `binary` mode, a user-supplied `command_output_limits` is **overwritten** with `"default"` before the super call | — | **yes** — silently discarded, no warning | Not REALM (it names no `command_output_limits` on the gripper) | no |
| 24 | I | `controllers/controller_view.py:297-323` | The controller-group identity hash is `sha256(repr(frozen_cfg))`; for a `torch.Tensor` config value `_freeze_for_hash` falls through to the value itself and `repr()` renders it at torch's **default 4-decimal print precision** | Isaac Lab has no cross-robot controller sharing | **yes** | **Inferred, not measured** (see R24). Two robots identical except for control limits differing below 1e-4 would share one controller group — and its `_clip_lo/_clip_hi` and `_isaac_kp/_isaac_kd` come from whichever registered first | no |
| 25 | — | `robots/robot.py:86` | `m.BASE_JOINT_CONTROLLER_POSITION_KP = 100.0` is defined and **never read** — zero references in the whole tree (`grep -rn BASE_JOINT_CONTROLLER_POSITION_KP` returns only the definition) | — | n/a | Dead macro. Listed so nobody hunts for its effect | no |
| 26 | I | `controllers/joint_controller.py:27,29`, applied `:127,130` | Runs its **own** impedance law on top of the PhysX drive, with `DEFAULT_JOINT_POS_KP = 50.0` and `DEFAULT_JOINT_VEL_KP = 2.0` | no equivalent — `ImplicitActuator.compute()` is a **pure pass-through** (`actuator_pd.py:115-140`), returning `control_action` unmodified | partly | OmniGibson interposes an extra control layer between the action and the PhysX drive. Live wherever `use_impedances=True` (row 5) | no |

## Prose where a row needs it

### R3 — the zero velocity target is what makes `kd` bite

`BaseController.step()` (`controller_base.py:476-482`) does, for POSITION control:

```python
ControllableObjectViewAPI.set_all_joint_position_targets(routing_path, enabled_rows, enabled_controls, self.dof_idx)
ControllableObjectViewAPI.set_all_joint_velocity_targets(routing_path, enabled_rows, enabled_controls * 0, self.dof_idx)
```

PhysX's implicit drive computes `F = kp·(q* − q) + kd·(q̇* − q̇)`. Pinning `q̇* = 0` every step turns
the `kd` term into pure velocity braking. Rows 1 and 3 therefore compound: on REALM's `finger_joint`
the asset asks for `kd = 0.0002` and gets `1e5`, applied against a hard-zero velocity target.
This is *not* separable from row 1 — fixing the gain without knowing about the zero velocity target
would mispredict the result.

Isaac Lab writes a zero velocity target the same way (`articulation.py:213-220`), so the *pinning* is
not the deviation; the `kd` it is multiplied by is.

### R8 — `DEFAULT_MAX_EFFORT` is a real clamp, not only a display sentinel

The established note calls `joint_prim.py:370` "a display sentinel, not a clamp". That is true of
the getter in isolation, but the value it returns is consumed as a control limit:

```
JointPrim.max_effort                       joint_prim.py:359-370   -> 100.0 if |raw| > 1e10
  -> EntityPrim.max_joint_efforts          entity_prim.py:1163-1169
  -> Robot.control_limits["effort"]        robot.py:2466-2471
  -> BaseController._control_limits         controller_base.py:140-143
  -> _clip_lo / _clip_hi                    controller_base.py:167-168
  -> BaseController.clip_control            controller_base.py:419-434   [hard clamp on commanded effort]
  -> _generate_default_command_output_limits controller_base.py:299-302  [action-space scaling]
```

So on any asset that authors an effectively-unlimited force limit (`FLT_MAX ≈ 3.4e38` is the usual
value an exporter writes for "no limit"), an EFFORT-mode controller has its commanded torques
clamped to **±100 N·m**, and a normalised action of `1.0` maps to 100 N·m rather than to the real
limit. There is no warning. The same argument applies to row 9 on the velocity axis (`clip_control`
for a VELOCITY controller, and `_normalize_velocities` at `entity_prim.py:778-800`).

**This is inert on REALM's current assets** and I want that stated plainly rather than implied: the
seven `panda_joint*` author `[87, 87, 87, 87, 12, 12, 12] N·m`, `finger_joint` authors
`maxForce = 16.5`, and the mimic followers author `0` — all far below `1e10`, so the sentinel never
fires. It is a live trap for any newly converted asset, not a current defect.

Note also (`controller_base.py:429-434`) that `clip_control` *undoes* the clip for
position-controlled joints flagged limitless, but has no such escape for the effort or velocity
axes.

**Isaac Lab has no equivalent software clamp**, which is what makes this a real divergence rather
than a difference in bookkeeping. `_clip_effort` (`actuator_base.py:355-364`) only fills the
*diagnostic* `computed_torque` / `applied_torque` buffers (`articulation.py:1832-1833`) for implicit
actuators; the real saturation is PhysX's own, via `effort_limit_sim` → `set_dof_max_forces`
(`articulation.py:794`).

### R14/R15 — dataset objects, and why they are in this audit

`dataset_object.py` is not in the nominal `controllers/robots/joint_prim` scope, but it is the only
other place in the tree that writes joint drive state, and it does so far more destructively than
anything in the robot path: it rewrites the authored drive of **every articulated dataset object**
in the scene. `_post_load` (`:318-330`) is gated only on `PrimType.RIGID`, not on `dataset_name`, so
it applies to custom assets too; `_initialize` (`:238-243`) is ungated entirely. If a task's
behaviour depends on how hard a drawer is to pull, that number comes from
`utils/constants.py:70-75`, not from the asset.

### R24 — the group-hash precision claim is inferred

`ControllerView._make_key` (`controller_view.py:316-323`) hashes `repr(frozen_cfg)`.
`_freeze_for_hash` (`:297-313`) tries `hash(value)` first; `torch.Tensor.__hash__` is identity-based
and therefore succeeds, so the tensor is returned as-is and `repr()` is what actually renders its
contents. Torch's documented default print options are `precision=4, threshold=1000`. **I did not
execute this** — no torch interpreter was reachable from the login node and I did not want to spend
a GPU allocation on it — so treat "4 decimal places" as read from torch's documented defaults, and
the collision consequence as reasoning from the code, not as a measurement. It becomes reachable
only with vectorised environments holding several robots of the same kinematic pattern, which is
new in this branch.

## Benign / by design — checked, do not re-check

- **B1. Mimic joints forced to `kp = kd = 0`** — `prims/joint_prim.py:179-182`, then `set_gains` at
  `:198-201`. Behaviourally equivalent: RoboLab's asset authors a **vestigial zero-gain** DriveAPI on
  those joints, so writing 0/0 changes nothing, and Isaac Lab would leave the same 0/0 in place. The
  motion comes from the PhysX *mimic constraint*, which neither stack touches. Not a deviation.
- **B2. OmniGibson never writes `physxJoint:armature`** — `grep -rn armature --include=*.py
  omnigibson/` returns **nothing**. The authored armature survives untouched. (Isaac Lab *does* write
  it when `ActuatorBaseCfg.armature` is set.) REALM writes armature itself, in
  `realm/environments/env_dynamic.py:248-264`, over `panda_joint1..7` only. Joint friction is
  likewise untouched on robots — the `JointPrim.friction` setter (`joint_prim.py:448-458`) has
  exactly one caller in the tree, and it is the dataset-object path of row 15.
- **B3. `set_control_type` never touches `max_effort`** — `joint_prim.py:164-204` writes gains only.
  The USD-authored force limit survives every gain override. This is why the measured Robotiq
  behaviour saturates at `maxForce = 16.5` regardless of what `isaac_kp` is set to, and why a
  stiffness-only A/B moved jaw penetration by <0.2 mm.
- **B4. Gravity disabling is milder than the reference config, not harsher, and is a measured
  no-op here** — row 7 disables gravity on non-base-fixed links; RoboLab's `droid.py:71` sets
  `disable_gravity=True` on the entire robot. Against RoboLab this is a *smaller* intervention. It
  is still a deviation from Isaac Lab's default (`None` → keep authored), so it stays in the table,
  but do not chase it as a REALM/RoboLab divergence. **Measured directly:** REALM already reports
  `disableGravity=True` on all nine gripper links, so the knob is a no-op. An earlier claim in this
  project of "RoboLab off / REALM on" was **wrong at runtime** and is retracted.
- **B5. Controller-group sharing does not leak gains between differently-configured robots** — the
  group key (`controller_view.py:316-323`) includes the full controller config, and `control_limits`
  is part of that config, so robots with genuinely different limits or gains land in different
  groups. Only the sub-1e-4 rounding case of row 24 is a concern.
- **B6. The control flush path applies no additional clamping** —
  `usd_utils.py:1513-1541` (`set_all_joint_*`) and `:1410-1424` (`flush_control`) are plain buffered
  writes into the articulation view. Everything that clamps does so earlier, in `clip_control`.
- **B7. `urdf_preprocessing.strip_mimic_joints`** (`utils/urdf_preprocessing.py:125-136`) has **zero
  callers**. Dead code; it does not destroy mimic constraints at import.
- **B8. `EntityPrim.set_joint_positions` docstring overclaims** — `entity_prim.py:636-637` says "both
  actual value and target values", but `:660-666` writes one *or* the other depending on `drive`.
  Harmless for robots, because `Robot.set_joint_positions` (`robot.py:996-1005`) resets the
  controllers whenever `drive=False`, so the next step recomputes a goal from the new state. Worth
  knowing before trusting the docstring on a non-robot articulation.
- **B9. Isaac Lab's effort limit is not an extra software clamp either** — see the closing paragraph
  of R8. Do not model it as one when comparing the two stacks.

## Traps that are not deviations, but will mislead you

1. **`stiffness` and `damping` default to `MISSING`, not `None`**, in `ActuatorBaseCfg`
   (`actuator_cfg.py:112`, `:122`). You must write `stiffness=None` explicitly to get USD-keeping
   behaviour — which is exactly what RoboLab's gripper does (`droid.py:117-125`).
2. **`velocity_limit` on an implicit actuator is silently discarded** (`actuator_pd.py:79-89` sets
   `cfg.velocity_limit = None` after a warning). RoboLab's `velocity_limit=5.0` on the gripper and
   `2.175`/`2.61` on the arm therefore **never took effect**; the USD `maxJointVelocity` is what
   applies. Use `velocity_limit_sim` to actually set it.
3. **Unit convention.** PhysX tensor-API getters return SI/radian; USD authors angular drives in
   degrees. Isaac Lab converts only on the USD *write* path (`schemas.py:619-628`). This is why
   authored stiffness 100 reads back as 5729.578 (= 100 × 180/π), and why an OmniGibson-vs-Isaac-Lab
   gain ratio quoted without reconciling units is wrong. An earlier "1e5× stiffer" figure in this
   project was exactly that error; the reconciled figure is **~1745×**.
4. **Aliasing** (`actuator_base.py:337-338`): no `.clone()`, and `.float()` on an already-float32
   tensor returns `self`. If an actuator group covers all joints, `actuator.stiffness` **aliases**
   `articulation.data.default_joint_stiffness`.
5. **A zero velocity target is written every step** alongside the position target
   (`articulation.py:213-220`), and `_data.joint_vel_target` is sticky — nothing clears it on reset.
   Both stacks do this; see R3.

## Explicitly not covered (other lanes)

Solver iteration counts (`objects/usd_object.py:64-65`, 32 position / 1 velocity, against RoboLab's
64 / 0 at `droid.py:76-77`) change actuation *fidelity* but are a physics-solver setting, not drive
state — flagged here as a pointer only, and covered in
[`rigid_bodies_and_collision.md`](rigid_bodies_and_collision.md) row 9 and
[`simulator_and_scene.md`](simulator_and_scene.md) row 8. Assisted-grasping constraint creation
(`robot.py:851-920`) can override actuation by welding a joint, but REALM runs
`grasping_mode="physical"`, so it never fires; it belongs to a grasping audit.

Also unresolved and outside this lane: `grep 'frame="parent"'` finds **three** sites composing to the
wrong level; only `rigid_prim.py:324` is patched. `geom_prim.py:250` (collision hull) and
`object_utils.py:88` (`compute_base_aligned_bboxes`) are not — see
[`transforms_and_assets.md`](transforms_and_assets.md) and `CHANGE_LEDGER.md`.
