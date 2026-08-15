# OmniGibson 3.9.1 vs Isaac Lab 2.2.0 — control and actuation

What OmniGibson does to a robot's actuation that raw Isaac Sim / Isaac Lab does not. Every row cites
`file:line`. OmniGibson paths are relative to `/mnt/home_lustre/sedlam56/projects/OG-lite_og391`;
Isaac Lab paths are in-SIF absolute under `/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif`
(`VERSION` = 2.2.0).

**Status:** the Isaac Lab side is a complete, verified extraction. The OmniGibson side covers what this
project measured directly and is **not exhaustive** — three sibling audits (rigid bodies, simulator,
transforms) were killed by an API session limit before reporting. Treat this as the control-domain
chapter of a reference that is still missing three chapters.

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
damping **0.0002**; OmniGibson runs it at **1e7 / 1e5**. In matched units that is ~1745× stiffer and
about seven orders more damped. Isaac Lab, given `stiffness=None, damping=None`, ran the same joint at
the authored values (5729.578 / 0.011459 in per-radian runtime units).

---

## Table

| # | site | what OmniGibson does | what Isaac Lab does / what the asset authored | silent? | class | impact |
|---|---|---|---|---|---|---|
| 1 | `controllers/controller_base.py:16-17`, `:206-207` | substitutes `DEFAULT_ISAAC_KP = 1e7`, `DEFAULT_ISAAC_KD = 1e5` when the config names no gains | keeps the USD-authored drive (`actuator_base.py:165-168`, `:317-331`) | **yes** | override of authored values | **measured**: 1745× on `finger_joint`; the direct cause of a rigid gripper |
| 2 | `controllers/controller_base.py:213` | in the **VELOCITY** branch, `isaac_kd` falls back to `DEFAULT_ISAAC_KP` (1e7), not `DEFAULT_ISAAC_KD` (1e5) | n/a | **yes** | likely upstream bug | unmeasured; affects velocity-controlled joints only |
| 3 | `simulator.py:1382` (in the `was_stopped` branch) | re-calls `robot.update_controller_mode()`, `robot.reset()`, `robot.keep_still()` on **every** `play()` after a `stop()` | gains are applied **once**, in `_initialize_impl` on the timeline PLAY event, guarded by `_is_initialized` (`asset_base.py:287-326`); **`reset()` does not re-apply them** (`articulation.py:172-182`; implicit `reset` is a no-op, `actuator_pd.py:111-113`) | **yes** | lifecycle divergence | a runtime gain change survives `env.reset()` in Isaac Lab but is **wiped by any stop/play** in OmniGibson |
| 4 | `robots/robot.py:675` | **asserts** that any joint not claimed by a controller carries no DriveAPI | only an `omni.log.warn` for unclaimed joints (`articulation.py:1725-1731`); unclaimed joints keep their USD drive untouched | no — it aborts | **constraint imposed on assets** | forced REALM's converter to strip DriveAPI from mimic joints for the asset to load at all |
| 5 | `prims/joint_prim.py:370` | `max_effort` getter returns `DEFAULT_MAX_EFFORT = 100.0` whenever `\|raw\| > INF_EFFORT_THRESHOLD` | reads back from the simulator deliberately (`articulation.py:1887`) | **yes** | diagnostic sentinel | reading a force limit reports **100** where the real value is FLT_MAX — cost one agent a wrong conclusion |
| 6 | `robots/robot.py:495-510` | disables gravity on every link not fixed to the base, for all controllable objects | `disable_gravity` is a per-config spawn flag only (`schemas_cfg.py:77`), never applied globally; `disable_gravity()` is never called as a method | **yes** | deviation from Isaac defaults | **measured no-op here** — REALM already matched RoboLab, which sets it per-robot |
| 7 | `controllers/joint_controller.py:27,29`, `:127,130` | `DEFAULT_JOINT_POS_KP = 50.0`, `DEFAULT_JOINT_VEL_KP = 2.0` for its own impedance law | no equivalent — `ImplicitActuator.compute()` is a **pure pass-through** (`actuator_pd.py:115-140`), returning `control_action` unmodified | partly | extra control layer | OmniGibson interposes its own controller between action and PhysX drive |
| 8 | `prims/joint_prim.py:179-201` | forces `kp = kd = 0` on mimic joints, then calls `set_gains()` | leaves the authored 0/0 | n/a | **benign** — equivalent end state | none |

---

## Benign / by design — checked, do not re-check

- **Row 8**, mimic joints at zero gain: both stacks end at 0/0. Not a deviation.
- **Row 6**, gravity: measured directly — REALM already reports `disableGravity=True` on all nine
  gripper links, so the knob was a no-op. The earlier claim "RoboLab off / REALM on" was **wrong at
  runtime** and is retracted.
- **Effort clamping.** Isaac Lab's `_clip_effort` (`actuator_base.py:355-364`) only fills the
  *diagnostic* `computed_torque`/`applied_torque` buffers (`articulation.py:1832-1833`) for implicit
  actuators; real saturation is PhysX's, via `effort_limit_sim` → `set_dof_max_forces`
  (`articulation.py:794`). So Isaac Lab's effort limit is not an extra software clamp either.
- **No gravity or Coriolis compensation** in Isaac Lab's actuator path. `Articulation.write_data_to_sim`
  (`articulation.py:184-220`) and `_apply_actuator_model` (`:1801-1838`) contain no bias term, and the
  string `coriolis` does not appear anywhere under `isaaclab/`. Gravity compensation exists only in
  opt-in task-space controllers defaulting to `False` (`joint_impedance.py:32`,
  `operational_space_cfg.py:41`). **REALM's `use_gravity_compensation` / `use_cc_compensation` flags
  therefore have no Isaac Lab counterpart** — they are an OmniGibson-side addition.

## Traps that are not deviations, but will mislead you

- **`stiffness` and `damping` default to `MISSING`, not `None`**, in `ActuatorBaseCfg`
  (`actuator_cfg.py:112`, `:122`). You must write `stiffness=None` explicitly to get USD-keeping
  behaviour — which is exactly what RoboLab's gripper does (`droid.py:117-125`).
- **`velocity_limit` on an implicit actuator is silently discarded** (`actuator_pd.py:79-89` sets
  `cfg.velocity_limit = None` after a warning). RoboLab's `velocity_limit=5.0` on the gripper and
  `2.175`/`2.61` on the arm therefore **never took effect**; the USD `maxJointVelocity` is what
  applies. Use `velocity_limit_sim` to actually set it.
- **Unit convention.** PhysX tensor-API getters return SI/radian; USD authors angular drives in
  degrees. Isaac Lab converts only on the USD *write* path (`schemas.py:619-628`). This is why
  authored stiffness 100 reads back as 5729.578 (= 100 × 180/π), and why an OmniGibson-vs-Isaac-Lab
  gain ratio quoted without reconciling units is wrong. An earlier "1e5× stiffer" figure in this
  project was exactly that error; the reconciled figure is ~1745×.
- **Aliasing** (`actuator_base.py:337-338`): no `.clone()`, and `.float()` on an already-float32
  tensor returns `self`. If an actuator group covers all joints, `actuator.stiffness` **aliases**
  `articulation.data.default_joint_stiffness`.
- **A zero velocity target is written every step** alongside the position target
  (`articulation.py:213-220`), and `_data.joint_vel_target` is sticky — nothing clears it on reset.

---

## Not covered

Three sibling audits were killed by an API session limit before reporting, so these domains are
**absent**, not clean:

- **rigid bodies, mass, collision, materials** — though the worst finding in the whole project lives
  here: `prims/rigid_prim.py` composed collision-geom CoM to the geom's *immediate parent* rather than
  the link, inflating pad inertia **77×**. See `CHANGE_LEDGER.md`.
- **simulator, physics scene, lifecycle**
- **transforms, articulation state, asset import**

Also unresolved: `grep 'frame="parent"'` finds **three** sites composing to the wrong level;
only `rigid_prim.py:324` is patched. `geom_prim.py:250` (collision hull) and `object_utils.py:88`
(`compute_base_aligned_bboxes`, which feeds live REALM perturbation code) are not.
