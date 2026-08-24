# Robots and configs

`--robot <name>` selects a YAML from `realm/config/robots/`. The name you pass is the **filename
stem** — `--robot DROID_robolab_v2` loads `DROID_robolab_v2.yaml`.

**The default is `DROID`.** That default is set independently in the two eval entry points, the
environment constructor, and both `evaluate` functions, and the
`scripts/eval.sh` wrapper does not override it. If you pass no `--robot`, you get `DROID`.

> The `name:` **inside** the file is a separate thing: it is the key observations are built against,
> and some configs deliberately carry a `name:` that differs from their filename so they share an
> observation profile. Changing one does not change the other.

## The DROID family

Franka Panda arm with a Robotiq 2F-85 gripper. This is what the benchmark runs on.

| `--robot` | Arm controller | What distinguishes it |
|---|---|---|
| `DROID` | `CustomJointController` | **The baseline.** Task-space-weighted joint PD, non-zero joint friction and armature, effort limits, and an explicit wrist-camera aperture. The asset (mounted vs unmounted) is resolved at runtime from the task suite. |
| `DROID_default_pd_control` | stock OmniGibson `JointController` | Swaps REALM's controller for OmniGibson's own, and zeroes friction and armature. No task-space gains, no effort limits. A reference point, not a better config. |
| `DROID_polaris_control` | `IndividualJointPDController` | Scalar-gain PD. **Orphaned** — nothing in the repo references it. |
| `DROID_ee_control` | `DroidEndEffectorController` | End-effector control: the policy commands an **absolute** 6-DOF pose, solved by IK. |
| `DROID_ee_delta_control` | `DroidEndEffectorController` | The same, but the policy commands 6-DOF pose **deltas**. |
| `DROID_no_wrist_cam` | `CustomJointController` | See the caveat below — the name is ahead of what the file does. |
| `DROID_robolab` | `CustomJointController` | The RoboLab gripper asset, v1. |
| `DROID_robolab_v2` | `CustomJointController` | The RoboLab gripper asset, v2. Opt-in; see below. |
| `DROID_robolab_v2_ee_control` | `DroidEndEffectorController` | `DROID_robolab_v2` with end-effector control. Carries a required height-offset override. |

### `DROID` vs `DROID_robolab_v2`

**Same arm, different gripper physics.** The stock asset models the 2F-85 with extra prismatic joints
that make the fingers effectively rigid. The RoboLab asset models the real four-bar linkage as one
actuated revolute joint plus five mimic followers. That changes DOF count (11 vs 13), the
gripper-open/closed conventions used to normalise observations, the base column, and which camera
prims exist.

**Every REALM robot needs registering before it will load** — including `DROID`. OmniGibson 3.9.1
discovers robots by globbing the dataset directory for `<data>/*/models/<name>/<name>.yaml`, and
REALM's definitions live in the repo, linked in by symlinks that are **not tracked in git**. Run
`scripts/install_robot_definitions.py`; it installs all five in one pass and exits on the first
failure, so the state is all-or-nothing. On a machine where it has not been run, **none** are
registered.

Internal performance and integrity tooling generally defaults to `DROID_robolab_v2`, which is why you
will see it all over `scripts/clara/` and the debug probes — including the batch launcher, whose
`ROBOT` default is `DROID_robolab_v2` rather than the `DROID` this page documents. The shipped
user-facing default is still `DROID`.

> `DROID_robolab` and `DROID_robolab_v2` differ only in the USD they point at. v2 exists so the two
> can be A/B compared without disturbing v1.

### Caveat: `DROID_no_wrist_cam` does not remove the wrist camera

Read the file before relying on the name. It contains no sensor include/exclude filter and does not
change `obs_modalities` — the only differences from `DROID.yaml` are an omitted vision-sensor block
and omitted effort limits. Omitting the vision-sensor block means the wrist camera falls back to
OmniGibson's default aperture, which **widens** the wrist view to roughly 150° rather than removing
it. Some downstream code nevertheless treats this config as camera-less. Treat this as a known
inconsistency, not a working option.

## Other platforms

Present, but not what the benchmark is built or validated on.

| `--robot` | Notes |
|---|---|
| `UR5` | UR5e with a Robotiq gripper. `proprio` observations only — **no RGB**. Its definition carries a header warning that it is a best-effort port and is not exercised by the smoke test. |
| `UR5_default_pd_control`, `UR5_aligned_pd_control` | Gain variants of the above, with friction and armature zeroed. |
| `WidowX` | Stock OmniGibson robot, stock controllers, `proprio` only. No REALM definition exists for it. |

## Configs named in scripts but absent from the tree

These appear as defaults in debug probes and in the change ledger, but the YAML files are **not in
the repo**: `DROID_robolab_xflat`, `DROID_robolab_curlgrip`, `DROID_robolab_curlgrip_ee_control`,
`DROID_robolab_padspring`. They were removed when the bendy-gripper investigation was reverted — see
[Gripper compliance findings](Gripper-Compliance-Findings). The probes that default to them will fail
on their own defaults. Do not treat them as available.

## Robot definitions and assets

A **definition** is the YAML OmniGibson discovers; a **config** is what REALM passes `--robot`. Five
definitions live in `realm/robots/definitions/`:

| Definition | Used by |
|---|---|
| `droid` | `type: DROID` configs on non-`REALM_DROID10` task suites |
| `droid_mounted` | `type: DROID` configs on the `REALM_DROID10` benchmark suite |
| `droid_robolab` | `DROID_robolab` |
| `droid_robolab_v2` | `DROID_robolab_v2`, `DROID_robolab_v2_ee_control` |
| `ur` | all three `UR5*` configs |

`droid` and `droid_mounted` are otherwise identical — same joints, same default pose.

USD assets live in `realm/robots/panda_robotiq/` and `realm/robots/ur5/` and are committed to the
repo, not fetched. `robolab_franka_robotiq_2f_85_flattened.usd` is the **upstream source** that
The committed `droid_robolab*` assets were converted from the RoboLab vendor source; do not point a
definition at it directly — the definitions say so explicitly.

Definitions reference their USDs by absolute `/app/...` paths, which resolve only when the repo is
bind-mounted at `/app` inside the container.

## Controllers REALM adds

Registered in `realm/robots/controller_registry.py`:

| Registered name | Purpose |
|---|---|
| `CustomJointController` | Task-space-weighted joint PD — the gains combine a joint-space and a Jacobian-projected task-space term. Forced to effort mode. This is what `DROID` uses. |
| `IndividualJointPDController` | Plain scalar-gain joint PD. |
| `DroidEndEffectorController` | Cartesian end-effector control; converts 6-DOF pose commands to joint targets through an IK solver, with capped per-step deltas. |
| `CustomGripperController` | Subclass of OmniGibson's gripper controller. Overrides the opening behaviour so the outer finger joints are driven from *measured* inner-finger positions rather than being slammed to the joint limit. |

Two gotchas, both documented in that module:

- **Two different classes share the class name `IndividualJointPDController`.** They are told apart
  only by the registered alias. `CustomJointController` is *not* the one named
  `IndividualJointPDController`.
- Importing the registry **rebinds OmniGibson's stock gripper controller** to REALM's subclass as a
  side effect of auto-registration.

### End-effector control: the height offset

The EE controller adds a fixed height offset to the commanded z, defaulting to a value correct for
the *mounted* DROID asset, whose base link sits well above the floor. The RoboLab asset's base link
is at z = 0, so `DROID_robolab_v2_ee_control` must override the offset to `0.0`. Getting this wrong
makes the arm stretch upward and diverge — and the config notes that the vectorized integrity test
does not catch it.

## Cameras

**External cameras are task-driven, not robot-driven.** Two scene cameras are defined in
`realm/config/env/external_sensors/camera_config.yaml`, with a named pose library in
`camera_extrinsics.yaml`. Which pose is used comes from the task config; the second camera is deleted
entirely unless `--multi-view` is passed. The poses are resolved relative to the robot's spawn pose,
so the robot affects *where* they end up but not *which* config is chosen.

**The wrist camera is robot-driven, and three things must stay in sync:**

1. the robot config's sensor include-filter, which decides which camera prims exist;
2. the observation profile in `realm/inference/utils.py`, keyed by the config's `name:`;
3. an assertion run at environment build time that fails if the camera index resolves to a different
   prim than the profile expects.

The wrist-camera index is a **creation-order** index, not a property of the asset — filtering a
camera out renumbers the survivors. A robot with no observation-profile entry silently falls back to
the `DROID` profile. The filter must be an *include* list, because sensor names are matched by
substring and excluding `wrist_camera` would also remove `wrist_camera_flipped`.

`UR5*` and `WidowX` declare proprioception only, have no observation profile, and the wrist-camera
assertion deliberately passes them through.

## See also

- [Running evaluations](Running-Evaluations) — where `--robot` fits in the flag surface
- [Tasks and perturbations](Tasks-and-Perturbations)
- [OmniGibson deviations](OmniGibson-Deviations) — why the gripper physics differ between assets
