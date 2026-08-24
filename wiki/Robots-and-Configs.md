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
| `DROID` | `CustomJointController` | **The baseline.** Task-space-weighted joint PD on the mounted RoboLab v2 asset. |
| `DROID_default_pd_control` | stock OmniGibson `JointController` | Swaps REALM's controller for OmniGibson's own, and zeroes friction and armature. No task-space gains, no effort limits. A reference point, not a better config. |
| `DROID_polaris_control` | `IndividualJointPDController` | Scalar-gain PD. **Orphaned** — nothing in the repo references it. |
| `DROID_ee_control` | `DroidEndEffectorController` | End-effector control: the policy commands an **absolute** 6-DOF pose, solved by IK. |
| `DROID_ee_delta_control` | `DroidEndEffectorController` | The same, but the policy commands 6-DOF pose **deltas**. |
| `DROID_no_wrist_cam` | `CustomJointController` | See the caveat below — the name is ahead of what the file does. |
| `DROID_robolab_v2` | `CustomJointController` | Explicit alias for the same canonical asset and controller as `DROID`. |
| `DROID_robolab_v2_ee_control` | `DroidEndEffectorController` | `DROID_robolab_v2` with end-effector control. Carries a required height-offset override. |

### Canonical DROID embodiment

All DROID profiles use the mounted RoboLab v2 asset: a 7-DOF Franka arm plus the compliant Robotiq
four-bar linkage represented by one actuated joint and five mimic followers (13 total DOFs).

**Every REALM robot needs registering before it will load** — including `DROID`. OmniGibson 3.9.1
discovers robots by globbing the dataset directory for `<data>/*/models/<name>/<name>.yaml`, and
REALM's definitions live in the repo, linked in by symlinks that are **not tracked in git**. Run
`scripts/install_robot_definitions.py`; it installs every definition in one pass and exits on the first
failure, so the state is all-or-nothing. On a machine where it has not been run, **none** are
registered.

The user-facing `DROID` and explicit `DROID_robolab_v2` filenames resolve to the mounted model;
`DROID_robolab_v2_bare` retains the same arm and gripper without the base column.

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

A **definition** is the YAML OmniGibson discovers; a **config** is what REALM passes `--robot`.
REALM 1.0.0 has mounted and unmounted definitions of the same RoboLab v2 embodiment:

| Definition | Used by |
|---|---|
| `droid_robolab_v2` | every default/mounted `DROID*` robot config |
| `droid_robolab_v2_bare` | explicit `DROID_robolab_v2_bare` no-column variant |
| `ur` | all three `UR5*` configs |

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

The EE controller adds the mounted arm-base height to commanded z. Every EE profile explicitly uses
the measured RoboLab v2 offset, `0.863891` m.

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
