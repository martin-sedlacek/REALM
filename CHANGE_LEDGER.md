# Change ledger — what was changed, how far it reaches, and how to undo it

Written 2026-08-15. Purpose: make every change from the vectorization + gripper-compliance work
**revertable individually**, and say honestly which ones can affect things beyond their intended
target. Read the "blast radius" column before assuming a change is local.

Session range: REALM `ecff61f..ab8d31d` (123 commits), OG-lite `e30899f..83b21d5` (13 commits).

---

## 1. OG-lite — the risky ones, because they are engine-wide

OG-lite is bound over the image's OmniGibson only when `MODE=oglite`. `rr` defaults to `MODE=stock`,
so **none of these affect a default run** unless the bind is used or the fix is carried into the
image as a patch.

| commit | what | blast radius | revert |
| --- | --- | --- | --- |
| `83b21d5` | `rigid_prim`: compose collision-geom CoM to the **link** frame, not the geom's immediate parent | **⚠ SEE BELOW — wider than intended** | `git revert 83b21d5` |
| `27806cb` | `transform_utils`: 5 buffers built on CPU, filled from device-bound inputs | engine-wide, **CPU path byte-identical** (audited) | `git revert 27806cb` |
| `eaba43e` | 2 device sites, robot load + proprioception | same | `git revert eaba43e` |
| `efe0f72` | contact-API index/mask tensors, joint-limit properties | same | `git revert efe0f72` |
| `1ccfb2a` | 3 device coercions, serialize + compute-backend | same | `git revert 1ccfb2a` |
| `a1ee0d2` | a device mismatch + **opt-in** escape hatch for the mimic-DriveAPI experiment | same; hatch is off by default | `git revert a1ee0d2` |
| `43c3c7d` | 2 device coercions (`serialize`, `set_position_orientation`) | same | `git revert 43c3c7d` |
| `ec7373b` | match an exported asset's up axis to the stage's | **any asset whose `upAxis` differs from the stage** | `git revert ec7373b` |
| `7c59ed5` | place an articulation by its ROOT LINK, not whichever prim was read | **any articulation whose root link is offset from its entity prim** | `git revert 7c59ed5` |
| `1dcc5bb` | `OmniSurfaceMaterialPrim.preset_name` default | assets using that material; without it they do not load | `git revert 1dcc5bb` |
| `59af7c0` | prune the object-init queue by **identity**, not name | multi-scene runs only; strictly narrower than the old test | `git revert 59af7c0` |
| `0eba7e7` | empty contact index tensors, report unqueryable bodies, larger descriptor pool | contact queries generally | `git revert 0eba7e7` |
| `ef7442b` | scene-file objects loading 100 m too high in scenes `idx != 0` | multi-scene runs only | `git revert ef7442b` |

### ⚠ `83b21d5` reaches further than "the gripper"

It is scoped by a **frozenset of link NAMES**, not by asset or robot:

```
base_link, left_outer_knuckle, right_outer_knuckle, left_outer_finger, right_outer_finger,
left_inner_finger, right_inner_finger, left_inner_knuckle, right_inner_knuckle
```

Eight of those are Robotiq-specific and effectively unique. **`base_link` is not** — it is one of the
commonest link names in the BEHAVIOR dataset. So any object with a link called `base_link` **whose
collision geometry sits under an intermediate Xform** now gets a different (corrected) centre of mass,
and therefore different inertia and different dynamics.

The correction is a genuine bug fix, so the new value is the *right* one — but it is still a silent
behaviour change for objects nobody measured, and it invalidates comparison against any result
collected before it.

**If odd dynamics show up on non-gripper objects, this is the first thing to revert.** Narrowing it —
e.g. keying on the robot/prim path rather than the bare link name, or dropping `base_link` from the
set — is cheaper than reverting and is the recommended fix if that happens.

**Not yet carried into the stock image.** There are seven `realm/misc/` patches applied by both
`.docker/realm_og391.def` and `.docker/realm_og391.Dockerfile`; `83b21d5` is **not** among them, so a
rebuilt image will not contain it.

---

## 2. REALM — shipped behaviour

### Opt-in, zero effect unless selected by name

These add new robot configs and variant USDs. Nothing selects them unless a run names them, so they
cannot perturb existing results. Deleting the files is a complete revert.

- `DROID_robolab_curlgrip{,_ee_control}.yaml` + `droid_robolab_curlgrip.usd` + its definition —
  **`nf=200`**. Chosen against the 18-26x compliance gap, which is now understood to be a *symptom* of
  the CoM bug. **Likely to be retired**: with `83b21d5` in, it double-compensates.
- `DROID_robolab_padspring*.yaml` (6 configs) + `droid_robolab_padspring.usd` + definition — the
  pad-pivot spring route. Its supporting evidence was measured on a **closed-jaw press**, which is the
  wrong load case; the code stands, the numbers do not.
- One additive symlink under `data/` via the documented `scripts/install_robot_definitions.py`
  mechanism. No existing file modified.

### Changes the default path

| area | effect | note |
| --- | --- | --- |
| `realm/environments/perturbations/*` | all 16 perturbations vectorized | single-env behaviour intended to be unchanged; that was the design constraint throughout |
| `env_vector.py`, `env_base.py`, `env_dynamic.py` | phased vector reset, batched joint resets | as above |
| `t9_vbpose_nostopplay.py` | `DRAWER_Z_MIN = 0.2` for drawer tasks; step budget allows one shared joint-reset loop | harness only; task 0 verified bit-identical |
| `realm/config/scenes/scenes.yaml`, three task YAMLs | scene/task config | check against 1.1.1 before trusting a comparison |
| `DROID_robolab_v2.yaml` gripper block | **unchanged** — no gains were ever added here | the `isaac_kp`/`isaac_kd` work was all probe-side |

**Arm physics was held byte-identical throughout and verified, not assumed** — `arm_0` controller
block, top-level `friction`/`armature`, and the seven `panda_joint*` DOFs, across 133 authored
attributes and 25 link prims (`CURLGRIP_ARM_IDENTICAL`, `SHIP_ARM_IDENTITY_OK`,
`scripts/debug_probes/ship_arm_identity.py`).

### Probe-only — no runtime effect

Everything under `scripts/debug_probes/` and `scripts/clara/`. The paths refactor
(`scripts/clara/lib/paths.sh` + 18 scripts) changes only how harness scripts resolve their own root.

---

## 3. Results that should NOT be compared across this work

- **Task 8 `open_drawer` before the upAxis fix.** The cabinet was placed lying on its back, so
  `init_openness_fraction` started at 0.62 instead of 0. Two rubric stages are *absolute*
  (`> 0.125`, `> 0.65`), so they scored partly for free. Any pre-fix task-8 number is inflated.
- **Anything measured on the closed-jaw press.** The probe closed the jaw before descending
  (`GRIP_CLOSE = 1.0`), which loads the pads along the linkage's stiff axis. Different experiment.
- **Anything using `d_tip_sep` / `d_base_sep`.** Hull-derived tip positions are invalid on this asset
  — the hull tip sits 116 mm from the pad link origin, the same CoM bug expressed differently — and
  the sign reads **backwards**. Superseded by `d_tipg_sep` (hull-free).
- **Non-gripper objects with a `base_link`, after `83b21d5`.** See the warning above.

---

## 4. Fastest full rollback

```sh
# engine-side (also just: stop using MODE=oglite)
cd ~/projects/OG-lite_og391 && git revert --no-commit e30899f..HEAD && git commit

# REALM shipped behaviour, keeping the harness and probes
cd ~/projects/REALM_og391 && git revert --no-commit <commit> && git commit
```

Reverting OG-lite alone restores stock engine behaviour, since REALM's own changes do not depend on
the OG-lite fixes except where noted (the drawer tasks need `ec7373b`; multi-scene runs need
`ef7442b` and `59af7c0`).
