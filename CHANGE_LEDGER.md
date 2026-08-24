# Change ledger — what was changed, how far it reaches, and how to undo it

Written 2026-08-15. Purpose: make every change from the vectorization + gripper-compliance work
**revertable individually**, and say honestly which ones can affect things beyond their intended
target. Read the "blast radius" column before assuming a change is local.

Session range: REALM `ecff61f..ab8d31d` (123 commits), OG-lite `e30899f..83b21d5` (13 commits).

---

## 1.0.0 — first major release: the restructure + the number-moving batch (2026-08-19)

1.0.0 is the release boundary for the whole 2026-08-18/19 body of work, owner's call: everything
from `b9a6fb3` onward falls under it. The bulk is **behaviour-preserving by construction** and
verified with old-vs-new equivalence harnesses (bit-identical outputs incl. RNG stream state):
the perturbations decomposition, Phase-0 dead-code deletion, robots fail-fast + IK documentation,
inference-client adapters, logging dedup, `realm/paths.py`, the upstream drawer-fix merge, the
test-driver dedup, and the CLAUDE.md/wiki realignment. None of that moves a number.

**The three fixes below DO CHANGE BENCHMARK SCORES**, and they are why the major version turns.
They had been flagged `KNOWN ISSUE` in place and gated since the behaviour-preserving cleanup
passes; the owner's call on 2026-08-19 was to fix them and recompute rather than preserve. Every
B-HOBJ cell, every push-task VB-POSE cell, and every V-AUG cell recorded before 1.0.0 is **not
comparable** to a 1.0.0+ run and must be recomputed. All three ship in one commit under this
VERSION so "which semantics produced this number" is answerable from the tag alone.

| where | what changed | blast radius | revert |
| --- | --- | --- | --- |
| `b_hobj.py` | The six log-uniform factors are now all APPLIED: mass is scaled by `s_mass` (was: an unrelated `U(0.25, 3)` draw), and the previously discarded `s_mvel` / `s_fric` now scale `joint.max_velocity` / `joint.friction` (setters verified present in OmniGibson 3.9.1 `prims/joint_prim.py`). The unrelated uniform draw is REMOVED, shifting the shared RNG stream. Baselines snapshot all five joint properties. | every B-HOBJ cell: different mass distribution (log-uniform ~[0.37, 2.72] vs uniform [0.25, 3]), joints now also vary max-velocity and friction, and every RNG draw after b_hobj's shifts | revert the 1.0.0 commit |
| `vb_pose.py::_perturb_switch` | The nudge no longer aliases `env.init_poses[...]["pos"]` — it offsets a `clone()` — so the switch offset no longer COMPOUNDS across resets: repeat N perturbs from the authored pose, not from wherever repeat N-1 left it. `init_poses` is no longer mutated by VB-POSE. | push-task (task 7) VB-POSE cells only; repeat 1 unchanged, repeats 2+ see smaller (non-accumulated) offsets | revert the 1.0.0 commit |
| `v_aug.py` + `env_dynamic.py` | One canonical draw range, `SIGMA_RANGE=(0, 2.5)` / `ALPHA_RANGE=(0.25, 1.5)` in `v_aug.py`, imported by the per-reset draw. Chosen because it is the only range that ever reached a rendered observation — the construction-time draw (`0-3.0` / `0.5-2.0`) was always overwritten before the first distortion. That dead construction-time draw is REMOVED, shifting the shared RNG stream for V-AUG-active runs. | every V-AUG cell: identical distortion distribution, but different concrete draws (stream shift) | revert the 1.0.0 commit |

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
| `bf1e416` + `b90febe` | `droid_robolab_v2_mounted.usd`: drop a duplicate articulation root on `/panda/table`, repoint `panda_table_joint` off two dangling body targets | **that asset only** — 2 changed prim specs of 1031, no attribute/mass/pose touched; it did not construct at all before | `git checkout 6154f19 -- realm/robots/panda_robotiq/droid_robolab_v2_mounted.usd` |

### `83b21d5` over-reached, and has been narrowed — `6d04cc9`, `0fed598`

**Original risk (now closed).** It was scoped by a frozenset of link **names**, one of which was
`base_link` — among the commonest link names in the BEHAVIOR dataset. Any object with a `base_link`
whose collision geometry sat under an intermediate Xform would have had its centre of mass, inertia
and dynamics silently changed.

**Narrowed in two steps:**

- `6d04cc9` — gates on gripper **structure**, not the name: a link qualifies only if its parent also
  carries `left_inner_finger` **and** `right_inner_finger`. No BEHAVIOR object has that, and it cannot
  be defeated by a scene reusing a link name.
- `0fed598` — drops `base_link` from the set entirely, so **no generic name appears in it at all**.
  Eight Robotiq-specific names remain.

**What is measured, and what is not** — an earlier version of this file said dropping `base_link` cost
nothing "measured, not assumed". That was **wrong and is retracted**: the number was quoted from a run
still in startup.

- **Measured:** the name gate against the structural gate — all nine CoMs bit-identical, `nf_eq` delta
  exactly `0.00e+00`, curl identical at +0.3280° / +0.3887°. So `6d04cc9` is verified inert.
- **Measured, and it is NOT zero:** the `base_link`-dropped variant (`0fed598`) has since run.

| finger | `base_link` IN | force | `base_link` OUT | force | Δcurl | Δ% | Δ / noise |
| --- | --: | --: | --: | --: | --: | --: | --: |
| L | +0.3280° | 8.53 N | +0.3296° | 14.61 N | +0.0016 | 0.49% | 1.33× |
| R | +0.3887° | 14.14 N | +0.3913° | 7.36 N | +0.0026 | 0.67% | 2.17× |

Earlier claims of "digit-identical" and "identical to four decimals" were **false and are retracted**.
The deltas are 1.3× and 2.2× the 0.0012° noise floor — small, but not zero.

Read at the strength the data supports: contact force changed 1.71× on L and 0.52× on R (the load
split between fingers essentially swapped) while curl moved under 0.7%, so `curl_deg` is close to
force-insensitive here and a sub-1% residual across runs loaded that differently reads as run
variation rather than a `base_link` effect. The sign agrees too — omitting `base_link` slightly
*increased* the curl, the opposite of the mount's inertia having been load-bearing. That is
**consistent with** the `PhysicsFixedJoint` grounded-chain argument but is a consistency check, not a
clean confirmation.

**Practical upshot: dropping `base_link` costs at most ~0.7% of the curl, not zero.** Anyone deriving
an `nf` against the patched stack should know the fix is a hair weaker without it.

**The drop still stands, on grounds that do not depend on that run:**

1. Structural — `panda_hand_joint` is a `PhysicsFixedJoint` (`panda_link8` → `base_link`), so the
   mount is welded into the grounded chain and its inertia cannot enter the `*_inner_finger_joint`
   mimic constraint. Analytical, not measured.
2. Scope — it is the only non-gripper-specific name in the set, and the whole point of the narrowing
   is that no generic name remains.
3. **Constraint compliance** — `base_link` is a 0.2888 kg body on the wrist whose corrected CoM moves
   ~45 mm. Correcting it changes what the *arm* carries, against the standing requirement that arm
   physics stay byte-identical. On its own that is sufficient reason to exclude it.

If the pending run comes back *different* from the structural-gate run, that would mean `base_link`
does couple into the pad constraint despite the fixed joint — surprising, and worth chasing rather
than burying.

**Residual caveat, stated because it is not verified:** the gate's *acceptance* side is verified
against the real asset (8 accepted, `base_link` and all nine `panda_link*` rejected). Its *rejection*
of BEHAVIOR objects rests on the structural argument alone — the dataset USDs are `*.encrypted.usd`
and stock `pxr` cannot open them without the key, so **no BEHAVIOR object was ever actually run
through the gate.**

### Detector — how to tell if an object was affected

**A mirrored left/right pair receiving an identical centre of mass, including the sign of y.** Two
mirror-image bodies cannot share a CoM y-component. Read `link.center_of_mass` for both members of any
left/right pair: if they match exactly rather than differing in y's sign, that object's CoM was
composed in the geom's parent frame and its inertia tensor is wrong.

Only objects whose collision geometry sits under an **intermediate Xform** can be affected — where
geoms are direct children of the link, the old and new expressions are algebraically identical.

### Which run modes actually carry the fix

At the time of this investigation, every measured result used the now-retired `MODE=stockfix`
comparison workflow. The validated release SIF subsequently incorporated the OG-lite implementation
wholesale; the historical patch files and mode were removed during release cleanup.

### ⚠ The loader patch fixes ONE of three sites with the same defect

`grep 'frame="parent"'` in OmniGibson finds **three** places that compose to the geom's immediate
parent rather than to the link:

| site | feeds | patched by `83b21d5`? |
| --- | --- | --- |
| `prims/rigid_prim.py:324` | centre of mass → inertia | **yes** |
| `prims/geom_prim.py:250` `points_in_parent_frame` | the collision **hull** | **no** |
| `utils/object_utils.py:88` `compute_base_aligned_bboxes` | **base-aligned bounding boxes** | **no** |

The hull site is what put `collision_boundary_points_world` **116 mm** from the pad link origins and
made hull-derived tip separation read **backwards** — the measurement failure that cost hours of
direction confusion. Measured: on the unpatched asset the hull and hull-free observables have
*opposite signs* on the right finger; after the asset-side fix they agree to 0.003 mm and the offset
collapses from 129.0 mm to 4.8 mm.

**The "no" in that table is measured on the patched engine, not only inferred from `grep`.** Ratio of
the hull observable to the hull-free one, over every press of four runs (a sign flip means the hull
reads backwards):

| run | `rigid_prim` CoM site | hull/hull-free ratio | hull usable? |
| --- | --- | --- | --- |
| `mass_authored` (`--mass` only, stock loader) | broken | −0.57 … −0.87 | no, backwards |
| `inertia_comfix` (**OG-lite `83b21d5`**, `MODE=stockfix`) | **fixed** | **−0.49 … −1.06** | **no, still backwards** |
| `mass_authored_anchor` (`--mass --anchor`, stock loader) | fixed | **+0.969 … +0.995** | **yes** |

So patching `rigid_prim.py` demonstrably does not repair the hull: the loader-patched run is as
backwards as the unpatched one. Only removing the dropped transform from the asset fixes both.

**Live REALM code is affected — but via the HULL site, not the bbox site.** Corrected: an earlier
version of this file attributed it to `object_utils.py:88`. Wrong. The two `get_base_aligned_bbox()`
calls in perturbation object replacement — `replace_obj()` in
`realm/environments/perturbations/object_sampling.py`, and `sb_vrb()` in
`realm/environments/perturbations/sb_vrb.py` — route through
`USDObject.get_base_aligned_bbox` → **`geom_prim.py:250`**, i.e. **site 2**. `object_utils.py:88` is a
third independent copy whose only caller is offline metadata tooling.

That makes the exposure *worse*, not better, and it joins up with the table above: the live REALM path
runs through the one site that is both **measured broken** and **not repaired by the loader patch**.
Quantified on the Robotiq links — every collision **and visual** hull is **61.09–192.66 mm** off
centre, with extents wrong by up to **31.80 mm** — against **0.00 mm on all eight `panda_link*`** under
the same loader and the same robot. The defect is asset-structure dependent, which is exactly why it
survived unnoticed.

**Consequence for the route choice, and it is now decided on evidence rather than taste:** the loader
patch fixes the curl but leaves the hull — and therefore `get_base_aligned_bbox()` in live perturbation
code — still wrong. Only removing the dropped transform from the asset (`xform-flatten`, or
`--mass --anchor`) repairs both. Not measured on any BEHAVIOR object; flagged, not quantified.

**Consequence for choosing a route:** the asset-side fixes (`xform-flatten`, or `--mass --anchor`)
repair all three sites at once, because they remove the dropped transform rather than compensating for
it in one consumer. The loader patch repairs one. That is an argument for the asset route on
correctness grounds, independent of the maintenance argument.

---

## 2. REALM — shipped behaviour

### Opt-in, zero effect unless selected by name

These add new robot configs and variant USDs. Nothing selects them unless a run names them, so they
cannot perturb existing results. Deleting the files is a complete revert.

- ⚠ **DEPRECATED, pending `fix-validate`** — `DROID_robolab_curlgrip{,_ee_control}.yaml` +
  `droid_robolab_curlgrip.usd` + its definition — **`nf=200`**. Chosen against the 18-26x compliance
  gap, which is now understood to be a *symptom* of the CoM bug, so the value **compensates for a bug
  rather than describing physics**. That bug is fixed by either route independently — engine side
  OG-lite **`15b4072`** (the `83b21d5..15b4072` `rigid_prim` CoM line), asset side the
  **`droid_robolab_xflat`** asset on a stock loader — and on a fixed build the curl is restored at the
  **authored `nf=1000`**, so `nf=200` **double-compensates**. Retained *only* until branch
  `fix-validate` reports whether the fixed build at authored `nf=1000` is sufficient on its own; that
  verdict decides deletion. A deprecation banner now heads both configs and the definition. The
  measured `nf` ladder in the definition was taken on the **broken** build — do not re-derive an `nf`
  from it.

| file | ships | status | revert |
| --- | --- | --- | --- |
| `realm/config/robots/DROID_robolab_curlgrip.yaml` | `nf=200` (via `model:`) | deprecated, banner added | delete the file |
| `realm/config/robots/DROID_robolab_curlgrip_ee_control.yaml` | `nf=200` (via `model:`) | deprecated, banner added | delete the file |
| `realm/robots/definitions/droid_robolab_curlgrip/droid_robolab_curlgrip.yaml` | points at the `nf=200` USD | deprecated, banner added | delete the directory |
| `realm/robots/panda_robotiq/droid_robolab_curlgrip.usd` | `naturalFrequency 200` on the 4 inner mimic joints | deprecated, asset left untouched | delete, or rebuild via `scripts/make_curlgrip_gripper_usd.py` |

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
| historical vector perturbation probe | `DRAWER_Z_MIN = 0.2` for drawer tasks; step budget allows one shared joint-reset loop | harness only; task 0 verified bit-identical |
| `realm/config/scenes/scenes.yaml`, three task YAMLs | scene/task config | check against 1.1.1 before trusting a comparison |
| `DROID_robolab_v2.yaml` gripper block | **unchanged** — no gains were ever added here | the `isaac_kp`/`isaac_kd` work was all probe-side |

**Arm physics was held byte-identical throughout and verified, not assumed** — `arm_0` controller
block, top-level `friction`/`armature`, and the seven `panda_joint*` DOFs, across 133 authored
attributes and 25 link prims (`CURLGRIP_ARM_IDENTICAL`, `SHIP_ARM_IDENTITY_OK`). The one-off probe
was later removed during release cleanup.

### Probe-only — no runtime effect

The temporary debug probes and Clara investigation harnesses had no runtime effect and were later
removed during release cleanup. The retained `scripts/clara/lib/paths.sh` changes only how active
cluster scripts resolve their own root.

The `mass-authored` variant generator was in this class too: it wrote a `.usda` into
`tmp/variants/` on demand and nothing loads it unless a run passes `--variant-usd`. It never writes
to `droid_robolab_v2.usd` and touches nothing under `data/`. Verified per-variant by
Its validation found 543 non-gripper prims / 2788 authored attributes identical to the shipped
asset (523 of them arm prims), and all 22 collision + visual geoms unmoved to 0.0 nm.

---

## 2b. Engine facts established by measurement, not changed

Not changes — **properties of stock OmniGibson / the Omniverse physics parser** that were measured
during this work and that any audit of OmniGibson deviations should carry. Both are the *same*
defect as `83b21d5`, expressed in two more places, and both are still present in stock.

### OmniGibson does not derive-then-respect mass properties; it computes-then-overwrites

`RigidPrim.update_meshes()` ends with `self.center_of_mass = com`. That setter is
`RigidPrimView.set_coms()` (`omnigibson/utils/deprecated_utils.py`), whose **stopped-simulation**
branch — which is where `_post_load` runs — does
`prim.GetAttribute("physics:centerOfMass").Set(...)`: a direct write into the *scene stage's* edit
target, on the prim the robot USD was referenced into. No composition arc can outrank that.

Consequences, all measured on `MODE=stock`:

- **`physics:centerOfMass` authored in an asset is discarded on every load.** An asset cannot defend
  that field. Anyone shipping a USD with a hand-authored CoM should know it will not be used.
- **`physics:mass`, `physics:diagonalInertia` and `physics:principalAxes` ARE honoured, verbatim.**
  Authored tensors reproduced RoboLab's runtime values to 0.00062% on all nine gripper links, masses
  bit-identical — and PhysX does **not** re-apply a parallel-axis shift to an authored tensor even
  when it then accepts a CoM 128 mm away.
- **Diagnostic:** a wrong CoM shared by a mirrored pair gives the two sides *different* effective
  inertias where symmetry demands they match (measured `nf_eq` 253 vs 217 on the two pads). An L/R
  asymmetry in a symmetric mechanism's effective inertia is the signature of a shared, wrong CoM.

### `UsdPhysics.MassAPI` on a Mesh under a collider Xform is silently ignored — by BOTH stacks

Every `Defeatured_*` **Mesh** prim in the robolab gripper authors `physics:mass`,
`physics:centerOfMass` and `physics:diagonalInertia` — the real CAD numbers, identical in
`droid_robolab_v2.usd` and RoboLab's `robolab_franka_robotiq_2f_85_flattened.usd`. **Nothing reads
them.** Aggregated into the link frame they give the fingertip pad **0.0392547 kg**; the body PhysX
actually builds is **0.00951321 kg** — in REALM *and* in RoboLab-through-Isaac-Lab. base_link is
2.11x apart, the inner knuckle 1.08x.

The cause is the same one behind `83b21d5`: `UsdPhysics.CollisionAPI` sits on the
`Defeatured_*_01` **Xform**, so the Mesh beneath it is not the collider prim and its `MassAPI` is not
the collider's `MassAPI`. The Omniverse **physics parser** and OmniGibson's **loader** make the same
mistake about the same prim, in two different places.

This is not gripper-specific and is the part worth carrying into a general audit: **any asset whose
`CollisionAPI` is on an Xform above its Gprims silently loses that Gprim's authored mass
properties**, and gets density-derived values instead. It is also a trap for anyone fixing such an
asset — those CAD numbers look exactly like the data you want, and mass-normalised they are 1.45x
the tensor PhysX actually realises for the pad. Do not trust them.

---

## 3. Results that should NOT be compared across this work

- **Every task 8 / task 9 score on record — INVALID, not failed.** The `impact_drawer` cabinet was
  authored with `purpose = "guide"` on all 56 of its geoms, so it never reached the colour pass and
  contributed **0 px to every camera, wrist included**, on every run made before `8598e59`. The
  policy was asked to open an object absent from all of its inputs, so task 8's `SR 0.000` was never
  a policy result and no task-8/9 number measures anything. Do not quote them as outcomes, and do
  not read them as a floor either — an invalid cell bounds nothing.
- **Task 8 `open_drawer` before the upAxis fix — a *second*, independent defect.** The cabinet was
  placed lying on its back, so `init_openness_fraction` started at 0.62 instead of 0. Two rubric
  stages are *absolute* (`> 0.125`, `> 0.65`), so they scored partly for free, which inflates the
  pre-fix numbers. Recorded for completeness only: it is not why those numbers are unusable, and
  fixing the upAxis did not make the post-fix ones usable — the render defect spans both sides.
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
