# OmniGibson 3.9.1 vs Isaac Sim / Isaac Lab 2.2.0 — transforms, articulation state, asset import

What OmniGibson does to poses, xformOps, articulation state and imported assets that raw Isaac Sim /
Isaac Lab does not, or that overrides what the asset authored. Every row cites `file:line`.
OmniGibson paths are relative to `/mnt/home_lustre/sedlam56/projects/OG-lite_og391`; REALM paths to
`/mnt/home_lustre/sedlam56/projects/REALM_og391`.

**Status.** The OmniGibson side is a direct read of the source plus static USD measurement of REALM's
two custom assets (`usd-core` 0.26.8 on the host CPU — no Kit, no GPU). Numbers labelled *measured*
were read out of a completed run in this session and the command is given. Numbers attributed to
another audit are cited as such. Anything I reasoned about but did not measure is labelled
**inferred** in place. The Isaac side is an in-SIF extraction against
`/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif`; the two citations that reframe a
conclusion (`isaacsim .../impl/xform_prim.py:1212-1222` and `:1237-1242`) I re-read myself. Cells
still unresolved say `NOT ESTABLISHED` rather than carrying a guess.

---

## The headline: frames composed to the wrong level

This is not one bug. It is one **expression**, repeated at four sites, three of which are still live.

OmniGibson repeatedly wants a geom's pose *in its link's frame* and reaches for
`XFormPrim.get_position_orientation(frame="parent")`. That accessor is documented at
`prims/xform_prim.py:267` as "get position relative to the object parent" — **one level up**. It
equals the link frame only when the geom is a **direct child** of the link. It very often is not:
`GEOM_TYPES` is `{Sphere, Cube, Cone, Cylinder, Mesh}` (`utils/constants.py`), so whenever an
intermediate `Xform` carries the collision APIs (or a DCC exporter's grouping node sits in the way),
the prim OmniGibson wraps is the `Mesh` and the `Xform → link` transform is **silently dropped**.

**Isaac has no counterpart layer at all.** It never composes a collision geom's frame itself:
`RigidPrim.get_coms` / `set_coms` are pure passthroughs to the PhysX tensor view
(`isaacsim/core/prims/impl/rigid_prim.py:1170-1172`, `:1421`), and PhysX returns the CoM already
"*given in the actor frame*" (`omni/physics/tensors/impl/api.py:3639-3644`). Isaac's only
geometry-frame composition is `UsdGeom.BBoxCache.ComputeWorldBound` — straight to **world**, never to
a parent (`isaacsim/core/utils/bounds.py:136`, `:139`, `:170`, `:227`), and Isaac Lab uses no bbox
utilities at all. So OmniGibson's whole collision-geom CoM / hull layer is an **OmniGibson-only
addition**, and every bug in it is OmniGibson's alone.

The up-axis defect is the same failure wearing a different hat: Kit's metrics assembler *appends*
`xformOp:rotateX:unitsResolve` to the referencing prim's `xformOpOrder`, and
`XFormPrim._set_xform_properties` strips only the **unsuffixed** `rotate*` / `transform` ops
(`prims/xform_prim.py:114-125`), so the appended op is invisible to every pose setter while
post-multiplying every pose they write. In both cases a transform that sits at a level nobody looked
at survives into the result.

**That last one is inherited, not invented.** OmniGibson's strip list is byte-identical to Isaac's
own — `isaacsim/core/prims/impl/xform_prim.py:1212-1222`, same ten entries in the same order, same
`ClearXformOpOrder()`-then-remove structure, same read-pose / write-pose-back bracketing
(`:1211`, `:1256`). The rotate-type `unitsResolve` blind spot is therefore an **upstream Isaac gap
that OmniGibson inherited**. But Isaac has one piece of handling OmniGibson does **not** — see row 5.

### The four instances

| site | what it feeds | patched? |
|---|---|---|
| `prims/rigid_prim.py:324` (`local_pos, local_orn = mesh.get_position_orientation(frame="parent")`, consumed at `:332`) | collision-geom **centre of mass** → link inertia tensor → mimic-joint stiffness | **yes**, OG-lite `83b21d5` → `6d04cc9` → `0fed598` → `15b4072`; scoped by structure to a Robotiq 2F-85 (`:60-67`), so **every other asset is still wrong** |
| `prims/geom_prim.py:250` (`points_in_parent_frame`) → `prims/rigid_prim.py:559`, `:604-612`, `:585-593` | collision + visual **convex hulls** → `RigidPrim.aabb`, `EntityPrim.aabb`, `get_base_aligned_bbox`, `Robot._infer_finger_properties` | **no** |
| `utils/object_utils.py:88` (`compute_base_aligned_bboxes`) | per-link base-aligned **bounding boxes** written into dataset metadata | **no** |
| Kit metrics assembler vs `prims/xform_prim.py:114-125` (unsuffixed-only strip) | **every pose** written on the affected prim | **yes** for the mechanism, OG-lite `ec7373b` (`objects/usd_object.py:355-364`), but see the two residuals below |

The CoM instance is the **`audit-rigid` sibling's**; it inflated pad inertia **77.3×** and is the root
cause of the rigid-gripper investigation. Cited here as one instance of the pattern — not
re-derived, not duplicated.

### Measured: the hull instance

`RigidPrim._compute_points_on_convex_hull` (`prims/rigid_prim.py:550-574`) concatenates
`mesh.points_in_parent_frame` over a link's geoms and hulls them; `collision_boundary_points_world`
(`:604-612`) then applies **the link's** world transform via
`XFormPrim.transform_local_points_to_world` (`prims/xform_prim.py:416-417`, which uses
`get_world_pose_with_scale(self.prim_path)` — the link's full world xform, correctly). So:

```
OG:      P = W_link · ( X_geom · p )
correct: P = W_link · ( X_intermediate · X_geom · p )
```

Measured statically on `realm/robots/panda_robotiq/droid_robolab_padspring.usd` (scratchpad
`audit-transform_probe_hull.py`, `..._probe_sep.py`). Every 2F-85 collision geom sits under exactly
one `Defeatured_*_01` intermediate Xform. The dropped transform is a **90° (left) / 180° (right)**
rotation about z — the mirror — plus, on the `*_inner_finger` links, a translation of
**133.7 mm**. Resulting per-geom centroid displacement in the link frame:

| link | geom | displacement |
|---|---|---|
| `base_link` | `basestep` | 38.3 mm |
| `left_outer_knuckle` | `Finger1step` | 96.0 mm |
| `left_outer_finger` | `finger2step` | **128.4 mm** |
| `left_inner_knuckle` | `finger3step` | 129.0 mm |
| `left_inner_finger` | `finger4step` / `fingertipsstep` | **117.5** / 148.6 mm |
| `right_outer_knuckle` | `Finger1step` | 134.4 mm |
| `right_outer_finger` | `finger2step` | 181.0 mm |
| `right_inner_knuckle` | `finger3step` | 166.6 mm |
| `right_inner_finger` | `finger4step` / `fingertipsstep` | **117.3** / 136.1 mm |

The 128.4 mm on `left_outer_finger` is the same 128.3 mm the CoM audit reported, from the same
dropped transform — confirming these are **one defect at two code sites**, not two defects. The
117.5 / 117.3 mm on the pad links is this project's **116 mm** pad-origin offset.

Two signatures make it unmistakable, and both are visible in the numbers:

- **Left and right come out identical.** OG's link-frame centroid for `left_outer_knuckle` and for
  `right_outer_knuckle` is the same vector `(-0.04054, 0.05448, 0.0)`, including the sign of y —
  impossible for a mirrored pair. Correct values are `(0.05448, +0.04055, 0)` and
  `(0.05448, −0.04055, 0)`.
- **Extents come out axis-swapped.** Pad link hull extent, OG `(27.0, 57.18, 31.15)` mm vs correct
  `(27.0, 31.15, 57.18)` mm.

**Separations do not survive it.** Pad-to-pad, in the robot-root frame at the USD-authored rest
pose, hulling exactly as `_compute_points_on_convex_hull` does:

| quantity | OmniGibson | correct | error |
|---|---|---|---|
| min hull-vertex separation | **60.43 mm** | **83.04 mm** | −22.6 mm (−27%) |
| centroid separation | 113.45 mm | 108.49 mm | +5.0 mm |
| pad centroid y (L, R) | +0.173, +0.060 | +0.054, −0.054 | both pads on the same side |

This **refines** the earlier working belief that "hull-derived separations survive it, absolute hull
positions do not". Separations survive *within one geom* (the dropped transform is rigid, so a single
geom's own diameter is preserved up to axis relabelling) and *within one link* when all its geoms
share the same intermediate — but **not between the left and right pads**, because the dropped
rotations there differ (90° vs 180°). Recorded as a correction, per the runbook rule.

**A second, independent asset shows it.** `custom_assets/impact_drawer/usd/cabinet.usd`: each of the
five drawer links' 11 collision geoms sits under a three-deep chain
`ObjectCapture / Geometry / Mesh` carrying `rotateXYZ = (90, 0, 0)`. Per-geom centroid displacement
92–400 mm; the drawer link's hull extent comes out `(0.391, 0.098, 0.554)` m where the truth is
`(0.391, 0.554, 0.098)` m — y and z swapped. The cabinet's `base_link` has an intermediate
(`Geometry_01`) too, but it is identity, so that link measures 0.00 mm. The defect is
**asset-structure-dependent, not asset-dependent**.

**Downstream, in this stack.** `Robot._infer_finger_properties` (`robots/robot.py:1839`, `:1856`)
builds `_eef_to_fingertip_lengths`, `_default_ag_start_points` / `_default_ag_end_points`, and the
"which finger is the lower-y one" test (`:1867-1868`) directly out of
`collision_boundary_points_world`. On the unpatched asset that test is fed two point clouds whose
centroids are on the *same* side of y. `USDObject.get_base_aligned_bbox` (`objects/usd_object.py:999`)
reads the same hulls at `:1058-1060`.

### The bbox instance

`utils/object_utils.py:88` repeats the expression verbatim inside `compute_base_aligned_bboxes`:

```python
local_pos, local_orn = mesh.get_position_orientation(frame="parent")
pts_in_link_frame.append(get_particle_positions_from_frame(local_pos, local_orn, mesh.scale, pts))
```

Two things must be kept apart here, because conflating them would misattribute the risk:

- **`compute_base_aligned_bboxes` itself is offline and, in this tree, unreached.** Its only caller
  is `compute_obj_kinematic_metadata` (`utils/object_utils.py:130`), which has **zero callers** in
  the whole OmniGibson repo (`grep -rn compute_obj_kinematic_metadata .` returns only its own
  `def`). It is a dataset-authoring utility. Whatever it wrote into shipped metadata is baked in and
  I have not checked any shipped metadata against it.
- **`get_base_aligned_bbox` is a different function** (`objects/usd_object.py:999`) and it *does*
  reach live REALM code — `replace_obj()` in `realm/environments/perturbations/object_sampling.py`
  and `sb_vrb()` in `realm/environments/perturbations/sb_vrb.py`, both in perturbation object
  replacement. It is
  defective through the **hull** site (`objects/usd_object.py:1058-1060`), not through
  `object_utils.py:88`.

So: any object whose collision geometry sits under a non-identity intermediate Xform can get a wrong
bounding box during a perturbation. **Flagged, never quantified** — I did not measure a perturbation
bbox, and the BEHAVIOR dataset is not on disk on this host to check a replacement candidate.

### Why nobody noticed: BEHAVIOR assets have an identity intermediate

**Inferred from the converter source, not measured** (no BEHAVIOR asset available here).
`convert_urdf_to_usd` promotes each link's meshes into a `visuals` / `collisions` Xform created by
`lazy.pxr.UsdGeom.Xform.Define(side_stage, referrer_prim_path)` at
`utils/asset_conversion_utils.py:1312`, with the meshes referenced in underneath at `:1315-1334`.
`Xform.Define` authors **no** xformOps, so that intermediate is exactly identity — and where the
intermediate is identity, `X_geom` and `X_intermediate · X_geom` are algebraically the same
expression. Every BEHAVIOR-dataset object therefore hits the bug's null case. Assets that come from
a DCC export or a different converter — both of REALM's — do not.

### Verified clean — checked, and correct

These are pose paths in my files that I checked *specifically for this defect* and found right. A
verified-clean list is worth as much as the defect list; do not re-check these.

| site | why it is correct |
|---|---|
| `prims/geom_prim.py:238-243` `check_points_in_volume` | uses `self.scaled_transform` = `get_world_pose_with_scale(geom_path)` — the geom's own **full world** xform out of the Fabric hierarchy, so every ancestor is included by construction |
| `prims/geom_prim.py:257-268` `GeomPrim.aabb` | same: `scaled_transform` on the geom itself |
| `prims/xform_prim.py:409-417` `scaled_transform` / `transform_local_points_to_world` | the transform it fetches is correct and complete **for the prim it is called on**; the hull bug is that it is handed points expressed in a *different* frame, not that this is wrong |
| `prims/entity_prim.py:1086-1088` vs `:1109` | articulation pose write (`_articulation_view.set_world_poses`) and read (`root_link.get_position_orientation`) now address the **same** frame, the root link — OG-lite `7c59ed5` |
| `prims/entity_prim.py:1039-1051` | the stopped-branch root-link compensation composes `target @ pose_inv(root_local)` with `root_local` a **local** transform, so it is frame-agnostic and correct for both `"world"` and `"scene"` |
| `prims/xform_prim.py:113` / `:159` / `:161` | `_set_xform_properties` reads *and* writes through `XFormPrim.*` pinned to the same prim, so the op-order rewrite is pose-preserving; the round-trip assert at `:165-168` (1e-4 m / 1e-3 on R) is a real guard, not decoration — OG-lite `7c59ed5` |
| `prims/xform_prim.py:223-226` (the column-normalise "unscale") | exact for a **uniformly** scaled ancestor chain — the local 3×3 is `(1/s)·Q`, and column norms are `s`. For a non-uniform ancestor scale it is wrong, but the assert at `:229-232` fires; see trap 3 |
| `utils/transform_utils.py:497-570` `decompose_mat` / `:573` `mat2pose` | transposes into Gf's row-vector convention correctly, extracts translation from the right slot, and Gram-Schmidts scale and shear out. Checked by inspection of the convention handling — I did not run a numeric round-trip |
| `utils/transform_utils.py:1435-1466` `transform_points` | column-vector convention throughout, matching `get_world_pose_with_scale`'s `.T` at `utils/usd_utils.py:1265` |

**Not checked** (say so rather than implying clean): the cloth path (`ClothPrim`, particle
positions), `utils/usd_utils.py:1866-1932` (`_get_all_relative_poses`, relative Jacobians),
`omnigibson/sensors/` camera and sensor pose composition, and any pose path under
`omnigibson/systems/`.

---

## Table

| # | site | what OmniGibson does | what raw Isaac / Isaac Lab does, or what the asset authored | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 1 | `prims/geom_prim.py:250` → `prims/rigid_prim.py:559`, `:604-612` | composes a collision/visual geom's hull points to the geom's **immediate parent**, then applies the **link's** world transform | the asset authors an intermediate Xform between link and geom carrying (on REALM's 2F-85) a 90°/180° rotation + 133.7 mm | **yes** | **bug** (frames composed to the wrong level) | **measured**: geom centroids 38–181 mm off; pad-pad hull separation reads **60.43 mm** vs a true **83.04 mm**; left/right hulls come out identical instead of mirrored; link hull extents axis-swapped | no — stock |
| 2 | `utils/object_utils.py:88` | same expression inside `compute_base_aligned_bboxes` | same | **yes** | **bug**, same class | **not quantified.** Function is unreached in this tree (only caller `:130` has zero callers). The *live* bbox risk is row 1 reaching `objects/usd_object.py:1058-1060` ← `get_base_aligned_bbox` ← `replace_obj()` in `realm/environments/perturbations/object_sampling.py` and `sb_vrb()` in `realm/environments/perturbations/sb_vrb.py` | no — stock |
| 3 | `prims/xform_prim.py:114-125` | `_set_xform_properties` strips only **unsuffixed** `xformOp:rotate*` / `xformOp:transform`; `ClearXformOpOrder()` at `:130` then drops *any* remaining op from the **order** | **the same ten-entry list, verbatim**, at `isaacsim/core/prims/impl/xform_prim.py:1212-1222`. Kit's metrics assembler appends `xformOp:rotateX:unitsResolve` (**−90.0**, asserted by the assembler's own test `omni/metrics/assembler/core/tests/referenceTests.py:200-205`) and neither stack strips it | **yes** | **upstream Isaac gap, inherited** — not an OmniGibson invention | this project's up-axis failure; worked around upstream of the symptom by OG-lite `ec7373b` | mechanism inherited from Isaac, fix OG-lite |
| 4 | `objects/usd_object.py:355-364` | rewrites the **layer's `upAxis`** to match the stage before referencing | the asset authors `upAxis = Y` (**measured**: `cabinet.usd` `upAxis=Y`, stage Z) | yes | fix for row 3 | removes the assembler's reason to append | **OG-lite only** (`ec7373b`) |
| 5 | `prims/xform_prim.py:135-141` (and `objects/usd_object.py:358-364`, which normalises `upAxis` **only**) | OmniGibson's copy of `_set_xform_properties` **omits Isaac's `xformOp:scale:unitsResolve` fold**. `ClearXformOpOrder()` at `:130` then drops that op from the order, and the round-trip assert at `:165-168` checks **position and rotation only** — never scale — so it passes | Isaac multiplies the resolve factor **into** `xformOp:scale` and removes the property, `isaacsim/core/prims/impl/xform_prim.py:1237-1242`; its CHANGELOG records this as a deliberate fix (`isaacsim.core.prims/docs/CHANGELOG.md:96`, *"take in consideration scale:unitsResolve attribute if authored"*) | **yes** — the assert cannot see it | **bug: OmniGibson drops a fix Isaac already shipped** | **inferred, not measured** — I have no MPU≠1 asset here (both REALM assets measure `metersPerUnit = 1.0`, so the path is untaken). Mechanism is nailed though: Isaac's own test asserts the op carries `[100.0, 100.0, 100.0]` for a cm-authored layer (`isaacsim/core/utils/tests/test_stage_utils.py:144`), so a cm asset would load at **1/100 scale**, silently | no — stock, and **worse than Isaac** |
| 6 | *(residual of row 4)* | the fix stops *new* ops being appended; it does not remove ops already **baked into the asset file** | **measured**: `cabinet.usd` ships 5 prims — the drawer links `drawer_blender_cut_00..04`, each carrying `RigidBodyAPI` — with `xformOp:rotateX:unitsResolve = -90.0` already in `xformOpOrder` | **yes** | **residual gap** | neutralised in practice for this asset: `_set_xform_properties` runs (`prims/xform_prim.py:73`, `xform_props_pre_loaded` defaults False at `prims/prim_base.py:65`; the instanceable guard on the same line cannot trip — **measured**, 0 instanceable prims in either REALM asset, and `_ALLOW_INSTANCING = False` at `utils/asset_conversion_utils.py:84`) and `ClearXformOpOrder()` drops the op from the order pose-preservingly. **But see row 7** | OG-lite |
| 7 | `robots/robot.py:305` | sets `xform_props_pre_loaded = True`, so `_set_xform_properties` is **skipped entirely** for every robot | n/a | **yes** | **exemption that reopens row 3** | a robot USD keeps whatever ops it ships. **Measured** on `droid_robolab_padspring.usd`: 16 prims carry `xformOp:rotateZYX`, all `<link>/geometry` and `<link>/visual__geometry` — all zero (max 3.0e-7°), so **no effect on this asset**. Any robot asset with a non-zero extra op would be silently post-multiplied | no — stock |
| 8 | `objects/usd_object.py:261-262` | `for p in stage.Traverse(): p.RemoveAPI(ArticulationRootAPI); p.RemoveAPI(PhysxArticulationAPI)` — strips **every** authored articulation root, then re-derives one from a body0/body1 heuristic (`:286-307`) and re-applies at `:328-333` | whatever the asset author declared as the articulation root, plus every `physxArticulation:*` attribute authored alongside it (solver iteration counts, `articulationEnabled`, sleep/stabilization thresholds). Only `enabledSelfCollisions` is re-set (`:331-333`) | **yes** | **override of authored values** | not quantified; explains why an asset's own articulation configuration never takes effect | no — stock |
| 9 | `objects/usd_object.py:365-370` | re-exports the whole asset to a fresh temp layer on **every** `load()` | Isaac references the asset file directly (`isaacsim.core.utils.stage.add_reference_to_stage`, wrapped at `utils/usd_utils.py:2570`) | yes | deviation from Isaac defaults | per-object layer copies; also the reason the assembler's content-hash-keyed `UnitsAdjust` layer materialised for only the first reference | no — stock |
| 10 | `utils/asset_conversion_utils.py:338-405` `_add_xform_properties` | the **non-pose-preserving twin** of `_set_xform_properties`: `ClearXformOpOrder()` at `:372`, removes `xformOp:transform` and every unsuffixed `rotate*` at `:358-375`, re-orders at `:404` — with **no pose read/write around it** | an authored `xformOp:transform` (**measured**: `cabinet.usd` `/cabinet` and `/cabinet/base_link` use `xformOp:transform` as their *only* op) would be deleted outright | **yes** | **would destroy authored transforms** | **latent only.** Callers: `:505` (freshly created meta-link prims — safe) and `examples/robots/import_custom_robot.py:340`, `:460`. **Not on the object load path.** Dangerous if ever pointed at a real asset | no — stock |
| 11 | `utils/asset_conversion_utils.py:1242-1250` | during mesh promotion, copies every authored attribute of the wrapper prim **onto** the child mesh (overwriting the child's own `xformOp:*`), then `Sdf.CopySpec`s the child up to the wrapper's path (`:1259`) | the docstring at `:1208-1210` says the mesh and xforms are "combined"; they are **overwritten**, and the intermediate Xform found at `:1014-1026` (`resolve_imported_geometry`) is discarded entirely, never composed | **yes** | **bug**, same class as rows 1-3, in the import path | **inferred, not measured** — I have no asset that exercises this path on this host. Harmless when the mesh and the intermediate are identity, which is the common Isaac-URDF-importer case | no — stock |
| 12 | `utils/asset_conversion_utils.py:1328` | forces `approximation = "convexHull"` on **every** collision mesh at import | whatever the URDF/collision authoring intended | yes | **constraint imposed on assets** | not quantified; later overridable from metadata | no — stock |
| 13 | `utils/asset_conversion_utils.py:907-920` | URDF import config: `default_drive_type = JOINT_DRIVE_NONE`, `default_drive_strength = 0.0`, `position_drive_damping = 0.0`, `density = 0.0`, `fix_base = False`, `self_collision = False` | Isaac's URDF importer defaults (position drive, non-zero stiffness, 1000 kg/m³): `NOT ESTABLISHED` in-SIF | yes | **override of authored values** | every URDF drive gain and every implicit density is discarded at import; mass comes only from `<inertial>` | no — stock |
| 14 | `objects/dataset_object.py:319-329` | rewrites **every** prismatic and revolute joint to `drive:*:physics:type = "acceleration"`, `damping = DEFAULT_*_JOINT_DAMPING`, `stiffness = 0.0`, targets `0.0` | the USD's authored drive block | **yes** | **override of authored values** | see `control_and_actuation.md` for the actuation-side consequences | no — stock |
| 15 | `objects/dataset_object.py:285-300` | overrides link **mass**/**density** with a per-category average (`get_avg_category_specs()`), split by link volume | the asset's authored `UsdPhysics.MassAPI` | partly (warns only on an unknown category, `:281-283`) | **override of authored values** | not quantified here | no — stock |
| 16 | `prims/entity_prim.py:319-377` `_update_joint_limits` | multiplies every **prismatic** joint's limits by the object scale and writes them back through `JointPrim.lower_limit` / `upper_limit` | the authored `physics:lowerLimit` / `physics:upperLimit`. When stopped, `deprecated_utils.py:132-133` writes them **into the USD prim** | **yes** | **override of authored values** | not quantified. Two latent traps: (a) the setter at `prims/joint_prim.py:491-493` reads the *other* limit through a getter that substitutes `DEFAULT_MAX_POS = 1000.0` for an unlimited joint (`:476`, `:510`), so an unlimited joint can be written as ±1000·scale and thereafter report `joint_has_limits == True` (`:538`); (b) `EntityPrim.scale`'s setter (`:1243-1251`) does **not** re-run it, so a post-load scale change leaves limits stale | no — stock |
| 17 | `prims/entity_prim.py:167` | `assert th.all(self.original_scale == 1.0)` — the entity Xform must ship unit scale | USD permits any scale there | no — it aborts | **constraint imposed on assets** | an asset with a non-unit top-level scale cannot load | no — stock |
| 18 | `prims/entity_prim.py:98` | `self.scale = self.scale` in `_initialize` — re-writes every link's scale after physics init | comment at `:95-97`: Isaac's articulation warm-up can reset `xformOp:scale` on bodies with no collision geometry | yes | workaround for an Isaac behaviour | none intended | no — stock |
| 19 | `utils/usd_utils.py:1245`, `:1256` | **all** pose reads go through the **Fabric** hierarchy (`og.sim.fabric_hierarchy.get_world_xform`), not USD and not the physics view; every write ends with `og.sim.fabric_hierarchy.update_world_xforms()` (`prims/xform_prim.py:259`) | Isaac reads **USD by default** — recursive `GetLocalTransformation()` up the parent chain plus `Orthonormalize()`, `isaacsim/core/utils/xforms.py:182-187` (not `ComputeLocalToWorldTransform`) — with Fabric only on `usd=False` (`isaacsim .../xform_prim.py:662-667`), and `RigidPrim` prefers the **PhysX tensor view** when a handle exists (`rigid_prim.py:310-313`) | yes | architecture deviation | correctness depends on Fabric being synced. `prims/xform_prim.py:455` (`scale` setter) does **not** call `update_world_xforms`, so a scale write leaves the Fabric world xform stale until the next pose write — **inferred from the source, not measured** | no — stock |
| 20 | `prims/rigid_dynamic_prim.py:157` | `EntityPrim._load_state` (`prims/entity_prim.py:1590`) restores an articulation's pose by calling `root_link._load_state` → `XFormPrim._load_state` (`prims/xform_prim.py:512-514`) → `RigidDynamicPrim.set_position_orientation` → `_rigid_prim_view.set_world_poses` — the **RigidPrimView**, while the ordinary setter (`prims/entity_prim.py:1086`) uses the **ArticulationView** | Isaac Lab **never** writes a pose through a rigid-body view on an articulation link — its only such view is for counting shapes (`isaaclab/envs/mdp/events.py:205-207`), and `RigidObject` rejects assets with an enabled articulation root (`rigid_object.py:480-490`). isaacsim actively gates it: `is_prim_non_root_articulation_link` (`isaacsim/core/utils/prims.py:871-873`, *"can't have a transformation applied to it"*) suppresses pose writes at `xform_prim.py:152-166` and warns at `:291-292`. `omni.physics.tensors` documents the "does not modify … articulation links" caveat only for `set_kinematic_targets` (`api.py:3428`), giving **no guarantee** for `set_transforms` | **yes** | **suspected bug — API asymmetry between the pose path and the state-restore path** | **not verified at runtime.** The Isaac-side evidence above is strong but circumstantial; do not act on it until someone measures a state round-trip on an articulated object with a non-zero root pose | no — stock |
| 21 | `prims/xform_prim.py:508-514` | `XFormPrim._dump_state` / `_load_state` store and restore the **world**-frame pose | n/a | yes | frame choice | a state dumped in one scene and loaded into another lands at the first scene's world coordinates. Related and **confirmed**: OG-lite `ef7442b` — `_load_scene_prim_with_objects` wrote object poses with the default `frame="world"` while the scene prim was parked at `INITIAL_SCENE_PRIM_Z_OFFSET = -100`, baking `+100 m` into every local z; measured 70 of 128 objects above z=50 in each of scenes 1..3 | bug stock, fix OG-lite (`ef7442b`) |
| 22 | `utils/transform_utils.py:418`, `:740`, `:1575` | `assert torch.allclose(det(rmat), torch.tensor(1.0))` — the reference `1.0` is built **on the CPU** | n/a | no — it raises | **device bug**, same family as OG-lite's `27806cb` | mixed-device `allclose` raises the moment the matrix is GPU-resident. `mat2quat` is on the `mat2pose` path (`mat2pose` `:573` → `decompose_mat` `:497` → `:567`) | stock; **not** covered by `27806cb` |
| 23 | `utils/transform_utils.py:1558` | `R = torch.eye(3) + sin(θ)·K + …` — the identity is on the CPU while `K` was device-coerced two lines above (`:1549`) | n/a | no — it raises | **incomplete OG-lite fix** | `27806cb` fixed the early-return `eye(3)` at `:1538` and `K` at `:1549` but missed `:1558`. Called from `robots/robot.py:3936` | **OG-lite** (partial fix) |
| 24 | `utils/usd_utils.py:2287-2292` | polygon → triangle conversion is a naive **fan** (`[i, i+j+1, i+j+2]`), and the mesh's `orientation` (`leftHanded`/`rightHanded`) attribute is never read | USD authors winding explicitly | **yes** | **override of authored values** | **inferred**: a left-handed mesh yields inverted winding, so `trimesh.is_volume` is False and `get_mesh_volume_and_com` (`:2423-2431`) silently falls back to the **convex hull's** CoM. Not observed on either REALM asset | no — stock |
| 25 | `utils/usd_utils.py:2307-2338` | `mesh_prim_shape_to_trimesh_mesh` builds Cone/Cylinder along **z** unconditionally and never reads `UsdGeom.Cone/Cylinder`'s `axis` attribute; `Capsule` raises | USD's `axis` defaults to `Z` but may be `X` or `Y` | **yes** | **override of authored values** | **inferred**, not observed. Feeds `GeomPrim.points` (`prims/geom_prim.py:148`) for non-Mesh geoms | no — stock |
| 26 | `prims/xform_prim.py:223-226`, `:229-232` | converts world→local by dividing the rotation block by its **column norms**, then **asserts** the result is orthogonal — so a non-uniformly scaled ancestor makes the pose setter **abort** | Isaac runs a full `Gf.Transform` factorization (rotation / scale / shear / pivot) so the residual parent scale lands in the discarded scale factor: `isaacsim/core/utils/numpy/transformations.py:41-53` (torch backend identical, `torch/transformations.py:47-60`). **No orthogonality assertion anywhere in isaacsim or isaaclab** | no — it aborts | **constraint imposed on assets** | not quantified. `EntityPrim.scale`'s setter (`:1243-1251`) can create a non-uniform link scale, so this is reachable. Note the **warp** backend (`warp/transformations.py:73-79`) uses `ExtractRotation()` on the raw scaled matrix and is *not* scale-safe — do not treat "Isaac handles it" as backend-independent | no — stock |
| 27 | *(not done — recorded as an option)* | nothing; OmniGibson never touches the assembler's carb settings | Kit exposes `/metricsAssembler/conformToXformCommonAPI`, which makes the assembler bake into `xformOp:scale` / `xformOp:rotateXYZ` **instead of** emitting suffixed `unitsResolve` ops (`omni/metrics/assembler/core/bindings/_metricsAssembler.pyi:142-145`; behaviour pinned by `.../core/tests/referenceTests.py:224-236`). Isaac Lab sets only `metricsAssembler.changeListenerEnabled = false` (`apps/isaaclab.python.kit:130-131`) and never sets this one | n/a | **available lever, unused** | **untested here.** Would address rows 3 and 5 at the source rather than per-asset, since the ops it produces *are* in the strip list. Worth a look before anyone hand-rolls more per-asset normalisation | n/a |

---

## Benign / by design — checked, do not re-check

- **`prims/xform_prim.py:223-226`, the column-normalise "unscale".** It looks like a hack and it is
  exact: with USD's post-multiplied scale the local 3×3 is `(1/s)·Q` and the column norms *are* `s`.
  For a non-uniform ancestor scale it is wrong, and the orthogonality assert at `:229-232` catches
  that. Correct as written — but see the new table row 26: catching it is itself a divergence,
  because Isaac handles that case instead of refusing it.
- **`prims/entity_prim.py:1039-1051`, the root-link compensation** (OG-lite `7c59ed5`). Composes
  `target @ pose_inv(root_local)` with a **local** transform, so it is frame-agnostic; skipped
  entirely when the root link sits at the entity origin (`:1043`), which every BEHAVIOR asset does.
  Not a behaviour change for the dataset.
- **`objects/usd_object.py:355-364`, the up-axis rewrite** (OG-lite `ec7373b`). Rotates nothing: up
  axis is layer metadata, every caller places the object explicitly afterwards, and for a Z-up asset
  the branch is untaken. Verified untaken on `droid_robolab_padspring.usd` (**measured**: `upAxis=Z`).
- **`utils/usd_utils.py:2522-2523`, `create_usd_stage` forcing `metersPerUnit=1.0` / `upAxis=Z`.**
  This is the OmniGibson stage convention and applies only to stages OmniGibson creates.
- **`utils/urdf_preprocessing.py:125` `strip_mimic_joints`.** **Zero callers — confirmed dead code.**
  Do not attribute any behaviour to it. (Settled elsewhere; recorded here so it is not re-opened.)
- **Asset conversion is not a source of divergence between REALM's two converters.**
  `droid_robolab_v2.usd` vs `robolab_franka_robotiq_2f_85_flattened.usd` were diffed exhaustively
  elsewhere: articulation root, kinematic tree, both `FixedJoint`s, all five mimic blocks, joint
  limits and velocity limits **identical**; colliders agree to 5.4e-11 m and world anchors to 7.6 nm.
  A useful negative — if two runs differ, the converter is not why.
- **`prims/entity_prim.py:235-241` and `:310-315`, `_compute_articulation_tree`'s relaxed asserts.**
  REALM-side relaxations of upstream single-root / joint-count assertions, already deliberate.

---

## Corrections made while writing this

Two claims in the first draft of this chapter were wrong and are recorded rather than quietly fixed.

1. **"Separations survive the dropped transform."** They do not, between left and right. Measured:
   60.43 mm against a true 83.04 mm. Superseded above.
2. **"`metersPerUnit` is unnormalised, so the units correction gets post-multiplied through the same
   blind spot as the rotation."** Wrong mechanism. The scale-type op is *dropped from the op order*
   by `ClearXformOpOrder`, not composed — and the round-trip assert never checks scale, so it passes.
   The result is a silently **lost** unit conversion, not a silently applied one. Isaac folds the op
   into `xformOp:scale` and OmniGibson does not; that is row 5 as it now stands.

---

## Traps that mislead rather than deviate

1. **`frame="parent"` is not wrong everywhere.** Where a geom is a direct child of its link, the OG
   expression and the correct one are *algebraically identical*. A site using `frame="parent"` is not
   automatically a defect — you have to look at the asset's prim tree. This is why the bug is
   invisible on the entire BEHAVIOR dataset (see "Why nobody noticed").
2. **`_set_xform_properties` and `_add_xform_properties` have the same strip list and opposite
   safety.** `prims/xform_prim.py:114-125` reads the pose, rewrites the order, writes the pose back
   and asserts the round trip. `utils/asset_conversion_utils.py:338-405` does the same rewrite with
   **none** of that. Do not reason about one from the other.
3. **The orthogonality assert's message is misleading.** `prims/xform_prim.py:232` says "local
   transform is not orthogonal". The real cause is almost always a **non-uniformly scaled ancestor**,
   which `EntityPrim.scale`'s setter (`:1243-1251`) can create on any link.
4. **`prims/rigid_kinematic_prim.py:124-165` returns `mass = 0.0` and `center_of_mass = zeros(3)` for
   kinematic-only links**, and the setters are no-ops. Reading a kinematic object's mass tells you
   nothing about what the asset authored.
5. **The kinematic pose cache goes stale silently.** `prims/rigid_kinematic_prim.py:111-115`
   `clear_kinematic_only_cache` has exactly two callers — `:68` (its own setter) and
   `prims/entity_prim.py:1056` (the *stopped* branch). Move an ancestor by any other route and
   `get_position_orientation` keeps returning the old world pose. **Inferred from the call graph, not
   measured.**
6. **`prims/xform_prim.py:247` vs `:256-257`.** A single pose write opens **two** `editing_usd`
   contexts — translate through `set_attribute` (`prims/prim_base.py:275-276`) and orient in its own
   block. Each exit runs `SynchronizeToFabric`, so there is an intermediate state with the new
   translation and the old orientation. Harmless single-threaded; do not assume a pose write is
   atomic in Fabric.

---

## How to reproduce the measurements

All static, host CPU, no Kit and no GPU. `usd-core` 0.26.8 from PyPI into a scratch prefix:

```sh
python3 -m pip install --target=$SCRATCH/pylibs usd-core numpy scipy
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
PYTHONPATH=$SCRATCH/pylibs python3 $SCRATCH/audit-transform_probe_hull.py \
    realm/robots/panda_robotiq/droid_robolab_padspring.usd     # per-geom displacement table
PYTHONPATH=$SCRATCH/pylibs python3 $SCRATCH/audit-transform_probe_sep.py \
    realm/robots/panda_robotiq/droid_robolab_padspring.usd     # pad-pad separation, OG vs correct
PYTHONPATH=$SCRATCH/pylibs python3 $SCRATCH/audit-transform_probe_ops.py \
    custom_assets/impact_drawer/usd/cabinet.usd \
    realm/robots/panda_robotiq/droid_robolab_padspring.usd     # xformOpOrder + upAxis inventory
```

The probes reproduce `_compute_points_on_convex_hull` exactly: they hull in the **local** frame, as
`prims/rigid_prim.py:558-574` does, before applying the link's world transform.
