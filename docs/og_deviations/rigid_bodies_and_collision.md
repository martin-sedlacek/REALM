# OmniGibson 3.9.1 vs Isaac Lab 2.2.0 — rigid bodies, mass and collision

What OmniGibson does to a rigid body's mass, inertia, centre of mass, collision shapes and contact
parameters that raw Isaac Sim / Isaac Lab does not — and what it does to values the **asset itself
authored**. Every row cites `file:line`.

Paths. OmniGibson is relative to `/mnt/home_lustre/sedlam56/projects/OG-lite_og391` (fork of
BEHAVIOR-1K OmniGibson 3.9.1; the port commit `25c73e1` is the stock baseline every "OG-lite only?"
answer was diffed against). REALM is relative to `/mnt/home_lustre/sedlam56/projects/REALM_og391`.
RoboLab is `/mnt/home_lustre/sedlam56/projects/RoboLab`. Isaac Lab paths are **in-SIF absolute**
under `/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif`, rooted at
`/workspace/isaaclab/source/isaaclab/isaaclab/`. PhysX schema defaults are from
`realm_og391.sif`, `.../isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/generatedSchema.usda`.

**Status.** Complete for the files in scope (`prims/rigid_prim.py`, `rigid_dynamic_prim.py`,
`rigid_kinematic_prim.py`, `geom_prim.py`, `cloth_prim.py`, `material_prim.py`, `objects/*`). The
Isaac Lab contrast is a verified in-SIF extraction. Asset-side numbers are measured with `usd-core`
on the host CPU against the real USDs — see [What was measured](#what-was-measured-and-how).

Companion chapter: `docs/og_deviations/control_and_actuation.md`.

---

## The headline: `frame="parent"` is not the link frame — a pattern, not an incident

`grep 'frame="parent"'` over `omnigibson/` finds **three** sites that compose a geom's transform to
the geom's *immediate parent* while the surrounding code treats the result as being in the **link**
frame. **One of the three is patched.**

The premise all three share is false. `XFormPrim.get_position_orientation(frame="parent")`
(`prims/xform_prim.py:283`) returns `get_local_pose(self.prim_path)` — one level up, full stop. That
equals the link frame only when the geom is a direct child of the link. `GEOM_TYPES` is
`{"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}` (`utils/constants.py:100`) and **excludes `Xform`**,
so whenever an intermediate `Xform` carries the collision APIs and the `Mesh` sits beneath it, the
wrapped geom is the `Mesh` and the `Xform`→link transform is silently dropped.

That is exactly how REALM's Robotiq 2F-85 is built. **Measured** on
`realm/robots/panda_robotiq/droid_robolab_v2.usd`: all nine gripper links carry
`PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` on a `Defeatured_*_01` **Xform**, with the `Mesh`
one level below; the eight `panda_link*` arm links instead carry them on the `Mesh` itself, under an
identity `geometry` Xform. So the same loader is correct on one half of the robot and wrong on the
other — which is why nothing caught it for so long.

### Site 1 — `prims/rigid_prim.py:324`, `:332` — centre of mass. **Patched, narrowly.**

```python
local_pos, local_orn = mesh.get_position_orientation(frame="parent")   # rigid_prim.py:324
...
coms.append(T.quat2mat(local_orn) @ (com * mesh.scale) + local_pos)    # rigid_prim.py:332
```

On the 2F-85 the dropped transform is a 90° (left) / 180° (right) rotation — the **mirror**. Every
left/right link pair therefore came out with an identical CoM *including the sign of y*, which is
impossible for a mirrored pair, and the fingertip pads landed **128.347 mm** from their true
centroid. That displacement enters each pad's inertia about its own pivot as `m·d²` =
**1.496e-04 kg·m²** against a true **1.937e-06** — an inflation of **77.3×**. A PhysX mimic joint
realises stiffness `k ~ ω²·I`, so at the authored `naturalFrequency` the fingertips came out ~77×
too stiff and would not curl under load. *(Established by the CoM-fix chain; see `CHANGE_LEDGER.md`.
Not re-derived here.)*

A diagnostic worth keeping: a wrong CoM shared by a mirrored pair gives the two sides **different**
effective inertias where symmetry demands they match — `nf_eq` 253 vs 217 (commit `ab28282`).

Fixed in OG-lite `83b21d5` → `6d04cc9` → `0fed598` → `15b4072` (`rigid_prim.py:325-330`), gated on
gripper **structure** rather than a link name (`rigid_prim.py:68-75`): the link's parent must also
carry `left_inner_finger` and `right_inner_finger`, which no BEHAVIOR object does. Scope is now
**eight** links; `base_link` was dropped in `0fed598` and the drop **measured** neutral in `15b4072`
(`curl_deg` +0.3280 L / +0.3887 R at the authored `nf=1000`, identical to four decimals with and
without it) because the 2F-85's `base_link` is welded to `panda_link8` and sits in the grounded
chain.

### Site 2 — `prims/geom_prim.py:250` — the collision and visual hulls. **Not patched.**

```python
position, orientation = self.get_position_orientation(frame="parent")   # geom_prim.py:250
```

`points_in_parent_frame` feeds `RigidPrim._compute_points_on_convex_hull` (`rigid_prim.py:559`),
which stores the result as `collision_boundary_points_local` / `visual_boundary_points_local`
(`rigid_prim.py:596`, `:577`) — *local* meaning the **link** frame — and
`{collision,visual}_boundary_points_world` (`rigid_prim.py:604`, `:585`) then applies the **link's**
world transform to them. Same false premise, second consumer.

**Measured in this run** (static, `usd-core`, `droid_robolab_v2.usd` at its authored pose;
`audit-rigid_measure_frames.py`) — OmniGibson's world-frame hull versus the true one:

| link | collision-hull AABB centre error | AABB extent error |
|---|---:|---:|
| `panda_link0` … `panda_link7` | **0.00 mm** | 0.00 mm |
| `base_link` | 61.09 mm | 8.48 mm |
| `left_outer_knuckle` | 101.16 mm | 25.20 mm |
| `left_inner_knuckle` | 125.69 mm | 7.07 mm |
| `right_inner_finger` | 133.64 mm | 25.66 mm |
| `left_inner_finger` | 133.79 mm | 26.03 mm |
| `left_outer_finger` | 137.35 mm | 31.72 mm |
| `right_outer_knuckle` | 141.77 mm | 24.97 mm |
| `right_inner_knuckle` | 161.69 mm | 5.52 mm |
| `right_outer_finger` | **192.66 mm** | 31.80 mm |

The visual hull is displaced identically (the `visual__Defeatured_*_01` Xforms carry the same
transforms), so `visual_aabb` is wrong by the same amounts.

This site is **measured broken on the loader-patched engine**, not merely inferred from `grep`
(commit `0ed25c9`): the ratio of a hull-derived observable to its hull-free control, over every
press of three runs, was `-0.49 .. -1.06` under OG-lite `83b21d5` — as backwards as the
`-0.57 .. -0.87` of the unpatched stock loader — against `+0.969 .. +0.995` for the asset-side fix.
**Patching `rigid_prim.py` demonstrably does not repair the hull.** A 116 mm hull offset from the pad
link origins (commit `3bf4ab9`) is what made hull-derived tip separation read **backwards** and cost
hours of direction confusion.

**This site reaches live REALM code.** `USDObject.get_base_aligned_bbox` consumes
`link.visual_boundary_points_world` / `link.collision_boundary_points_world` at
`objects/usd_object.py:1058-1060`, and REALM calls it during object replacement at
`realm/environments/perturbations/_helpers.py:200` and
`realm/environments/perturbations/sb_vrb.py:74`. Any object whose collision geometry sits under an
intermediate Xform can get a wrong bounding box during a perturbation. Not quantified on any
BEHAVIOR object.

### Site 3 — `utils/object_utils.py:88` — `compute_base_aligned_bboxes`. **Not patched.**

```python
local_pos, local_orn = mesh.get_position_orientation(frame="parent")
pts_in_link_frame.append(get_particle_positions_from_frame(local_pos, local_orn, mesh.scale, pts))
```

The variable is literally named `pts_in_link_frame`. A third independent copy of the same
composition. Its only caller is `compute_kinematic_metadata` (`utils/object_utils.py:130`), which is
offline asset-authoring tooling, so this one does not reach a running episode — but it would bake the
error into any metadata regenerated with it.

### Why this is a pattern and not three bugs

All three would be repaired at once by removing the dropped transform from the **asset**; the loader
patch compensates in exactly one. That is a correctness argument for the asset-side route, not just a
maintenance one.

**How to find the next one:** any code that calls `get_position_orientation(frame="parent")` on a
geom and then treats the result as link-frame. The other `frame="parent"` hits in the tree
(`robots/robot.py:1986`, `object_states/particle_modifier.py:443`, `:497`, `:1146`,
`systems/macro_particle_system.py:891`, `:904`, `object_states/attached_to.py:206`,
`prims/entity_prim.py:349`) operate on prims that *are* direct children of what they compose to, or
round-trip a get/set pair through the same frame. They were checked, not assumed.

---

## The second theme: OmniGibson computes, then overwrites

`update_meshes()` ends with

```python
self.center_of_mass = com          # rigid_prim.py:364
```

The setter (`rigid_dynamic_prim.py:203-210`) calls `RigidPrimView.set_coms`, whose
**stopped-simulation** branch (`utils/deprecated_utils.py:1006-1021`) writes `physics:centerOfMass`
straight onto the prim through the **scene stage's edit target** — a local opinion that no
composition arc from a referenced layer can outrank.

**An asset cannot defend its centre of mass.** Measured (commits `ab28282` / `6541869`): authoring
`physics:centerOfMass` on the gripper links changed nothing, while `physics:mass`,
`physics:diagonalInertia` and `physics:principalAxes` **are** consumed verbatim — reproducing
RoboLab's runtime inertia tensors to **0.00062%** on all nine gripper links. PhysX does not
parallel-axis-shift an authored tensor even when it then accepts a CoM 128 mm away, which is why the
wrong CoM survives alongside a right-looking inertia.

Isaac Lab has no counterpart. `MassPropertiesCfg` has exactly two fields, `mass` and `density`
(in-SIF `sim/schemas/schemas_cfg.py:159-181`), and `modify_mass_properties` loops over
`["mass", "density"]` and nothing else (`sim/schemas/schemas.py:458-462`). The strings
`centerOfMass`, `center_of_mass`, `diagonalInertia` and `principalAxes` **do not occur anywhere** in
the Isaac Lab source tree; `set_coms` appears only in the optional runtime event
`randomize_rigid_body_com` (`envs/mdp/events.py:394`, which reads the current value and adds a delta)
and in tests. RoboLab's `droid.py` passes no `mass_props` at all
(`robolab/robots/droid.py:69-77`), so PhysX gets what the USD says.

---

## The third: "the collider is the Xform" defeats the physics parser too

Every `Defeatured_*` **Mesh** in both REALM's and RoboLab's 2F-85 authors the real CAD
mass / CoM / diagonal inertia via `UsdPhysics.MassAPI`, and **nothing reads them**, in either stack:
`CollisionAPI` is on the `Defeatured_*_01` **Xform**, so the `Mesh` is not the collider prim and its
`MassAPI` is not the collider's. Aggregated, those authored numbers give the pad **0.0392547 kg**
where the body PhysX actually builds is **0.00951321 kg** — in REALM and in
RoboLab-through-Isaac-Lab alike (commit `ab28282`). Not gripper-specific: any asset whose
`CollisionAPI` sits on an Xform above its Gprims loses those Gprims' authored mass properties.

It is also a **trap**: those CAD numbers look exactly like the data you want, and are 1.45× the
realised tensor for the pad, mass-normalised.

The same shape defeats two OmniGibson code paths — rows 5 and 6 below — because `_find_geom_prims`
records `MeshCollisionAPI` only for prims whose type is in `GEOM_TYPES` (`rigid_prim.py:262-265`).

---

## Table

`class`: **A** = deviation from an Isaac/PhysX default; **B** = override of a value the asset
authored; **C** = constraint imposed on assets (no opt-out); **D** = latent defect.

| # | site (`file:line`) | what OmniGibson does | what Isaac Lab does, or what the asset authored | silent? | class | measured impact | OG-lite only? |
|---|---|---|---|---|---|---|---|
| 1 | `prims/rigid_prim.py:324`, `:332` | composes each collision geom's CoM to the geom's **immediate parent**; the comment above claims the link frame | n/a — a defect, not a policy | **yes** | D | pads **128.347 mm** off true centroid; pad inertia about its pivot **77.3×** inflated (1.496e-04 vs 1.937e-06); left/right CoMs identical *including sign of y* | no — stock. **Patched** by OG-lite `83b21d5`→`15b4072` at `:325-330`, scoped to 8 Robotiq links |
| 2 | `prims/geom_prim.py:250` | same composition in `points_in_parent_frame`; result consumed as link-frame at `rigid_prim.py:559`, `:577`, `:596` and world-transformed at `:585`, `:604` | n/a | **yes** | D | **measured this run**: collision- and visual-hull AABB centres off by **61.09–192.66 mm**, extents by up to **31.80 mm**, on all 9 Robotiq links; 0.00 mm on all 8 arm links. Still broken *on the patched engine* (`0ed25c9`) | no — stock, **unpatched**. Reaches live REALM code via `usd_object.py:1058-1060` |
| 3 | `utils/object_utils.py:88` | third copy of the same composition, into a variable named `pts_in_link_frame` | n/a | **yes** | D | not quantified; sole caller `object_utils.py:130` is offline metadata tooling | no — stock, unpatched |
| 4 | `prims/rigid_prim.py:364` → `utils/deprecated_utils.py:1006-1021` | writes the geometry-derived `physics:centerOfMass` into the **scene stage's edit target** on every load | Isaac Lab's schema path never writes a CoM: `MassPropertiesCfg` has only `mass`/`density` (`schemas_cfg.py:159-181`), `modify_mass_properties` loops only those two (`schemas.py:458-462`); `set_coms` exists only in the opt-in event `envs/mdp/events.py:394` | **yes** | B | authoring a CoM in the asset **changes nothing**; `mass`/`diagonalInertia`/`principalAxes` *are* honoured (0.00062% match to RoboLab) | no — stock |
| 5 | `prims/rigid_prim.py:262-265` gate + `:438` | `set_collision_approximation()` iterates `_mesh_collision_apis`, which `_find_geom_prims` fills only for prims whose **type** is in `GEOM_TYPES` | n/a | **yes — no error, no effect** | D | **measured**: on `droid_robolab_v2.usd` all 9 Robotiq links carry `MeshCollisionAPI` on the `Defeatured_*_01` Xform, so the list is empty and the call is a no-op for them; it works for the 8 `panda_link*`, whose API is on the Mesh | no — stock |
| 6 | `prims/rigid_prim.py:335-337` | oblong-shape → `boundingCube` fallback gated on `prim.HasAPI(MeshCollisionAPI)` on the **Mesh** | n/a | **yes** | D | same blind spot as row 5: never fires for a link whose `MeshCollisionAPI` sits on an Xform | no — stock |
| 7 | `prims/rigid_prim.py:258-259`, `:106-107`, `:276-277` | intends to overwrite `contactOffset` → **0.001** and `restOffset` → **0.0**, but collects targets via `HasAPI(PhysxSchema.PhysxCollisionAPI)` | PhysX schema defaults are `physxCollision:contactOffset = -inf` and `restOffset = -inf`, i.e. "picked by the simulation from the shape extent" (`generatedSchema.usda:516`, `:530`). Isaac Lab's `CollisionPropertiesCfg.contact_offset`/`rest_offset` default to `None` (`schemas_cfg.py:131`, `:139`) = keep authored; RoboLab passes no `collision_props` (`droid.py:69-77`) | **yes** | D | **RESOLVED — this is dead code.** `PhysxSchema` declares **no** auto-apply (`plugInfo.json`: 0 hits for `autoApply`; no `apiSchemaAutoApplyTo` in `generatedSchema.usda`) and **no** asset inspected lists any `Physx*` API in `apiSchemas`. So `_physx_collision_apis` is empty, the write never fires — and neither do `set_contact_offset`, `set_rest_offset`, `set_torsional_patch_radius`, `set_min_torsional_patch_radius` (`rigid_prim.py:382-427`). The asset's authored `contactOffset = 0.02` is equally inert | no — stock |
| 8 | `prims/rigid_prim.py:194-197` | any link with **no collision meshes** gets `mass = 1e-6`, `density = 0.0`, unconditionally | Isaac Lab `MassPropertiesCfg.mass` defaults to `None` (`schemas_cfg.py:169`), and `sim/utils.py:62-64` returns early on `None` so the authored value stands | **yes** | B | **measured**: `realm/robots/ur5/ur5e_robotiq.usd` `tool0` authors `physics:mass = 1e-4` with zero child prims → replaced by `1e-6`, 100× lighter. Same for its `base_link` | no — stock |
| 9 | `objects/usd_object.py:64-65`, `:419-420` | sets `solverPositionIterationCount = 32`, `solverVelocityIterationCount = 1` on every non-cloth, non-kinematic USDObject, written to `physxArticulation:*` via `entity_prim.py:1266-1281`, `:1297-1312` | RoboLab asks the articulation for **64 / 0** (`droid.py:73-76`) but caps the *scene* at `max_position_iteration_count = 32`, `max_velocity_iteration_count = 1` (`robolab/core/environments/base.py:179-180` → `simulation_context.py:775-779`), so its **effective** position count is also 32 | **yes** | B | **measured**: `droid_robolab_v2.usd` authors **64 / 64** on `/panda` → OmniGibson runs 32 / 1. Stock OG's `franka_panda.usda` and all four BEHAVIOR objects sampled already author 32 / 1, so this is a **no-op on OmniGibson-native assets** and bites only foreign ones. Net difference vs RoboLab is velocity iterations only (1 vs 0) | no — stock |
| 10 | `prims/entity_prim.py:30`, `:85-86` | sets `sleep_threshold = 5e-5` on every non-kinematic entity (articulation-level via `entity_prim.py:1376-1391`) | **the PhysX schema default is already 0.00005** for both `physxRigidBody:sleepThreshold` (`generatedSchema.usda:464`) and `physxArticulation:sleepThreshold` (`:1464`); Isaac Lab's cfg default is `None` = keep authored (`schemas_cfg.py:110`, `:34`) | **yes** | B | **corrects an earlier assumption**: this is *not* a deviation from the PhysX default, it re-asserts it. It only bites an asset that authored otherwise — `droid_robolab_v2.usd` authors `physxRigidBody:sleepThreshold = 0.5` on three arm links, but for an articulation PhysX uses the articulation value, which the asset does not author. Net effect on these assets: none measured | no — stock |
| 11 | `objects/usd_object.py:330-332` | **always** writes `physxArticulation:enabledSelfCollisions` from the Python `self_collisions` kwarg — `False` for USDObject/DatasetObject, `True` for `Robot` (`robots/robot.py:115`) | PhysX schema default is **`True`** (`generatedSchema.usda:1460`); Isaac Lab's `enabled_self_collisions` defaults to `None` = keep authored (`schemas_cfg.py:25`); RoboLab explicitly asks `False` (`droid.py:74`) | **yes** | C | **measured**: `droid_robolab_v2.usd`, stock `franka_panda.usda` and all four BEHAVIOR objects author `False`. REALM's robot definitions ask `true` (e.g. `realm/robots/definitions/droid_robolab_v2/droid_robolab_v2.yaml:41`), so the robot runs with self-collisions **on** where RoboLab runs it off. There is no "leave the asset's value" option | no — stock |
| 12 | `objects/usd_object.py:261-262`, `:327-329` | strips `ArticulationRootAPI` **and** `PhysxArticulationAPI` from every prim in the asset, then re-applies to a prim it computes itself | Isaac Lab applies articulation props to the prim the cfg names, and never strips | **yes** | C | the asset loses any say in *where* its articulation root lives. Note `RemoveAPI` edits `apiSchemas` only — the authored `physxArticulation:*` **values** survive and are then overwritten by rows 9 and 11 | no — stock (the root-choice heuristic around it was relaxed by OG-lite `7c59ed5`) |
| 13 | `objects/dataset_object.py:289`, `:296` | with `gm.FORCE_CATEGORY_MASS = True` (`macros.py:282`), distributes the category-average mass over links by volume fraction — but the denominator `total_volume` sums **all** links while mass is assigned only to links with collision meshes | n/a — OmniGibson-specific policy | **yes** | D | **measured this run** (`audit-rigid_meta_volume.py`): meta links return their *visual* volume (`rigid_prim.py:518`), so the object receives only `(1 − meta_share)` of its category mass. `bottom_cabinet/glefdh` **36.25%**, `microwave/vuezel` **75.24%**, `stove/yhjzwg` 99.73%, `breakfast_table/lcsizg` 100% | no — stock |
| 14 | `objects/dataset_object.py:299-300` | with `FORCE_CATEGORY_MASS = False`, sets `mass = 0.0` and `density = category_density` on every collision link | Isaac Lab `MassPropertiesCfg.density` defaults to `None` (`schemas_cfg.py:176`) | **yes** | B | **measured**: every BEHAVIOR object sampled authors `physics:mass = 1.0`, a placeholder — so overriding it is the point. Listed for completeness | no — stock |
| 15 | `objects/dataset_object.py:303-308` | applies `ig:centerOfMass` to `root_link.center_of_mass` with the comment "we do NOT need to apply a scale" | the geometry-derived CoM path **does** scale (`prims/rigid_prim.py:332`, `com * mesh.scale`) | **yes** | D | the two CoM paths disagree about whether the value is in scaled units. Unmeasured — none of the four objects sampled authors `ig:centerOfMass` | no — stock |
| 16 | `utils/usd_utils.py:1223-1230` | on any `convexHull` approximation, forces `physxConvexHullCollision:hullVertexLimit = 60` | Isaac Lab 2.2 has **no** `MeshCollisionPropertiesCfg` (not exported by `sim/schemas/__init__.py:52-59`); its mesh converter defaults to `convexDecomposition` (`sim/converters/mesh_converter_cfg.py:36`) and never touches the hull vertex limit | **yes** | C | clamps an authored limit; nothing in the assets sampled authors one | no — stock |
| 17 | `utils/usd_utils.py:2455-2462` | rejects a mesh as "too flat **in the world frame**" when `min_extent < 1e-5`, but `min_extent` comes from `geom_prim.extent` (`geom_prim.py:292-299`), documented as "the **unscaled** 3d extent … in its local frame" | n/a | **yes** | D | **measured**: the 8 arm collision meshes are authored in centimetre-ish units with `xformOp:scale = 0.01`, so their local `min_extent` is **5.48–16.60** against a 1e-5 threshold — the test passes trivially, and would also pass for a genuinely 1e-5 m-thin shape. Extent/radius ratios 2.55–15.06, well under the 95 cutoff | no — stock |
| 18 | `prims/rigid_prim.py:183-186` | applies `PhysxContactReportAPI` with `threshold = 0.0` to **every** non-visual-only link | Isaac Lab gates contact reporting per asset (`activate_contact_sensors=True`, `droid.py:68`) | **yes** | A | not measured (a cost, not a correctness, deviation) | the `_contact_reporting_wanted` opt-out at `:183` and `:537-549` (`gm.CONTACT_REPORTING_PATTERNS`, `macros.py:231`) is **OG-lite**; the unconditional apply is stock |
| 19 | `simulator.py:574-577` + `scenes/scene_base.py:713-726` | creates a self-filtering `fixed_base_fixed_links` group and puts every fixed link of every fixed-base non-robot object into it; plus `structural_doors` filtered against it | no Isaac Lab equivalent — collision filtering there is explicit per scene | **yes** | A | not measured | no — stock |
| 20 | `prims/rigid_prim.py:331-332` | scales the geom's volume and CoM by `mesh.scale`, the geom's **own** `xformOp:scale` only (`xform_prim.py:420-431`), not the accumulated scale — though `XFormPrim.world_scale` exists | n/a | **yes** | D | **measured**: latent on the assets in hand — every intermediate `Defeatured_*_01` and `geometry` Xform has scale `(1,1,1)`. The OG-lite fix at `:325-330` corrects position and orientation but **not** this | no — stock; not covered by the OG-lite patch |
| 21 | `prims/geom_prim.py:338` | `get_applied_physics_material()` reads `GetDirectBinding(materialPurpose="physics")` on the **geom**, which ignores bindings inherited from ancestors | USD resolves physics-material bindings down the namespace; PhysX uses the resolved one | **yes** | D | **measured**: in `droid_robolab_v2.usd` the pads' *direct* binding is empty but `ComputeBoundMaterial(materialPurpose="physics")` returns `/panda/PhysicsMaterial` (`staticFriction = 2.0`, `dynamicFriction = 2.0`, `frictionCombineMode = max`), bound one level up on `/panda/{left,right}_inner_finger`. OmniGibson reports "no physics material" for a surface PhysX is running at µ = 2.0 | no — stock |

---

## Benign / by design — checked, do not re-check

- **Sleep threshold** (row 10). `5e-5` **is** the PhysX schema default for both
  `physxRigidBody:sleepThreshold` and `physxArticulation:sleepThreshold`
  (`generatedSchema.usda:464`, `:1464`). OmniGibson re-asserts it rather than deviating from it.
- **Primitive collision approximation.** `objects/primitive_object.py:186-193` picks
  `boundingSphere` for Sphere, `boundingCube` for Cube, `convexHull` otherwise. Isaac Lab's mesh
  spawners do **exactly the same** (`sim/spawners/meshes/meshes.py:334-344`). Identical policy, not a
  deviation.
- **Solver iteration counts on OmniGibson-native assets** (row 9). Stock `franka_panda.usda` and all
  four BEHAVIOR objects sampled already author 32 / 1; OmniGibson's "default" is baked into its own
  asset pipeline. And RoboLab's own scene cap makes its effective position count 32 as well, so the
  only real difference against RoboLab is 1 velocity iteration vs 0.
- **`enabledSelfCollisions` on dataset objects** (row 11). `DatasetObject`'s default `False` matches
  the `False` every sampled asset authors. Only `Robot` (default `True`, `robots/robot.py:115`)
  changes anything.
- **Overriding `physics:mass` on BEHAVIOR objects** (row 14). Every dataset object sampled authors
  `physics:mass = 1.0` on every link — a placeholder, not data.
- **`physics:approximation`.** Every BEHAVIOR collision mesh sampled already authors `convexHull`
  (46/46, 35/35, 14/14, 6/6 on the four objects), as does `droid_robolab_v2.usd` (19/19). OmniGibson
  neither needs to nor does set it on the normal load path; `apply_collision_approximation` runs only
  from `setup_collision_apis` (`usd_utils.py:1176`, used by
  `scenes/static_traversable_scene.py:91` and `systems/macro_particle_system.py:1232`), the oblong
  fallback, `PrimitiveObject`, and explicit caller requests.
- **`prims/material_prim.py` contains no physics material at all** — it is entirely MDL/visual
  (`OmniPBRMaterialPrim`, `VRayMaterialPrim`, `OmniSurfaceMaterialPrim`). OmniGibson creates a
  `PhysicsMaterial` only from an explicit `link_physics_materials` dict
  (`objects/usd_object.py:423-434`) or `Robot`'s `finger_static_friction` /
  `finger_dynamic_friction` (`robots/robot.py:356-364`). It never installs a default physics
  material — and neither does RoboLab, which passes no `physics_material` in its spawn cfg. So an
  unbound collider falls through to PhysX's scene default in **both** stacks. Isaac Lab's
  `RigidBodyMaterialCfg` (in-SIF `sim/spawners/materials/physics_materials_cfg.py:30`) would give
  `static_friction = 0.5`, `dynamic_friction = 0.5`, `restitution = 0.0`, combine modes `"average"`,
  `compliant_contact_stiffness = 0.0`, `compliant_contact_damping = 0.0` (`:42-78`) — but only when a
  spawner asks for one. The only OG-lite change in `material_prim.py` is `1dcc5bb`
  (`OmniSurfaceMaterialPrim.preset_name` default), visual only.
- **Applying `PhysxRigidBodyAPI` to every link** (`rigid_prim.py:168`). Isaac Lab's
  `modify_rigid_body_properties` does the same (`sim/schemas/schemas.py:273-280`). Note the side
  effect, which is shared: applying the schema makes previously-dormant authored `physxRigidBody:*`
  attribute values live.
- **`RigidKinematicPrim`'s no-op mass/CoM/density/gravity** (`prims/rigid_kinematic_prim.py:123-181`)
  returning `0.0` / `zeros(3)` and swallowing writes. Correct: a kinematic body has no dynamics. Just
  do not read a mass off a kinematic link and believe it.
- **`RigidPrim.volume` uses visual meshes for meta links** (`rigid_prim.py:518`). Correct on its own
  terms — a meta link has no collision geometry. It is the *denominator* at
  `dataset_object.py:289` that makes it wrong (row 13), not this property.
- **Cloth remeshing** (`prims/cloth_prim.py:111-124`) rewrites the mesh at load whenever the scale is
  non-unit or the asset predates the current dataset. It is loud (`log.warning` at `:121`),
  invalidates the cached settled/folded/crumpled configurations by design, and has no Isaac Lab
  counterpart. Not a silent deviation.

---

## Traps that are not deviations, but will mislead you

- **"The pads have no physics material" is false.** The binding is on the **link**
  (`/panda/{left,right}_inner_finger` → `/panda/PhysicsMaterial`, µs = µd = 2.0, combine mode `max`)
  and USD resolves it down to the collider. Only `GetDirectBinding` — which is what
  `geom_prim.py:338` calls — comes back empty; `ComputeBoundMaterial(materialPurpose="physics")` on
  the pad Mesh returns `/panda/PhysicsMaterial`. **This corrects an earlier note in this project that
  treated the empty binding as normal on both stacks.**
- **`PhysxSchema` API schemas are never auto-applied.** `plugInfo.json` for the PhysxSchema plugin
  contains zero occurrences of `autoApply`, and `generatedSchema.usda` declares no
  `apiSchemaAutoApplyTo`. Combined with the fact that `droid_robolab_v2.usd`, stock
  `franka_panda.usda` and the BEHAVIOR objects list **no** `Physx*` API in `apiSchemas` while still
  authoring `physxCollision:contactOffset`, `physxRigidBody:sleepThreshold`,
  `physxArticulation:solverPositionIterationCount` and more, this means those authored values are
  **inert unless something applies the schema**. It is what makes row 7 dead code, and it is the
  reason a `HasAPI` check and a "the attribute is authored" check disagree here.
- **`p.RemoveAPI(...)` at `usd_object.py:261-262` does not delete authored attribute values** — it
  edits `apiSchemas`. The `physxArticulation:*` values survive and are overwritten later instead.
- **`RigidDynamicPrim.mass` silently falls back to `volume × density`** when PhysX reports zero
  (`rigid_dynamic_prim.py:213-224`), and `density` falls back to a hardcoded **1000.0** when both are
  zero (`:236-252`). A mass read out of OmniGibson is not necessarily a mass PhysX has.
- **`self._collision_meshes` is keyed on the bare prim name** (`rigid_prim.py:281`,
  `mesh_name = prim.GetName()`). Two collision geoms under one link sharing a name would silently
  collapse to one — affecting `volume`, the hulls and `set_collision_approximation`, though **not**
  the CoM, which accumulates in the `coms`/`vols` lists. Not observed in any asset sampled; the
  2F-85's two-geom links use distinct names.
- **`RigidPrim.volume` (`:518-519`) and the `vols` list in `update_meshes` (`:331`) compute volume
  differently** — the former with `world_frame=True` (full accumulated transform), the latter as
  local volume × the geom's own scale. They feed different consumers and can disagree.
- **RoboLab's `PhysxCfg` block drops 10 of its 18 keys, and that is RoboLab's doing, not Isaac
  Lab's.** `robolab/core/environments/base.py:171-190` lists both Isaac Lab 2.2 and 2.3 field names
  and guards each with `if hasattr(physx, attr_name)` (`:192-194`). Isaac Lab 2.2's `PhysxCfg` has 22
  fields (`sim/simulation_cfg.py:20-161`) and **none of them are silently dropped** — every one
  reaches PhysX, via `physics_context.py:153-187` or `simulation_context.py:742-787`. The keys
  RoboLab loses are the ones that are not `PhysxCfg` fields at all: `contact_offset`, `rest_offset`,
  `num_position_iterations`, `num_velocity_iterations`, `max_depenetration_velocity`, `num_threads`,
  `relaxation`, `warm_start`, `shape_collision_distance`, `shape_collision_margin`. In particular
  **`contact_offset` and `rest_offset` do not exist on `PhysxCfg`** (zero hits in
  `sim/simulation_cfg.py`) — there is no scene-level contact offset in Isaac Lab; offsets are
  per-collider on `CollisionPropertiesCfg`, defaulting to `None`. So RoboLab's `contact_offset: 0.02`
  / `rest_offset: 0.01` at `:176-177` never took effect.
- **Isaac Lab's only contact/rest-offset write is opt-in and runtime.**
  `randomize_rigid_body_collider_offsets` (`envs/mdp/events.py:397`) calls
  `root_physx_view.set_rest_offsets` / `set_contact_offsets` and skips entirely when its distribution
  params are `None`, which is their default.
- **REALM's `update_robot_physics`** (`realm/environments/env_dynamic.py:270-277`) reads
  `physxMeshCollision:approximation`; the assets author `physics:approximation` (the
  `UsdPhysics.MeshCollisionAPI` attribute), which is also what OmniGibson writes
  (`usd_utils.py:1232`). Different attribute name — REALM-side, noted so it is not mistaken for
  OmniGibson behaviour.

---

## What was measured, and how

Asset-side numbers come from static inspection with `usd-core` 26.08 on the host CPU (no Kit, no
GPU) against the real USDs. Scripts are in the shared agent scratchpad, prefixed `audit-rigid_`:

| script | what it produces |
|---|---|
| `audit-rigid_measure_frames.py` | the row-2 hull-error table: replicates `points_in_parent_frame` → link-world exactly, compares against the true geom→world transform. AABB errors use every vertex, so no centroid approximation enters |
| `audit-rigid_meta_volume.py` | the row-13 meta-link mass deficit, on four BEHAVIOR objects decrypted read-only into the scratchpad. Volumes are raw triangle-fan signed volumes × the geom's own scale, where OmniGibson uses trimesh with `world_frame=True` and a convex-hull fallback — so the **ratio** is the finding, not the absolute volume |

Assets inspected: `realm/robots/panda_robotiq/droid_robolab_v2.usd`;
`realm/robots/ur5/ur5e_robotiq.usd` (its references to remote Isaac assets do not resolve offline —
only locally-defined prims were used);
`data/datasets/omnigibson-robot-assets/models/franka/franka_panda/usd/franka_panda.usda`; BEHAVIOR
objects `bottom_cabinet/glefdh`, `breakfast_table/lcsizg`, `microwave/vuezel`, `stove/yhjzwg`.

Numbers attributed to commits (`83b21d5`, `ab28282`, `0ed25c9`, `3bf4ab9`, `6541869`, `15b4072`)
were established elsewhere in this project and are **not** re-derived here; see `CHANGE_LEDGER.md`.

Every "OG-lite only?" answer was decided by `git diff 25c73e1 HEAD -- <file>`, where `25c73e1` is the
BEHAVIOR-1K 3.9.1 port commit. `rigid_dynamic_prim.py`, `rigid_kinematic_prim.py`, `geom_prim.py`,
`cloth_prim.py`, `dataset_object.py` and `primitive_object.py` are **byte-identical to the port**, so
every finding in them is stock upstream behaviour. In `rigid_prim.py` the only OG-lite changes are
the CoM fix (`:28-75`, `:325-330`) and the contact-reporting gate (`:86-103`, `:183`, `:537-549`); in
`usd_object.py` and `entity_prim.py` none of the lines cited above are OG-lite.

---

## Not covered

- Sleep/wake behaviour at runtime, contact-report throughput cost, and the cloth particle system
  beyond the remesh trigger.
- Whether `omni.physx`'s USD parser reads `physxCollision:*` attributes when the corresponding API
  schema is **not** applied. USD-schema semantics say it should not; confirming it would need one
  print from a running Kit session. It does not change any OmniGibson-side conclusion above — row 7
  is dead code either way — but it decides whether the asset's authored `contactOffset = 0.02` is
  live or inert.
- Row 15 (`ig:centerOfMass` and scale) and rows 16, 18, 19, 20: mechanism established from the code,
  no measurement.
