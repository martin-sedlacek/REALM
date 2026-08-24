# Perturbations in a vector env

State as of 2026-08-13. Companion to `README.md` (which covers how the vector env is *built*) and
`SCALING.md` (throughput). This file covers what had to change for REALM's perturbations to be
correct when N scenes share one simulator, and what is still open.

These findings were verified with a temporary vector perturbation probe; its sweep fanned
several perturbations across several interactive allocations.

## Why perturbations were the hard part

Everything a perturbation touches is per-member — its own scene, its own objects, its own
instruction. But three of the operations it uses are **global**:

| operation | acts on |
|---|---|
| `og.sim.stop()` / `og.sim.play()` | every scene |
| `og.sim.step()` / `og.sim.render()` | every scene |
| `og.sim.dump_state()`, `og.sim._objects_to_initialize` | every scene |

REALM applies perturbations **per member inside `reset()`**, so every one of those was being issued
N times, each time disturbing the other N-1 members mid-reset.

`RealmVectorEnvironment.reset()` therefore hoists them out of the per-member loop:

```
1. every member restores its own scene                   (no global state touched)
2. ONE joint-reset loop for every member that asked      (drawer tasks only)
3. ONE og.sim.stop(), only if a member's perturbation needs it
4. every member's perturbations run
5. repair the sim's object-init queue, then ONE og.sim.play()   (see "eviction" below)
6. work the perturbations deferred because it needs a playing sim
7. ONE joint-reset loop again, for the perturbations that ask for one
8. ONE settle loop driving all members together, if any asked for it
9. every member re-takes its main-object scoring reference
```

Perturbations never call the global operations directly. They route through
`perturbations/_helpers.py` — `sim_stop`, `sim_play`, `sim_step`, `after_play`, `settle` — which
no-op or defer when `env.in_vec_env`. `reset_joints()` follows the same shape via
`environments/joint_reset.py`. **Single-env behaviour is unchanged**, which is what lets the
historical numbers stay comparable.

Note the shape `settle()` uses, and which `reset_joints()` copies: in a vector env it **raises a
flag** rather than no-opping, and `RealmVectorEnvironment.reset()` asserts nothing is left flagged
when it finishes. No-opping would let a perturbation that never settles silently acquire the shared
settle, and let a drawer reset that lands outside a drain point silently not happen at all.

Only perturbations that ADD or REMOVE objects need a stopped sim (`NEEDS_STOPPED_SIM` =
V-SC, VB-MOBJ, VSB-NOBJ, SB-VRB). Pose writes work fine on a live sim, so VB-POSE and V-VIEW cycle
nothing at all — a pose write never needed a stopped sim, it just always had one.

## The bugs, and why they kept hiding behind each other

Four distinct defects. Each was only reachable once the previous was fixed, which is worth knowing
before concluding a fix is "the" root cause.

### 1. Scene-relative coordinates written in the world frame

`set_position_orientation` defaults to `frame="world"`, but `spawn_bbox` comes from authored
constants in `scenes.yaml` and OmniGibson's loader offsets authored poses by `scene_position`
(`scene_base.py`: *"local then works out to exactly the authored pose"*). So the data is
scene-relative.

Scene 0's origin **is** the world origin, so writing world coordinates there is a no-op — invisible
single-env, and invisible in every historical REALM result. For every other member the object lands
in scene 0's tile. Scenes tile ~25.3 m apart; `gm.PROXIMITY_GATE_RADIUS` is 1.5 m; so the object
ends up far from its own robot, the proximity gate drops it from the contact view, and **a body that
is not a contact ROW can never register a grasp**.

Measured (VB-POSE Vec=4, main object world x per member): `-0.251 / -0.218 / -0.183 / -0.065`
before, `-0.251 / 25.050 / 50.310 / 75.676` after. Contact rows for scenes 1-3 went 40 → 51,
matching scene 0.

Fixed in `vb_pose.py`, `v_sc.py`, `sb_vrb.py`. Note `set_position()` and
`set_bbox_center_position_orientation()` are world-only with no frame argument, so those call sites
either move to `set_position_orientation(frame="scene")` or convert explicitly via
`scene.convert_scene_relative_pose_to_world()`.

### 2. A sibling scene evicting objects from the init queue

`Simulator._pre_remove_object` prunes the **global** `og.sim._objects_to_initialize` **by name
alone** (`omnigibson/simulator.py`). Names are unique per *scene*, not per simulator, and
every member is built from the same task YAML — so member 1's `remove_object("corkscrew")` evicts
**member 0's** freshly added corkscrew. It stays on the stage and in the registry, but nothing ever
initialises it.

Fingerprint: the repair names exactly the YAML-**named** distractors and none of the per-member
sampled `distractor_<i>` ones, whose names do not collide.

**Fixed upstream 2026-08-14** in OG-lite: `_pre_remove_object` now matches on **identity**, which is
strictly narrower than the name test and always the correct entry, since `scene.add_object` already
forbids two live same-named objects in one scene. Against the fork the repair below finds nothing.

`vec_init_queue.repair_init_queue()` is **kept**, as a net rather than a workaround, because
the OG-lite bind is optional and the fix does not travel with the image. `rr` defaults to
`MODE=stock`, and `MODE=stockfix` — the configuration `make_stock_patch.sh` exists to prepare and
that both build recipes wire in — binds only `scenes/scene_base.py`, so it still runs the stock
`simulator.py`. Under either, the eviction is live and this repair is the only thing between it and
an opaque `Object must be initialized before dumping state!` (or `prim view [...] is not a valid
view` out of `play()`) raised from an unrelated call site much later. It is two comprehensions per
reset that needs a stopped sim, and it announces itself loudly, so the presence or absence of its
warning now also tells you which OmniGibson a run used.

Measured with 2 environments, one warmup reset, and 3 perturbed resets:

| | stock `simulator.py` | OG-lite identity match |
| --- | --- | --- |
| V-SC | 4 x `Re-queueing 5 object(s) ... scene0/corkscrew, table_knife, wineglass, water_glass, bottle_of_wine` | **0** |
| VSB-NOBJ | 4 x `Re-queueing 1 object(s) ... scene0/cube` | **0** |

Both PASS either way — the repair worked; it just no longer has anything to repair. Per-member object
poses and contact-row counts are identical before and after, so the only thing that changed is that
nothing is evicted.

**Follow-up worth doing:** add the `_pre_remove_object` one-liner to `make_stock_patch.sh` and to the
two build recipes alongside the `scene_base.py` patch. Until that happens, `MODE=stockfix` and the
rebuilt SIF carry the bug, and the net above is what keeps them working.

#### The same pop has a second half: the corpse keeps its slot

One wrong pop causes two symmetric problems, and `repair_init_queue()` fixes both:

  **(a)** a **live** object is knocked off the queue and is never initialised — the case above;
  **(b)** the object that was actually **removed** stays *on* the queue, and the next `play()` runs
  `initialize()` on a prim that has already been deleted from the stage.

(b) only bites when the removed object was itself still pending, which needs a member to **add** an
object and then **remove** it inside ONE stopped window. SB-VRB is the only perturbation that does
that: on a task with no target (`pick_spoon`) it adds a `receiver` and then, if the new verb is
put/stack, `replace_obj()`s it. Measured on task 4, Vec=2 — member 1 removed its own brand-new
`receiver`, the pop took member 0's instead, and the batched `play()` then did:

```
File "omnigibson/simulator.py", line 1273, in _non_physics_step
  obj.initialize()
...
Exception: prim view ['/World/scene_1/receiver/base_link'] is not a valid view
```

The repair therefore drops the corpses **first**, so the queue is clean before it looks for orphans.
Telling a corpse from a live object cannot be done by asking whether a prim exists at its path —
`replace_obj` re-creates the replacement at the **same** relative prim path, so the path is occupied
again a moment later. (Tried; it silently disabled the whole repair and the crash came straight
back.) Identity against the scene registry is what distinguishes them, with the empty-prim-path test
kept only for the case where nothing holds the name at all — which also has to spare
`scene.add_object(..., register=False)` particle-system templates.

### 3. The repair running too late

`play()` initialises whatever is on the queue and **then** calls `update()` on every object's states
— and `update()` asserts the state is initialised. So an evicted object makes `play()` **itself**
raise `Cannot update uninitialized state.`, before any repair placed after `play()` could run.

The repair therefore happens while the sim is still **stopped**, and `play()` does the
initialisation itself, in `_non_physics_step`'s own order (queued objects first, state updates
second).

V-SC never exposed this because it replaces *distractors*; VSB-NOBJ replaces the **main object**,
which carries updatable states. Which object gets evicted decides whether the crash lands in
`play()` or later in `dump_state()`.

### 4. Touching a new object before it is initialised

In a vector env `sim_play()` is a no-op and `after_play()` **defers** its block until the shared
play. So anything referencing a just-created object must be **inside** `_post_play`. VSB-NOBJ set
`ToggleState.visual_marker.visible` outside it and died on `'NoneType' object has no attribute
'visible'` — the marker is None until the state initialises, which needs a playing sim.

This one the vectorization refactor **introduced**. It is the failure mode to expect from any future
perturbation that creates objects.

## The scoring reference: `mo_pos_orig`, `mo_rot_orig`, `mo_bbox_orig`

`mo_pos_orig` / `mo_rot_orig` are the **start-of-rollout** reference the progression stages are
judged against — `check_lift_and_distance_condition()` (LIFT_SLIGHT, LIFT_LARGE, PUSH) measures both
the lift `pos.z - mo_pos_orig.z` and the travel `‖pos - mo_pos_orig‖`, and `check_rotated()`
(ROTATED) measures against `mo_rot_orig`.

`RealmEnvironmentBase.__init__` seeds them from the task config, which is only right while
`main_objects[0]` is still the object the config declared. Three perturbations change that **during**
`reset()`, after the seed:

| perturbation | what it does to `main_objects[0]` |
|---|---|
| SB-NOUN | pops a random distractor and swaps it in (`sb_noun.py`) |
| VSB-NOBJ | replaces it with a freshly sampled object (`vsb_nobj.py`) |
| VB-MOBJ | replaces it with a rescaled copy (`vb_mobj.py`) |

Without a re-capture the reference described one object while the checks read another. Measured
2026-08-13, SB-NOUN on task 0, 6 resets: right after
`reset()` the reference sat 0.111–0.465 m (mean 0.285 m) from the object being scored, and
LIFT_SLIGHT answered True **at rest** on 3 of 6 resets — progression that never happened.

`RealmEnvironmentBase.capture_mo_reference()` re-takes both from the live object. It must be called
**only at the end of a reset, never while stepping**: it records where the object *started*, and a
reference that followed the object would drive both terms to zero and make every lift/distance check
permanently False — silently deleting the stage instead of fixing it. The reference probe's
`[FROZEN]` section tests that direction explicitly.

It is one method rather than a line in each perturbation so that a future perturbation that swaps
the object cannot forget it: every reset path ends there. The call sites are
`RealmEnvironmentDynamic.apply_perturbations()` (the phase that does the swapping, and the tail of
`reset()`) plus both warmups. `RealmVectorEnvironment.reset()` needs its own call, because
`apply_perturbations()` runs there before the shared play — exactly as it already needs its own
settle and its own deferred post-play drain.

### Why `mo_bbox_orig` is an anchor, not a live value

`mo_bbox_orig` is seeded on the line right after the other two and looks like it has the same
staleness shape. It does not, and `capture_mo_reference()` deliberately leaves it alone — for three
separate reasons, any one of which is enough:

- **It is an anchor, not a description of the current object.** Its only reader is VB-MOBJ, which
  computes `mo_bbox_orig * U(0.5,1.5)³` **every** reset and then rescales (`PrimitiveObject`) or
  removes-and-re-adds (`DatasetObject`) `main_objects[0]` at that size. Re-taking it would make each
  reset scale relative to the previous reset's already-scaled object — a multiplicative random walk
  that ends up pinned against `vb_mobj.py`'s `[0.02, 0.175] m` clip. Anchoring on the task config is
  what keeps VB-MOBJ's draw independent per reset, which is also what the harness's `size`
  observable assumes.
- **The staleness itself is unreachable.** The perturbations that re-point `main_objects[0]` at a
  *different* object are SB-NOUN and VSB-NOBJ, and REALM runs exactly one perturbation per process
  (`eval.py` builds `[SUPPORTED_PERTURBATIONS[perturbation_id]]`, `vector_eval.py`
  `[perturbation]`), so neither can ever precede VB-MOBJ. VB-MOBJ's own swap is
  same-category/same-model, and is the swap the anchor exists to survive.
- **There is nothing sound to capture.** For a `PrimitiveObject`, `vb_mobj.py` assigns this value to
  `mo.scale`, which is a scale *factor*; it only coincides with an extent because primitives are
  authored at scale 1. `get_position_orientation()` has no analogue for it.

If perturbations are ever **composed** — the same caveat `v_view.py` records — SB-NOUN followed by
VB-MOBJ would leave `mo_bbox_orig` describing an object that is no longer the target. The fix then
belongs in the perturbation that does the swapping (re-seed from the new object's *config*), not in
`capture_mo_reference()`: that method reads the live object, which is exactly what `mo_bbox_orig`
must not do.

### There is exactly one `replace_obj`

`RealmEnvironmentDynamic.replace_obj()` used to sit in `env_dynamic.py` alongside
`perturbations/object_sampling.replace_obj()`. It was a pre-refactor duplicate with **zero** call sites left
— every perturbation imports the `_helpers` one — and it still carried the bbox-centre-as-extent bug
that `_helpers` and `sb_vrb.py` have since fixed (a world-frame centre read as a half-width; §1).
Deleted rather than repaired, so there is only one copy to keep correct: the next person to wire up
"replace an object" has to find the live one.

## Testing: a pass is not evidence unless something asserts the effect

The historical vector perturbation probe checked that the sim cycles exactly as expected (0 for
pose-only, **exactly 1** for `NEEDS_STOPPED_SIM` — N is the original bug, 1 is the fix); each
member's main object is a contact ROW of its **own** scene; the grasp path runs; nothing leaves the
table; the instruction changes or does not; the scene is otherwise frozen; and — via `MOVES` — that
the perturbation still does what it claims.

That last one exists because **every other check passes trivially for a perturbation that has become
a no-op**: doing nothing leaves a perfectly healthy contact view. Measured: VB-MOBJ once passed with
object spread `0.0000` and no expectation recorded, which is indistinguishable from broken.

The observable has to match what the perturbation actually does, or it produces a confident **false
failure**:

| expectation | perturbations | observable |
|---|---|---|
| `objects` | VB-POSE | main-object xy across resets |
| `cameras` | V-VIEW | external sensor poses |
| `size` | VB-MOBJ | main-object AABB extent (pose is restored) |
| `identity` | VSB-NOBJ | `(category, model)` vs a pre-reset baseline |
| `distractors` | V-SC | the planted object set (main/target are `objects_to_skip`) |
| `nothing` | Default, V-AUG, V-LIGHT, S-* | control — must move nothing |

V-SC was briefly mapped to `objects`, which is a guaranteed false failure: it passes main and target
as `objects_to_skip` and re-randomises only distractors, so its main-object spread with a *working*
V-SC is exactly `0.0000`.

Two harness gotchas: **the historical probe exited 139 (SIGSEGV) whether it passed or failed** —
Isaac segfaults at teardown after the verdict prints, so grep `^PASSED`/`^FAILED` and never gate on
exit code. This is **script-specific, not universal**, and this sentence used to claim otherwise:
`examples/04_vector_evaluate.py` on this image does NOT segfault on a passing run. Measured over the
2026-08-18 matrix re-run, the only cell whose log contained a segfault or a traceback was
`8:VB-MOBJ`, which raises an intentional `NotImplementedError` — and a control build where nothing
raises passes with zero segfaults. So on that path a segfault means something really went wrong, and
treating it as teardown noise would hide a real crash. And
`og.log.info()` is invisible (`simulator.py:294` pins the root logger to WARNING), which is why the
repair logs at warning level.

## Coverage

Task choice is not cosmetic — it changes which code path runs. `put_green_block_into_bowl` (0) and
`stack_cubes` (6) have a `PrimitiveObject` main object; 1,2,3,4,5,7 have a `DatasetObject`; 8 and 9
do not load at all (below). VB-MOBJ *rescales* a PrimitiveObject but *removes and re-adds* a
DatasetObject, so **a task-0 pass says nothing about its add/remove path** — it needs task 4.

## Open

- **Upstream `_pre_remove_object` identity match** (see 2). Eliminates the class rather than
  repairing the damage.
- ~~**`open_drawer` / `close_drawer` do not build**~~ (`preset_name` `TypeError`) — CLOSED
  2026-08-14 by OG-lite `59af7c0`; both tasks load and pass `tests/test_vector_integrity.py`.
- ~~**`reset_joints()`'s batching is UNVERIFIED**~~ — CLOSED 2026-08-14, measured on a real
  cabinet; see `environments/joint_reset.py`.
- ~~**Scene 0's drawers never reached the commanded openness**~~ — CLOSED 2026-08-14, and worth
  reading, because the symptom pointed at three innocent places. `reset_joints()` commanded all five
  cabinet joints to a normalized -1.0 and in SCENE 0 ONLY the target joint settled at 0.17-0.19 m of
  a 0.30 m range, with `joint_02`/`joint_03` stopping dead at 0.2289/0.2288 m in every run —
  `init_openness_fraction` 0.62 where `open_drawer` is scored against 0. Scene 1 was perfect. It was
  NOT the joint-reset batching (reproduced at `--num_envs 1` with `run_joint_resets` bypassed), NOT
  `scene_base.py` re-applying object poses only for `idx != 0` (that path touches the scene FILE's
  objects; the cabinet is a task object added later by `og.Environment._load_objects`), and NOT the
  44 mm root-link z difference between the two scenes — that was a *consequence*. Scene 0's cabinet
  was placed **lying on its back**: root-link orientation ~identity instead of the config's
  `[0.7044, 0.0616, 0.0616, 0.7044]`, so its drawers slid vertically and jammed against
  `floors_jkaqil_0` and `breakfast_table_support`. `cabinet.usd` is authored `upAxis=Y` on a
  `upAxis=Z` stage, and Kit's metrics assembler compensates by appending
  `xformOp:rotateX:unitsResolve = 90` to the referencing prim's `xformOpOrder` — an op no
  OmniGibson pose setter writes and `XFormPrim._set_xform_properties` does not strip (it lists only
  the unsuffixed `rotate*`/`transform` ops), so it silently post-multiplies every pose set on that
  prim: `set_position_orientation(orientation=Q)` lands at `Q · Rx(90)`. It was applied to the FIRST
  reference to the asset only — the assembler's UnitsAdjust layer is content-hash keyed — which is
  the whole reason one member differed from another. Fixed in OG-lite
  `USDObject._preapply_articulation_root` by making the exported asset layer's up axis agree with
  the stage's, leaving the assembler nothing to insert in any scene.
  Temporary probes measured the stopped-drawer outcome, traced the pose phase, and identified the
  extra transform op. These one-off probes were removed during release cleanup.
- **Every vector reset does N global steps and 3N renders before any perturbation**, from
  `og.Environment.reset(get_obs=True)` per member. Measured harmless (<4e-5 m drift), but O(N).
  V-VIEW used to *double* that: it ended with a second per-member `og.Environment.reset()`, left
  over from when `og.sim.stop()` clobbered the scene, so a V-VIEW reset issued 2N global steps
  (measured 4 at Vec=2 against Default's 2, 8 at Vec=4 against 4). Removed 2026-08-14; the step
  count now equals Default's, and the camera spreads, per-member object poses and check-4 verdict
  are unchanged to every printed digit.
- **SB-NOUN degenerates** ~1/5 of resets by re-drawing the original PrimitiveObject (category
  literally `"object"`) → *"put the object in the bowl"*. Changing which objects it may draw alters
  what the perturbation means, so it is a human decision.
