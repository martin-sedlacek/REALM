# Perturbations in a vector env

State as of 2026-08-13. Companion to `README.md` (which covers how the vector env is *built*) and
`SCALING.md` (throughput). This file covers what had to change for REALM's perturbations to be
correct when N scenes share one simulator, and what is still open.

Verify anything here with `scripts/clara/interactive/t9_vbpose_nostopplay.py`; `t9_sweep.sh` fans
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
2. repair the sim's object-init queue                    (see "eviction" below)
3. ONE og.sim.stop(), only if a member's perturbation needs it
4. every member's perturbations run
5. ONE og.sim.play()
6. work the perturbations deferred because it needs a playing sim
7. ONE settle loop driving all members together, if any asked for it
```

Perturbations never call the global operations directly. They route through
`perturbations/_helpers.py` — `sim_stop`, `sim_play`, `sim_step`, `after_play`, `settle` — which
no-op or defer when `env.in_vec_env`. **Single-env behaviour is unchanged**, which is what lets the
historical numbers stay comparable.

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
alone** (OG-lite `simulator.py:1090-1093`). Names are unique per *scene*, not per simulator, and
every member is built from the same task YAML — so member 1's `remove_object("corkscrew")` evicts
**member 0's** freshly added corkscrew. It stays on the stage and in the registry, but nothing ever
initialises it.

Fingerprint: the repair names exactly the YAML-**named** distractors and none of the per-member
sampled `distractor_<i>` ones, whose names do not collide.

Worked around in `RealmVectorEnvironment._requeue_evicted_objects()`.
**The proper fix is upstream and is NOT applied**: `_pre_remove_object` should match on identity
(equivalently `(scene, name)`), which is strictly narrower and always correct, since
`scene.add_object` already forbids two live same-named objects in one scene.

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

## Testing: a pass is not evidence unless something asserts the effect

`t9_vbpose_nostopplay.py` checks: the sim is cycled exactly the expected number of times (0 for
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

Two harness gotchas: the script **exits 139 (SIGSEGV) whether it passes or fails** — Isaac segfaults
at teardown after the verdict prints, so grep `^PASSED`/`^FAILED` and never gate on exit code. And
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
- **`open_drawer` / `close_drawer` do not build**: `cabinet.usd` →
  `TypeError: missing a required argument: 'preset_name'` in `omnigibson/prims/material_prim.py`.
  2 of 10 tasks, and with them the drawer branches of SB-NOUN and S-LANG — those are the only tasks
  with a `synonyms:` block, so S-LANG's synonym path has never executed.
- **`reset_joints()` issues ~55 global `og.sim.step()`s per member per reset on drawer tasks** — the
  same class as the bug removed here, unreachable today only because those tasks do not load. The
  harness's step counter will catch it the day the asset is fixed.
- **Every vector reset does N global steps and 3N renders before any perturbation**, from
  `og.Environment.reset(get_obs=True)` per member. Measured harmless (<4e-5 m drift), but O(N).
- **SB-NOUN degenerates** ~1/5 of resets by re-drawing the original PrimitiveObject (category
  literally `"object"`) → *"put the object in the bowl"*. Changing which objects it may draw alters
  what the perturbation means, so it is a human decision.
