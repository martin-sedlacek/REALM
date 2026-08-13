# Vectorized REALM environments -- status and the scene-content bug

Written 2026-08-12 as a handoff. **Substantially revised 2026-08-13** (Clara, L40S job 190155) after
the bug was measured rather than inferred: the original diagnosis below was wrong, and the real cause
is upstream in OmniGibson 3.9.1. Root cause and fix are in
[The real bug](#the-real-bug-scene-file-objects-in-scenes-idx--0-load-100-m-too-high).

## TL;DR

`RealmVectorEnvironment` loads N REALM environments into one simulator and steps them with a single
`og.sim.step()`. 4 environments load, tile correctly, render distinct observations and step cleanly
(exit 0, no errors).

**The bug: every scene-file object in scenes `idx != 0` is loaded exactly 100 m too high** --
`INITIAL_SCENE_PRIM_Z_OFFSET` -- by stock OmniGibson 3.9.1. REALM then pins the breakfast table with
a `FixedJoint` at that lifted pose, so the table is the one thing no later reset can bring back down.
The task objects have nothing to rest on and fall to the floor. See `frames/montage_external.png`
and `frames/montage_wrist.png`.

> **Retracted:** this document previously said `apply_scene_fixes_from_cfg()` takes effect only in
> scene 0, and ranked "object names carry a globally-numbered instance suffix" as the leading
> hypothesis. **Both are false and were disproven by measurement** -- see
> [What the scene fixes actually do](#what-the-scene-fixes-actually-do-they-work). The visible
> symptom was real; the attributed cause was not.

## What was added

| file | what |
| --- | --- |
| `realm/environments/env_vector.py` | `RealmVectorEnvironment` -- new |
| `realm/environments/env_dynamic.py` | `in_vec_env` flag; `__init__` split into load + `post_play_setup()`; `bind_scene_handles()` / `finalize_setup()` halves; `pre_step()` / `post_step()`; `warmup_ee_cmd()` / `warmup_action()`; `apply_scene_fixes_from_cfg(manage_sim_state=...)` |
| `examples/03_vector_first_frames.py` | smoke test: load N envs, warm up, step once, save each member's first frame |
| `realm/sim_config.py` | `REALM_INCREMENTAL_CONTACT_CACHE` / `REALM_PROXIMITY_GATE` env-var knobs for the OG-lite macros (unrelated to this bug) |

### Why construction is three-phase

`og.sim.play()` and `og.sim.stop()` are **global** -- they act on every scene at once, so no member can
cycle them alone. Construction therefore goes:

1. build every member with `in_vec_env=True` (loads scenes, does not play)
2. one `og.sim.play()`, then `post_play_load()` + `bind_scene_handles()` per member
3. one stop/play cycle wrapped around `apply_scene_fixes_from_cfg(manage_sim_state=False)` for all
   members, then `finalize_setup()` per member

The single-env path is unchanged: `__init__` calls `post_play_setup()`, which runs the same three
pieces back to back in the same order.

### Why stepping cannot go through `Environment.step()`

`og.sim.step()` advances **every** scene. Calling a member's own `step()` would advance all N scenes
while applying only that member's action. `RealmVectorEnvironment.step(actions)` applies every
member's action first (`_pre_step`), steps once, then collects observations (`_post_step`) -- the same
shape as OmniGibson's own `VectorEnvironment`.

## Reproduce

On Clara (Apptainer, no Docker), from a held L40S allocation -- `scripts/clara/interactive/rr` supplies the
binds and picks stock vs OG-lite:

```bash
MODE=stock ./scripts/clara/interactive/rr \
  python -u examples/03_vector_first_frames.py --num_envs 4 --task_id 0 \
    --out_dir /logs/vector_first_frames
```

Reproduced verbatim on 2026-08-13 (job 190155): the new montage is indistinguishable from the
committed `frames/montage_external.png`, so the bug is live rather than a stale artifact.

Writes `env<i>_external.png`, `env<i>_wrist.png` and 2x2 montages. Takes ~11 min: ~90 s Isaac boot,
then ~2 min per scene, then warmup.

**Never wrap the in-container command in `bash -lc`.** Apptainer binds `$HOME`, so a *login* shell
re-sources the host `~/.bashrc`, prepends `~/miniconda3/bin` to PATH and shadows the container's
conda env -- you get host Python and `ModuleNotFoundError: No module named 'omnigibson'`. Use
`bash -c`, or call `python` directly. (The original Docker workflow used
`conda run --no-capture-output`; the `--no-capture-output` trap does not apply here because `rr`
never invokes `conda run`.)

Peak GPU for 4 scenes was ~26 GB of 32 GB on the old workstation with a 16.6 GB policy server also
resident. On a 46 GB L40S with no policy server, 4 scenes are comfortable.

## What is verified working

- 4 scenes load and tile side by side (`/World/scene_0` .. `/World/scene_3`), no overlap
- one `og.sim.step()` advances all members; `pre_step`/`post_step` split works
- every member renders its **own** cameras: all four external frames differ pairwise (asserted in the
  script, so "all four rendered the same tile" cannot pass silently)
- the robot exists in every scene and its wrist camera is correctly mounted -- visible in all four
  wrist frames
- the task objects exist in every scene (bowl, basket, marker, green cube, bottle all visible)
- external camera extrinsics land inside each member's own tile without knowing the tile offset,
  because sensors are loaded into their own env's scene and REALM already sets `pose_frame: "parent"`
- per-member object placement differs (placement is sampled per member while building its config)
- run completes with exit 0, no traceback, no segfault

## What the scene fixes actually do: they work

Measured 2026-08-13 with `scripts/clara/interactive/t1_scene_probe.py`, which wraps the real
`apply_scene_fixes_from_cfg` and dumps the scene immediately either side of it, per member. At
`num_envs=4`, task 0, stock container:

| | scene_0 | scene_1 | scene_2 | scene_3 |
| --- | --- | --- | --- | --- |
| object-name set before fixes (md5) | `b10042efd394` | `b10042efd394` | `b10042efd394` | `b10042efd394` |
| object-name set after fixes (md5) | `6f0c5e3b854f` | `6f0c5e3b854f` | `6f0c5e3b854f` | `6f0c5e3b854f` |
| `n_objects` before -> after | 128 -> 127 | 128 -> 127 | 128 -> 127 | 128 -> 127 |
| `breakfast_table_uhrsex_0.fixed_base` | False -> **True** | False -> **True** | False -> **True** | False -> **True** |
| `rootJoint` prim created on the stage | yes | yes | yes | yes |
| `straight_chair_pmpwwi_0` after fixes | `active=False` | `active=False` | `active=False` | `active=False` |

`apply_scene_fixes_from_cfg` is called exactly once per member, takes the config branch in every
member, and does identical work in every member.

**Object names are identical across scene copies**, so hypothesis 1 is dead:
`create_object_from_init_info` passes `name` straight through from the scene JSON, and
`scene_base.py:678` only asserts uniqueness *within* a scene. The `_0` suffix is part of the
authored asset name, not a runtime counter.

Note the chair check has to be `IsActive()`, not `IsValid()`: `scene.remove_object()` ends in
`delete_or_deactivate_prim()`, which may **deactivate** rather than delete. A deactivated prim still
satisfies `IsValid()` and would mislead the probe into reporting a removal that had not happened.

## The real bug: scene-file objects in scenes `idx != 0` load 100 m too high

Same run, world positions of the same asset:

```
scene_0  breakfast_table_uhrsex_0  pos=[ -0.4119, -1.9556,   0.6193]
scene_1  breakfast_table_uhrsex_0  pos=[ 24.8353, -1.9556, 100.6193]
scene_2  breakfast_table_uhrsex_0  pos=[ 50.0824, -1.9556, 100.6193]
scene_3  breakfast_table_uhrsex_0  pos=[ 75.3297, -1.9556, 100.6193]
```

The x offsets are the intended tiling (`SCENE_MARGIN = 10.0`). The **+100 in z is not intended** --
it is exactly `INITIAL_SCENE_PRIM_Z_OFFSET = -100.0`, negated. At construction end, **70 of 127
registered objects in each of scenes 1..3 sit above z = 50**, and none in scene 0.

### Mechanism (stock OmniGibson 3.9.1, `scenes/scene_base.py:_load_scene_prim_with_objects`)

1. The scene prim is parked at `initial_scene_prim_z_offset` (**-100**) for `idx != 0`, to avoid
   collisions while loading.
2. Each object's pose is then set with
   `obj.set_position_orientation(position=..., orientation=...)` -- **no `frame=` argument, and
   `XformPrim.set_position_orientation` defaults to `frame="world"`**. The setter converts the world
   target into the parent's frame using the parent's *current* transform, so the parked -100 is
   baked in: `local_z = intended_z + 100`.
3. The prim is then moved to `[last_scene_edge + margin + left_edge_to_center, 0, 0]`, z = 0, which
   carries every object up with it.

The prebuilt scene *structure* (walls, floor, rug) keeps its own authored local transform and lands
correctly at z ~ 0. That is why the tiles look like a room whose furniture has been deleted, rather
than like a scene that is uniformly displaced.

**This is not an OG-lite regression.** `md5sum` of `scene_base.py` is identical in the image and in
the OG-lite checkout (`3ea4bb3fd294236181b1a95609d7a520`).

There *is* an intended correction -- `Scene.initialize()` transforms every object pose by
`self.pose` and `load_state`s it, commented "In VectorEnvironment, the scene pose loaded from the
file should be updated" -- but it does not survive to construction end. `Simulator.import_scene`
wraps it in a **global** `play()` / `initialize()` / `step()` / `stop()` per scene, and a vector env
imports N scenes, so every import cycles play/stop over all previously imported scenes.

### Why it presented as "the scene fixes only apply to scene 0"

The fixes run while the table is at +100 and give it a `FixedJoint` **at that pose**. So the single
object REALM pins is the single object that cannot be recovered afterwards. Everything else can come
back down; the table cannot. After `warmup()` (which resets each member):

| | scene_0 | scene_1 | scene_2 | scene_3 |
| --- | --- | --- | --- | --- |
| objects above z = 50 | 0/128 | **70/128** | **70/128** | **1/128** |
| the one still lifted in scene_3 | -- | -- | -- | `breakfast_table_uhrsex_0` |
| `breakfast_table_uhrsex_0` z | 0.6193 | 100.6193 | 100.6193 | 100.6193 |
| `cube` z (task object) | 0.820 | **0.015** | **0.015** | **0.015** |
| `bowl` z (task object) | 0.840 | **0.035** | **0.035** | **0.049** |

This is the whole symptom, quantified: the table is 100 m up in every tile but scene 0, and the task
objects consequently rest on the floor at z ~ 0.02 instead of on a table top at z ~ 0.84.

It also explains the montage asymmetry that the earlier writeup read as "env1-3 are alike":
**scene_3 recovers almost fully on reset (only the pinned table stays up) while scenes 1 and 2 do
not**, which is why env3 shows a full kitchen and env1/env2 show a bare room. Scene 3 is the last
member reset; why the earlier members do not keep their restored poses is not yet established.

Two more things the earlier writeup got wrong, both now measured: the chair removal **succeeds in
every scene**, and "a chair env0 does not have" is `straight_chair_pmpwwi_1`, which is never in
`to_remove` and is simply occluded by the table in env0. All four members report identical robot
qpos (`q0=[0.0056, -0.4623, -0.1084]`), so nothing about the differences is robot or camera state.

### Second, independent bug: reset re-adds the removed chair

`n_objects` goes 127 -> **128** after warmup **in every scene, including scene 0**, and
`straight_chair_pmpwwi_0` returns to `active=True`. `Scene.reset(hard=True)` calls
`restore(self._initial_file)`, and `_initial_file` was captured at the end of `Scene.initialize()` --
before `apply_scene_fixes_from_cfg` ever ran. So the first reset undoes the removal.

Since scene 0 is affected too, this is not a vector-env bug at all. **Confirmed on the single-env
production path** with `scripts/clara/interactive/t3_single_env_chair.py`: removal is correct after
construction, then 2 of 2 resets bring the chair back. REALM calls `reset()` once per repeat, so
every repeat after the first runs with an object the task config asked to delete -- 24 of 25 at the
usual `--repeats 25`.

**It is a port regression: not an issue on OmniGibson 1.1.1** (per Martin, 2026-08-13). Results
collected on the old stack are unaffected; og391 results are. Candidate fix:
`scene.update_initial_file()` after applying the scene fixes. Not yet implemented.

## The fix

Re-apply the object poses once the scene prim is at its final position, in
`_load_scene_prim_with_objects` (applied in the OG-lite fork, `omnigibson/scenes/scene_base.py`):

```python
self._scene_prim.set_position_orientation(position=scene_position, orientation=identity_quat)
new_scene_edge = last_scene_edge + scene_margin + (aabb_max[0] - aabb_min[0])

for obj_name, obj in self._init_objs.items():
    obj.set_position_orientation(
        position=th.as_tensor(self._init_state[obj_name]["root_link"]["pos"], dtype=th.float32)
        + scene_position,
        orientation=self._init_state[obj_name]["root_link"]["ori"],
    )
```

`idx == 0` is untouched -- its prim is never parked and `scene_position` is zero -- so single-scene
behaviour is bit-identical, which matters because that is the path every production REALM eval uses.

`frame="scene"` would be the tidier spelling but is **not usable here**: it routes through
`Scene.convert_scene_relative_pose_to_world`, which reads `_pose_info`, and `_pose_info` is only
assigned by `Scene.load()` after this method returns.

Because the fix lives in OG-lite, vector envs must now run with the OG-lite bind
(`MODE=oglite` in `scripts/clara/interactive/rr`); the stock image still has the upstream behaviour.

### Verified

OG-lite `ef7442b`, `num_envs=4`, task 0, with warmup -- `frames_fixed/montage_external.png`:

| | scene_0 | scene_1 | scene_2 | scene_3 |
| --- | --- | --- | --- | --- |
| objects above z = 50, post-warmup | 0/128 | **0/128** | **0/128** | **0/128** |
| `breakfast_table_uhrsex_0` z | 0.6193 | **0.6193** | **0.6193** | **0.6193** |
| table x (tiling preserved) | -0.4119 | 24.8353 | 50.0824 | 75.3297 |
| `cube` z | 0.8199 | **0.8199** | **0.8199** | **0.8199** |
| `bowl` z | 0.8395 | **0.8395** | **0.8395** | **0.8395** |
| object z range | -0.150 .. 2.550 | same | same | same |

Compare the pre-fix table in
[Why it presented as ...](#why-it-presented-as-the-scene-fixes-only-apply-to-scene-0): the task
objects were at z ~ 0.015 in scenes 1-3, i.e. on the floor. All four tiles now render the table
with every task object on it.

**Not fixed by this, and still open:** the reset still re-adds `straight_chair_pmpwwi_0`
(`active=True`, `n_objects` 128) in every scene -- that is the separate `_initial_file` bug above,
and it is visible in the fixed montage as the black chair with the checkered seat.

## Other known gaps in vectorization (not yet investigated)

- **Perturbations that cycle the simulator.** `v_view` calls `og.sim.stop()` / `og.sim.play()` and
  `reset()` calls perturbations per member -- in a vector env that would disturb every other member.
  Only `Default` has been exercised. Anything beyond it needs the same batching treatment as the
  scene fixes.
- **`reset_joints()` steps the sim** 40 times for drawer tasks (`open_drawer` / `close_drawer`), from
  inside `RealmEnvironmentBase.__init__`. In a vector env that advances all scenes. Only the
  non-drawer path has been run.
- **EE control and world frame.** `_robot2world` uses the member's scene-local `robot_pos`. Whether
  the EE controller interprets an absolute-pose command in world or scene coordinates has not been
  checked, and the tiles are offset from each other. Joint control is unaffected. The first-frame
  test barely moves the arm, so it would not have caught this.
- ~~**`evaluate()` is still single-env.**~~ **Done 2026-08-13** -- `realm/vector_eval.py` +
  `examples/04_vector_evaluate.py`. It runs `repeats` rollouts in waves of `num_envs` and writes the
  same four artifacts as the single-env path. Inference is **sequential on purpose** (one policy call
  per member per chunk boundary, never batched): batching is a separate change and would hide desync
  bugs behind a fixed-shape batch. Members desync by construction -- a finished member is marked
  inactive, finalised immediately, and fed its last action as a hold command while the others run on.
- All members currently run the same task config, differing only in sampled object placement.
  Per-member perturbations or tasks would need `VectorEnvironment`-style construction from a list of
  configs rather than one.

## Sustained stepping

The first-frame smoke test proves construction and tiling; it takes one step, so it says nothing
about stability. `scripts/clara/interactive/t5_vec_sustained.py` drives the vector env for a
rollout's worth of steps and checks, every 50 steps, that no member has gone non-finite, that members
stay pairwise **distinct** (a shared-state bug would collapse them onto each other), that the task
objects stay on the table -- the regression guard for the 100 m z-offset -- and that per-step time is
not drifting.

4 members x 200 shared steps, `MODE=oglite`:

```
step   50   153.6 ms/step   m0..m3: mo_z=0.820 to_z=0.839
step  200   153.6 ms/step   m0..m3: mo_z=0.820 to_z=0.839
ms/step first quarter: 153.6      last quarter: 153.6      checks failed: 0
```

Flat to within 0.1% across the run, and **153.6 ms for 4 members is 38 ms per member-step** against
~90-130 ms for a single env.

## Environment notes

Originally two Docker containers, `realm_stock` and `realm_oglite`. On Clara there is no Docker, so
the same two conditions are one image plus a bind, selected by `MODE` in `scripts/clara/interactive/rr`:

| MODE | OmniGibson that is live |
| --- | --- |
| `stock` | the image's own 3.9.1 at `/behavior-src/OmniGibson` |
| `oglite` | host `OG-lite_og391/omnigibson` bound over the image's package |

Binding only the `omnigibson/` package rather than the whole repo directory leaves the image's
editable-install metadata intact; `__editable__.omnigibson-3.9.1.pth` resolves through the bound
path either way.

- **Vector envs now need `MODE=oglite`** -- the z-offset fix lives in the fork. The stock image
  still has the upstream behaviour, which is useful for reproducing the bug.
- Verify which source is live before trusting a comparison. `getattr(gm, ...)` is **not** a valid
  check: in the stock image `gm` is a `MacroDict` that returns a truthy `{'_read': set()}` for
  undefined macros rather than raising. Check the source instead:
  `python -c "import inspect, omnigibson.utils.usd_utils as uu; print('PROXIMITY_GATE' in inspect.getsource(uu))"`.
- The repo is bind-mounted at `/app` and `REALM/logs` at `/logs`, so write artifacts to `/logs/...`.
  `logs/` is gitignored, which is why the frames here were copied into `docs/vector_env/`.

## A real vectorized evaluation: 25 rollouts, pi0.5

Run 2026-08-13 (Clara, allocation 190155). Task 0 `put_green_block_into_bowl`, `Default`
perturbation, robot `DROID_robolab`, pi0.5 (`pi05_droid_jointpos`), `num_envs=4`, `repeats=25`,
`max_steps=500`, horizon 8, render-on-demand ON, `MODE=oglite`.

```bash
ALLOC=<jobid> NUM_ENVS=4 REPEATS=25 MAX_STEPS=500 ROD=1 ROBOT=DROID_robolab \
  RUN_ID=vec25_robolab EXPERIMENT=vec_pi05 \
  ./scripts/clara/interactive/go vec_eval_full ./scripts/clara/interactive/t6_vec_eval.sh
```

### Result -- RETRACTED 2026-08-13, the SR is invalid

> **Do not use the numbers in this section.** Every member of a wave shared ONE
> `task_progression` dict, because `env_base.py:48` assigned the module-level
> `TASK_PROGRESS_RUBRICS[task_type]` without copying it and
> `recompute_task_progression` mutates it in place. Progression therefore became an **OR across
> members** and stuck there (line 222 short-circuits on `is_completed_flag or checker(obs)`, so a
> stage already flagged by another member is never re-checked). One member grasping marked `GRASP`
> done for all four.
>
> Spotted by Martin from the videos: rollouts scored `SUCCESS` with the block never grasped. The
> giveaway was in the report all along -- members of a wave share identical
> `task_progression_timestamps`.
>
> So **SR 0.960 is an upper bound of the form "at least one member of the wave succeeded"**, not a
> per-rollout success rate. The true value is unknown and likely much lower.
>
> Fixed by deep-copying the rubric per environment. **The eval must be re-run**; SR cannot be
> recovered from the stored artifacts, because `PLACE_INTO` needs object poses and only qpos and
> actions were saved. The per-member trajectories (`qpos/`, `actions/`, `videos/`) ARE valid --
> only the progression, stage and SR columns are contaminated.
>
> Single-env is unaffected: one environment per process means one reference, and `reset()` clears
> it. The single-env baseline of SR 1.000 at n=10 still stands.

| | |
| --- | --: |
| ~~**SR**~~ | ~~0.960 (24/25)~~ **invalid** |
| **task_progression** | **0.984** (min 0.60) |
| stages | 24 SUCCESS, 1 MOVE_CLOSE |
| collisions_self | 0.00 (all runs) |
| collisions_env | 8.88 mean, range 1-23 |
| object_drops | 0.32 mean, range 0-3 |
| joint_path_length | 10.76 mean |
| cart_path_length | 1.73 mean |
| wall | 1586.8 s total, of which 640.5 s building the 4 envs |

**This matches the single-env baseline for the same task and robot: SR 1.000 at n=10.** 24/25 against
10/10 is well within binomial noise. Stock `DROID` would have been the wrong config to verify
against -- it scores ~0.200 here, where a vectorized result of 0.16 or 0.24 could not be told from
either noise or breakage, and where hardly any rollout would reach the success path.

`run_id` 0..24 are all present exactly once, so no wave dropped or duplicated a rollout.

### The success gate is real, not a proxy

`put` runs REACH -> GRASP -> LIFT_SLIGHT -> MOVE_CLOSE -> **PLACE_INTO**, and `PLACE_INTO` is
`(OnTop(block, bowl) or Inside(block, bowl)) and not is_grasping` -- an OmniGibson object-state query
plus release. `task_progression == 1.0` therefore means the block really was placed and let go.

The single failure is coherent: **run 24**, TP 0.60 = 3 of 5 stages, stalled at `MOVE_CLOSE`, with
**3 object_drops** -- it grasped the block and lost it three times, then ran out the full 500 steps.
`eval/run024_failure_sheet.png` shows exactly that: the arm reaches the block over and over, and the
block never leaves the table. It is a policy failure, not an infrastructure one.

### Desync worked

Members finish at different steps and the wave keeps running:

| wave | steps | wall | note |
| --- | --: | --: | --- |
| 1 | 416 | 100.8 s | members finished at 414, 414, 415, 415 |
| 2 | 209 | 47.6 s | |
| 3 | 334 | 78.3 s | |
| 4 | 354 | 81.8 s | |
| 5 | 299 | 69.7 s | members finished at 298, 299, 299, 299 |
| 6 | 319 | 72.4 s | |
| 7 | 500 | 81.3 s | **1 of 4 members recorded** (25 = 6x4 + 1); the other three still step |

Wave 7 is the useful one: with a single active member it runs 500 steps in 81.3 s (163 ms/step)
against ~230 ms/step when four members are active, which is the sequential inference cost showing up
exactly where it should and confirms the inactive-member hold path is cheap.

### Throughput

2431 shared steps in 531.9 s of stepping = **219 ms per shared step, i.e. ~55 ms per member-step**,
against ~90-130 ms/step for a single env. Startup is paid once for all 25 rollouts either way.

**Caveat:** rollout lengths cluster tightly *within* a wave (e.g. wave 3: all four at 334 steps) and
vary a lot *between* waves (209 to 500). Members do differ -- `collisions_env` ranges 2 to 23 within
one wave -- so they are not running identical trajectories; they are reset together and pi0.5 paces
them similarly. Worth remembering if you ever treat per-wave results as independent samples.

## How many environments fit on one L40S

Measured 2026-08-13 on a 46068 MiB L40S with the pi0.5 policy server resident (11839 MiB), which is
the operating configuration -- REALM shares one card between the sim and the policy.
`scripts/clara/interactive/t7_env_capacity.py` builds members one at a time and reports GPU memory
after each, so one run gives the per-scene cost instead of one run per candidate `num_envs`.

| scenes loaded | GPU used | free | increment |
| --: | --: | --: | --: |
| 0 (server + Isaac boot) | 11839 | 34229 | -- |
| 1 | 17043 | 29025 | **+5204** |
| 2 | 19799 | 26269 | +2756 |
| 3 | 22499 | 23569 | +2700 |

**The first scene costs 5204 MiB and every one after it ~2728 MiB.** The gap is one-time renderer and
Isaac allocation, not scene data.

Projected **load** ceiling from that marginal figure, keeping 3000 MiB free: ~10 scenes with the
server on the card, ~14 without. **That extrapolation turned out to be pessimistic** -- see below.

### Measured at 8 envs: it fits easily

Full `RealmVectorEnvironment`, built *and played*, `DROID_robolab`, 100 shared steps
(`t5_vec_sustained.py`):

| | |
| --- | --: |
| after building and playing 8 envs | **28533 / 46068 MiB used, 17535 MiB free** |
| after warmup and 100 steps | 28339 MiB, 17729 MiB free (flat) |
| checks failed | 0 |

8 scenes played cost 28533 - 11839 = **16694 MiB including the one-time overhead, ~2087 MiB per
scene on average** -- less than the 2728 MiB marginal the load probe measured over its first three
scenes. Isaac pools memory and later scenes reuse it, so **linear extrapolation from a handful of
scenes overestimates**. Do not trust the projection above; measure at the N you care about.

With 17.5 GB still free at 8 envs, memory is clearly not the wall.

### Throughput is the wall, and it bites before memory does

Same protocol both rows (rendering every step, no policy in the loop):

| members | ms per shared step | ms per **member**-step | member-steps/s |
| --: | --: | --: | --: |
| 4 | 153.6 | **38.4** | **26.0** |
| 8 | 452.0 | 56.5 | 17.7 |

**Doubling from 4 to 8 members makes aggregate throughput ~32% worse**, not better: per-member cost
rises 38.4 -> 56.5 ms and total member-steps/s falls 26.0 -> 17.7. Stability is not the problem --
the 8-env run was flat to 0.2% across 100 steps and passed every check. The simulation itself stops
scaling; note this test has **no inference at all**, so it is not the sequential-policy-call cost.

**So the useful operating point is at or below 4 envs on one L40S, and adding scenes past that is
counterproductive.** Memory would allow far more; the step loop will not.

For reference, the 25-rollout pi0.5 eval at 4 members with render-on-demand ran ~230 ms per shared
step (~58 ms per member-step), against 163 ms per step for a single active member in its final wave.
Sequential inference costs roughly +22 ms per step per extra member on top of the simulation term.

**Where to look next**, in order: find out what saturates between 4 and 8 members (GPU render
throughput for N x 3 cameras, CPU for N scenes' physics, or the contact matrices) -- that is what
would raise the ceiling. Batched inference only helps once the simulation scales.
