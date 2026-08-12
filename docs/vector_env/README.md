# Vectorized REALM environments -- status and open bug

Written 2026-08-12 as a handoff: the machine this was developed on is being retired mid-investigation.
The feature loads and runs; **one bug is open and reproducible**, documented in full below along with
the exact next diagnostic to run.

## TL;DR

`RealmVectorEnvironment` loads N REALM environments into one simulator and steps them with a single
`og.sim.step()`. 4 environments load, tile correctly, render distinct observations and step cleanly
(exit 0, no errors).

**The bug: `apply_scene_fixes_from_cfg()` appears to take effect only in scene 0.** In scenes 1..N-1
the breakfast table is not pinned and the chair that should be deleted is still present, so the task
objects end up on the floor. See `frames/montage_external.png` and `frames/montage_wrist.png`.

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

```bash
docker exec realm_stock bash -lc 'cd /app && conda run --no-capture-output -n behavior \
  python -u examples/03_vector_first_frames.py --num_envs 4 --task_id 0'
```

Writes `env<i>_external.png`, `env<i>_wrist.png` and 2x2 montages to `/app/logs/vector_first_frames`.
Takes ~9 minutes: ~80 s Isaac boot, then ~60-90 s per scene, then warmup.

**Never run `conda run` without `--no-capture-output`** -- it buffers all output until exit, and if the
process is killed the entire log is lost.

Peak GPU for 4 scenes was ~26 GB of 32 GB with a 16.6 GB policy server also resident, so 4 is close
to the ceiling on a 32 GB card while a policy server is up.

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

## The open bug

`frames/montage_external.png` (2x2, env0 top-left, env1 top-right, env2 bottom-left, env3 bottom-right):

- **env0** -- breakfast table present, task objects on it, chair removed. Correct.
- **env1, env2, env3** -- no table. Bare floor and rug. A chair that env0 does not have is present.

`frames/montage_wrist.png` confirms the robot and objects exist in all four: in env1-3 the gripper
hovers over the **rug**, with bowl, basket, marker and cube lying on the floor beneath it.

So the scene loads fine in every tile; what differs is `apply_scene_fixes_from_cfg()`. For this task
(`Pomaria_1_int` / `Table`) `realm/config/scenes/scenes.yaml` says:

```yaml
to_remove: ['straight_chair_pmpwwi_0']
to_fix:    ['breakfast_table_uhrsex_0']
```

Both entries are **hardcoded object names carrying a trailing instance index**. The observed symptom
is exactly what happens if neither entry matches in scenes 1..3: the table is never given its
`rootJoint`, so it is dynamic and gets displaced when the robot spawns inside it, and the chair is
never deleted. The task objects, which are placed at a height assuming a table surface at z=0.85,
then fall to the floor.

### Hypotheses, most likely first

1. **Object names are not identical across scene copies.** If OmniGibson derives the instance suffix
   from a counter that spans the simulator rather than the scene, scene 1's table is
   `breakfast_table_uhrsex_1`, which matches neither `to_fix` nor `to_remove`.
   *Check:* print `[o.name for o in env.omnigibson_env.scene.objects if "breakfast_table" in o.name]`
   per member. Note `scene_base.py:680` only asserts names are unique **within** a scene, so identical
   names across scenes are legal -- this hypothesis is about how the name is *derived*, not enforced.
2. **The batched stop/play changed the fixes' effect.** Single-env does stop -> fix -> play per env;
   the vector path does stop -> fix(env0) -> fix(env1) ... -> play. If `create_joint` or
   `remove_object` depends on state refreshed by `play()`, only the first member would take effect.
   *Check:* run the vector env with `num_envs=1`. If the table is correct there, the batching is
   implicated and hypothesis 1 is not the whole story.
3. **`remove_object` while iterating `scene.objects`.** `apply_scene_fixes_from_cfg` mutates the
   collection it is iterating, which can skip entries. This is a real latent bug regardless, but it
   does not obviously explain a clean scene-0-only split.

Note hypotheses 1 and 2 make **different** predictions for `num_envs=1`, so that single cheap run
discriminates between them. Run it first.

### Next diagnostic

```python
for i, env in enumerate(vec_env.envs):
    scene = env.omnigibson_env.scene
    tables = [o for o in scene.objects if "breakfast_table" in o.name]
    chairs = [o for o in scene.objects if "straight_chair" in o.name]
    print(i, scene.prim_path, "n_objects:", len(list(scene.objects)))
    for o in tables + chairs:
        print("   ", o.name, "fixed_base:", o.fixed_base, "pos:", o.get_position_orientation()[0])
```

Print this **twice** -- once right after `bind_scene_handles()` and once after `finalize_setup()` --
so it is clear whether the table is missing from the start or is displaced later by physics.

### Likely fix, once confirmed

If hypothesis 1 holds, match objects by category+model rather than by full name, i.e. compare
`re.sub(r"_\d+$", "", obj.name)` against the config entries with the same suffix stripped. Beware: a
scene may legitimately contain two instances of the same model, in which case stripping matches both.
Safer is to resolve the intended object per scene by its scene-local index.

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
- **`evaluate()` is still single-env.** Running a vectorized rollout also needs the inference client
  to batch N observations per step; nothing has been done there.
- All members currently run the same task config, differing only in sampled object placement.
  Per-member perturbations or tasks would need `VectorEnvironment`-style construction from a list of
  configs rather than one.

## Environment notes for resuming elsewhere

- Containers: `realm_stock` (image's own OmniGibson, `realm:og391`) and `realm_oglite` (the OG-lite
  fork bind-mounted at `/behavior-src/OmniGibson`). This work was done in **`realm_stock`**.
- Verify which OmniGibson is live before trusting anything: compare
  `md5sum /behavior-src/OmniGibson/omnigibson/utils/usd_utils.py` in the container against the host
  checkout. `realm_stock` has no bind mount there; a matching md5 means you are in `realm_oglite`.
- The repo is bind-mounted at `/app`, so images written to `/app/logs/...` appear on the host under
  `logs/...`. `logs/` is gitignored, which is why the frames here were copied into `docs/vector_env/`.
