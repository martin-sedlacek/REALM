# Vector-env scaling: state of things

Written 2026-08-13 on Clara (L40S, 46068 MiB), allocation 190155. A snapshot, not a conclusion —
several numbers below are confounded and say so. Companion to
[README.md](README.md) (correctness, the z-offset bug, the 25-rollout eval) and
[../perf/og391_step_profile.md](../perf/og391_step_profile.md) (single-env profile).

All measurements are the **max-juice** configuration, which is the only one worth measuring because
it is what a production eval runs:

    MODE=oglite + gm.INCREMENTAL_CONTACT_CACHE + gm.PROXIMITY_GATE_ENABLED  (both default ON)
    + ENABLE_VISUAL_UPDATES=False + OBJECT_STATE_UPDATE_WHITELIST=["ToggledOn"]
    + render_on_demand + rendering_mode=rt + robot DROID_robolab

Earlier numbers taken with rendering every step, or with `rendering_mode=r`, are **discarded** —
see [Retractions](#retractions).

---

## Status

| question | state |
| --- | --- |
| does a vectorized eval **run** end to end | yes — 25 rollouts, all artifacts, no crash |
| are its **success metrics** trustworthy | **NO, not before 2026-08-13** — shared `task_progression` made SR an OR across members; fixed, needs re-running. See README |
| how many envs fit in VRAM | **not the limit** — 16 envs played use ~21 GB of 46 GB |
| how many envs can be built | **16 confirmed at `rt`** after raising the RTX descriptor pool |
| is 8 faster than 4 | **yes**, consistently, in both measured batches |
| is 16 faster than 8 | **unknown** — run in flight |
| is 16 *economic* | **probably not** — 66 min build, see [Build cost](#build-cost-is-the-real-problem) |

---

## The ceiling was a renderer descriptor pool, not memory

At `num_envs=16`, scene loading segfaulted while loading **scene 10**:

```
[gpu.foundation.plugin]   Unable to allocate descriptor sets.
[rtx.denoising.plugin]    Failed to allocate ParameterBlock resources.
[rtx.denoising.plugin]    Failed to initialize parameter factory for, NRD.PackForNRD.
[rtx.raytracing.plugin]   Failed to allocate ParameterBlock resources
Fatal Python error: Segmentation fault
```

GPU memory at the time: **~17 GB of 46 GB**. A Vulkan descriptor pool is a fixed-capacity arena
sized at creation, so allocation fails on *slot count* regardless of free device memory.

### Root cause, established from the binaries

`libgpu.foundation.plugin.so` is the only binary containing `"Unable to allocate descriptor sets"`.
Its `rtx::resourcemanager::Context` init reads two capacity settings and creates the Vulkan pool
`"ResourceManager Descriptor Pool"` from them:

| setting | default | what it sizes |
| --- | --: | --- |
| **`/rtx/descriptorSets`** | **10000** | max descriptor-set count passed to `createDescriptorPool` |
| `/rtx/reservedDescriptors` | 131072 | bindless descriptor range counts (device-clamped) |

Failure site is `ResourceManagerContext.cpp:7403` in `allocateDescriptorSets`, terminating in an
`int3` — the segfault. `"Failed to allocate ParameterBlock resources"` has **no setting of its own**;
it comes from a header inlined into every `librtx.*.plugin.so`, which is why denoising and raytracing
both report it. It is purely the downstream symptom.

**NVIDIA ships `360000` / `900000` itself** — in 71 `extension.toml` files in this install, 36x the
default — but only inside `[[test]]` blocks, so nothing applies them at normal launch and OmniGibson
runs on the bare 10000.

### The fix

Applied in OG-lite `omnigibson/simulator.py`, alongside where it already injects `--/log/level=error`:

```python
sys.argv.append("--/rtx/descriptorSets=360000")
sys.argv.append("--/rtx/reservedDescriptors=900000")
```

These are read **once during RTX ResourceManager init**, so they must be set before startup —
`carb.settings` and `app.set_setting()` (everything REALM currently uses, including
`set_rendering_mode`) are post-launch and have no effect on them.

Confirmed reaching Kit:
```
Passing the following args to the base kit application:
  [..., '--/rtx/descriptorSets=360000', '--/rtx/reservedDescriptors=900000']
```

**Result: all 16 scenes import at `rt`, zero descriptor failures.** `play()` and `post_play_load()`
also completed at 16 scenes. No scene-DB instance errors appeared, so the secondary ceiling
(`/rtx/sceneDb/reservedInstances`, `/rtx-transient/scenedb/instanceBudget`) has not been reached.

Render products per env are **2** with `multi_view=False` and the robolab camera filter applied
(1 external + wrist); 3 without the filter. 16 envs is therefore 32-48 products.

---

## Throughput

96 steps, ROD firing every 8th step, no policy in the loop (so this is sim scaling, not
policy-server cost).

| N | render products/env | node share | ms/shared step | ms/**member**-step | member-steps/s | GPU MiB |
| --: | --: | --- | --: | --: | --: | --: |
| 4 | 3 | 2/node | 104.7 | 26.17 | 38.2 | 10506 |
| 8 | 3 | 2/node | 185.5 | **23.18** | **43.1** | 13707 |
| 4 | 2 | 3/node | 136.1 | 34.03 | 29.4 | 10449 |
| 8 | 2 | 3/node | 189.3 | 23.67 | 42.3 | 13649 |
| 16 | 2 | 3/node | *in flight* | | | |

**8 beats 4 in both batches** — by +13% (3-product batch) and +44% (2-product batch). Scaling is
sublinear: 2x the members costs ~1.4-1.8x the step time.

**Do not read the 2-product rows as "the camera patch made it slower."** Those ran three Isaac
instances to a node against two for the 3-product rows, so the absolutes are inflated by contention.
Compare *within* a batch only. Note also that GPU use barely moved (10506 -> 10449, 13707 -> 13649),
which is weak evidence the removed render product was cheap in memory — that is unexplained and
worth a clean measurement.

### Phase split — where the time goes

| N | products | `vec.pre_step` | `og.sim.step` + flush | `vec.post_step` |
| --: | --: | --: | --: | --: |
| 4 | 3 | 0.5 (1%) | 84.9 (81%) | 19.3 (18%) |
| 8 | 3 | 0.9 (0%) | 147.1 (79%) | 37.5 (20%) |
| 4 | 2 | 0.6 (0%) | 113.7 (84%) | 21.8 (16%) |
| 8 | 2 | 0.9 (0%) | 149.1 (79%) | 39.4 (21%) |

`post_step` — the N observation readbacks — is **linear**, ~4.7-5.5 ms per member. `pre_step` is
free. Everything else is `og.sim.step`.

### Probe medians (ms)

| probe | N=4 (3p) | N=8 (3p) | N=4 (2p) | N=8 (2p) |
| --- | --: | --: | --: | --: |
| `_sim_context.step(render=True)` | 143.8 | **259.2** | 177.2 | **276.3** |
| `og.sim.render` (explicit flush) | 63.0 | 99.0 | 59.2 | 101.1 |
| `_sim_context.step(render=False)` | 8.1 | 15.0 | 12.4 | 14.5 |
| `og.sim._non_physics_step` | 0.53 | 0.71 | 0.62 | 0.84 |

**Rendering is ~90% of `og.sim.step`.** On a render step, `render=True` (259 ms) plus the flush
(99 ms) is ~358 ms against 15 ms for each of the 7 blind steps.

**The contact cache is spent as a lever.** `_non_physics_step` is **0.71 ms** at N=8 — it was 134 ms
on stock 3.9.1 and 42.7 ms with the proximity gate alone. The perf doc's "lever #1, contact cache is
~50% of stepping" is finished; it is now ~0.4% of a step.

---

## Build cost is the real problem

Per-scene import time, single clock, from the 16-scene run:

| scene | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
| --- | --: | --: | --: | --: | --: | --: | --: |
| seconds | 198 | 151 | 152 | 173 | 247 | 295 | **365** |

**Total 3989 s = 66 min just to import 16 scenes**, and scene 16 costs 2.4x scene 2. N=8 shows the
same shape (139 -> 184 s). So there is a ~150 s flat floor per scene plus a term that grows with the
number of scenes already on the stage.

Splitting one import by log markers (omnigibson clock only):

| phase | scene 2 | scene 14 |
| --- | --: | --: |
| objects added -> first `RigidContactAPI` touch | **138.9 s** | **294.2 s** |
| contact API -> finger-link inference | 5.7 s | 9.2 s |
| finger inference -> `Imported scene N` | 5.9 s | 19.7 s |

~92% sits in the first window, which spans `scene.load()`'s object instantiation **plus** the
`play()` inside `import_scene`, and it doubles. **Caveat:** these are boundaries between incidental
warning messages, not instrumented phases, so this does *not* separate USD instantiation from
`play()`. Only the `play()` half is fixable by batching, so that split matters and is not yet
measured.

### Two mechanisms

**1. Global play/stop per import.** `Simulator.import_scene`:

```python
self._last_scene_edge = scene.load(...)
self._scenes.append(scene)
assert self.is_stopped()
self.play()          # GLOBAL - acts on every scene already on the stage
scene.initialize()
self.step()          # GLOBAL
self.stop()          # GLOBAL
```

Importing scene 16 plays/steps/stops a stage holding 15 other scenes, rebuilding their
physics/contact/articulation views. That is the O(N^2) term and it matches the growth above.

**2. Every scene gets its own copy of the USD.** `Scene.prebuild()` caches the authored USD once —
then defeats it:

```python
if scene_file_path not in PREBUILT_USDS:
    ...build the scene USD...
    PREBUILT_USDS[scene_file_path] = usd_path

shutil.copyfile(PREBUILT_USDS[scene_file_path], instance_usd_path)   # per scene
return instance_usd_path
```

So `add_asset_to_stage` references a *private layer per scene*. USD composition runs N times over N
distinct files, and prims from different layers cannot share a prototype — which is why every
multi-scene log carries:

```
Prototype prims (instancing prototypes) are present in the stage but omnihydra scene graph
instancing is not enabled! Please consider enabling it and reload the stage.
```

REALM also has `carb_settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)` **commented
out** at `realm/sim_config.py:112`, and only in the `pt` branch, so it never applies to `rt`.

**Unknown:** why the copy exists. Plausibly because Isaac mutates the referenced layer and sharing
would leak edits across scenes. Dropping it is a one-line experiment, and `t1_scene_probe.py` (which
caught the 100 m z-offset) would detect cross-scene contamination immediately.

---

## What RoboLab / Isaac Lab does differently

RoboLab is built on **Isaac Lab**, not raw Isaac Sim (160 imports from `isaaclab.managers`,
`InteractiveSceneCfg`, batched `robot.data.*`). Verified in
`isaaclab/source/isaaclab/isaaclab/scene/`:

| trick | Isaac Lab | OmniGibson / REALM |
| --- | --- | --- |
| env creation | `GridCloner` clones one authored env N times (`interactive_scene.py:136-138`) | `Scene.load()` per scene; **no cloner anywhere** |
| physics | `replicate_physics=True` **by default** (`cfg:87`, `cloner.replicate_physics` at `:189`) — docstring: *"allows for faster environment creation"* | full per-scene view construction, global play/stop per import |
| assets | instanceable USDs (`panda_instanceable.usd`) | not instanceable; instancing warning in every log |
| collisions | `filter_collisions=True` by default (`cfg:101`) | n/a |
| stepping | **one batched view** across envs, `joint_pos` is `[num_envs, dof]`, one tensor op | N independent env objects, `pre_step`/`post_step` loop in Python |

That last row is exactly our linear ~5 ms/member `post_step`.

**These are architecture, not settings.** OmniGibson's multi-scene design is heterogeneous full
scenes with ~128 registered objects each, which is why it loads rather than clones. Full cloning
would be a rewrite of its scene layer and is not proposed here.

---

## Next levers, ranked

1. **Drop the per-scene `shutil.copyfile`** so all scenes reference one cached layer. One line;
   attacks the ~150 s/scene floor and is the precondition for instancing. Verify with
   `t1_scene_probe.py` that scenes stay independent.
2. **Batch the play/stop in `import_scene`** — import all N with the sim stopped, then one
   `play()` / `initialize()` each / one `step()` / `stop()`. Kills the O(N^2) growth. REALM's vector
   env already batches the *scene fixes* this way for the same reason, so the pattern exists.
3. **Enable scene-graph instancing** for `rt` — the log warning is explicitly asking for it.
4. **Instrument `import_scene`** to separate `scene.load()` from `play()`. Cheap, and it decides how
   much 1 vs 2 is worth before either is attempted.
5. **Batched inference** in `vector_eval` — only worth it once the sim scales; sequential inference
   costs ~+22 ms/step per extra member, which is small next to the render term.

Not proposed: reducing render products further (already at the minimum of 2 that pi0.5 consumes),
`rendering_mode=r` (changes what the policy sees; would need an SR A/B), GPU dynamics (broken
upstream — see the perf doc s10b).

---

## Retractions

Claims made earlier in this investigation that are **void**:

1. **"8 envs is 32% worse than 4."** Measured with rendering every step and no policy — the wrong
   regime. Under ROD, 8 beats 4 in both batches.
2. **"~9 envs is the hard ceiling."** That was the unraised `descriptorSets=10000`. 16 loads at `rt`.
3. **"The dead robot camera is already excluded / the perf doc note is stale."** Wrong — I read a
   line that had been added minutes earlier in the same session and assumed it was pre-existing. The
   perf doc's note was accurate: `droid_robolab.usd` ships 2 cameras (`wrist_camera`,
   `wrist_camera_flipped`), stock `droid_mounted.usd` ships 1.
4. **Cross-clock gap analysis.** An earlier phase breakdown mixed Isaac `[N,NNNms]` and omnigibson
   `[HH:MM:SS]` timestamps, which have different origins (diverging ~350 s by scene 15). Those
   numbers were artifacts; the table above uses one clock.

Also corrected: the perf doc's s7 item 3 claim that **wrist cameras render at 128x128** is wrong —
the robot YAML's `sensor_config: VisionSensor` block applies to robot cameras, so they render at
1280x720. That makes s7 item 4's "<1%" cost for the second camera an underestimate.

---

## Caveats

- **Node contention is not controlled.** The 3-product batch ran 2 Isaac instances per node, the
  2-product batch 3 per node, alongside an unrelated cosmos3 sweep. Within-batch comparisons are
  sound; cross-batch absolutes are not.
- **No inference in the throughput runs.** Deliberate — it isolates sim scaling. Add ~+22 ms/step
  per extra member for a production estimate.
- **N=16 throughput is not measured yet.** Capacity is; throughput is in flight.
- **One earlier N=16 run was destroyed by editing the tree mid-run.** The repo is bind-mounted at
  `/app`, so edits are live for running jobs. It picked up a patched `utils.py` against scenes loaded
  under the old robot YAML and correctly tripped `assert_wrist_camera`. **Freeze the tree while long
  runs are reading it.**
- The `descriptorSets` read is gated on an internal override field that could not be proven inert
  statically. It evidently is not set here (the fix worked), but that is where to look if it ever
  stops working.

## Reproduction

```bash
# capacity: builds members one at a time, reports GPU memory after each
MAX_ENVS=12 ./scripts/clara/interactive/go capacity ./scripts/clara/interactive/t7_capacity.sh

# profiled throughput at a given N (max juice, ROD)
NUM_ENVS=8 STEPS=96 ./scripts/clara/interactive/go scaling_n8 ./scripts/clara/interactive/t8_scaling.sh

# same as an sbatch on another node
NUM_ENVS=8 sbatch --export=ALL,NUM_ENVS=8,STEPS=96 scripts/clara/interactive/sbatch_scaling.sh
```

## Uncommitted work this rests on

At time of writing, both repos have working-tree changes:

- **OG-lite**: `omnigibson/simulator.py` — the `descriptorSets` / `reservedDescriptors` injection.
- **REALM_og391**: the robolab 2-camera filter (`DROID_robolab.yaml`, `_v2.yaml`,
  `inference/utils.py`, `env_dynamic.py`), `env_vector.py`'s `on_first_env_built` hook, the switch of
  the default robot to `DROID_robolab` across the harness, and the untracked
  `t8_vec_scaling.py` / `t8_scaling.sh` / `sbatch_scaling.sh`.

None of it is committed yet.
