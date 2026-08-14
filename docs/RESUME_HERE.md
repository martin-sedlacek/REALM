# Resume notes -- updated 2026-08-13 (Clara, L40S)

Single entry point for picking this work back up. Detail lives in the linked documents; this file is
the map and the open-thread list.

Written 2026-08-12 as a handoff when the dev workstation was retired. **Continued 2026-08-13 on
Clara** under held allocation `salloc --no-shell` job 190155 (1xL40S / 32 CPU / 120 G / 24 h). Both
threads that were open then are now resolved; what replaced them is at the bottom.

## Where things stand

| thread | state |
| --- | --- |
| og391 port (OmniGibson 1.1.1 -> 3.9.1) | done, pi0.5 completes tasks, 16/16 perturbations pass |
| robolab asset (`droid_robolab`, `_v2`) | done, SR 0.750 / TP 0.850 under OG-lite |
| Apptainer sif + dataset on clara | done, checksums matched |
| repo cleanup + file splits | done |
| **vectorized environments** | **working end to end: SR 0.960 over 25 pi0.5 rollouts** -- [docs/vector_env/README.md](vector_env/README.md) |
| **OG-lite incremental contact cache** | **verified and measured: -23% of `Simulator.step`** -- [docs/perf/og391_step_profile.md](perf/og391_step_profile.md) s9 |
| **pre-port vs ported phase benchmark** | **done** -- [docs/perf/og391_step_profile.md](perf/og391_step_profile.md) s8 |

## Running anything at all: the Apptainer harness

The old workflow was `docker exec realm_stock` / `docker exec realm_oglite`. Clara has no Docker.
Both conditions are now one image plus a bind, wrapped in `scripts/clara/interactive/rr`:

```bash
MODE=stock  ./scripts/clara/interactive/rr python -u examples/02_evaluate.py ...   # image's own 3.9.1
MODE=oglite ./scripts/clara/interactive/rr python -u examples/02_evaluate.py ...   # OG-lite fork bound in
ALLOC=<jobid> ./scripts/clara/interactive/go <logname> ./scripts/clara/interactive/<script>.sh   # run + tee + EXIT marker
```

The harness lives under `scripts/clara/interactive/`, i.e. **tracked**. It used to live in `tmp/`,
which is gitignored -- that is exactly how `tmp/fork_ab_profile.py` was lost when the last machine
went away. Only artifacts (`tmp/interactive/logs/`, `tmp/interactive/prof/`) stay in `tmp/`.

| file | what |
| --- | --- |
| `scripts/clara/interactive/rr` | container wrapper, `MODE=stock\|oglite` |
| `scripts/clara/interactive/go` | run a script in the held allocation, tee to `tmp/interactive/logs/<name>.log` |
| `scripts/clara/interactive/show_macros.py` | prove a flag actually reached `gm` before spending a run on it |
| `scripts/clara/interactive/check_run.py` | the four REALM pass criteria, so "exit 0" can't be mistaken for a pass |
| `scripts/clara/interactive/t1_scene_probe.py` | per-member scene dump: names, z distribution, stage prims, fixes either side |
| `scripts/clara/interactive/profile_step.py` | contact-cache / `_non_physics_step` timing (replaces the lost `fork_ab_profile.py`) |
| `scripts/clara/interactive/profile_phases.py` | cold start / reset / step phases, **portable across 1.1.1 and 3.9.1** |
| `scripts/clara/interactive/t2_ab_contact.sh` + `analyze_ab.py` | interleaved A/B of the incremental contact cache |
| `scripts/clara/interactive/sbatch_phase_ref_og{111,391}.sh` + `compare_phases.py` | the pre-port vs ported benchmark |

## Closed: vector env scene fixes (was "open thread 1")

**The previous diagnosis was wrong and has been retracted.** `apply_scene_fixes_from_cfg()` applies
identically in *every* scene -- measured, with identical object-name digests, identical 128->127
object counts, `fixed_base` flipping to True and the `rootJoint` prim appearing on the stage in all
four members. The "globally numbered object names" hypothesis is dead: names are identical across
scene copies.

The real fault is **upstream in stock OmniGibson 3.9.1**: `Scene._load_scene_prim_with_objects`
parks the scene prim at `INITIAL_SCENE_PRIM_Z_OFFSET = -100` and then sets every scene-file object's
pose in the **world** frame while it is parked, so the offset is baked into local coordinates and the
subsequent move to z=0 lifts every object 100 m. 70 of 128 objects per scene, in scenes `idx != 0`.
REALM then pins the breakfast table with a `FixedJoint` at that lifted pose, which is why the table
was the one thing no reset could recover.

Fixed in OG-lite `ef7442b`; verified at `num_envs=4` (`above_50m=0` in all four scenes, task objects
back on the table at z=0.82 instead of 0.015 on the floor, `docs/vector_env/frames_fixed/`).
**Vector envs must now run `MODE=oglite`** -- the fix lives in the fork, not the image.

Both threads that were open here are now **closed** (2026-08-14):

1. **`reset()` re-adds the removed object -- FIXED.** `Scene.reset(hard=True)` restores from
   `_initial_file`, which `og.Environment.post_play_load()` captures *before*
   `apply_scene_fixes_from_cfg` ever runs, so the first reset undid the removal:
   `straight_chair_pmpwwi_0` came back (`n_objects` 127 -> 128) in **every** member of a Vec=3 env
   and on the plain single-env path. Since REALM calls `reset()` once per repeat, every repeat after
   the first ran with an object the task config asked to delete -- 24 of 25 at `--repeats 25`.

   **It is a port regression, not a historical problem: not an issue on 1.1.1** (per Martin,
   2026-08-13 -- do not re-verify there). Results on the old stack are unaffected; og391 results are.

   Fixed by `RealmEnvironmentDynamic.rebase_initial_file()` (`scene.update_initial_file()`) right
   after the scene fixes on both paths. Probes: `scripts/clara/interactive/t12_vec_chair.py`
   (vector, per member) and `t3_single_env_chair.py` (single env).
2. **Why scenes 1..N-2 do not recover on reset but the last one does -- ANSWERED, and it was not the
   init-queue eviction.** `Scene.restore()` cycles `og.sim.stop()` / `og.sim.play()` whenever it has
   objects to add; that cycle is global, and `og.sim.stop()` reverts every scene to its state at the
   last `play()`. So a member's restored poses -- applied by `load_state()` *after* its own play --
   are thrown away by the *next* member's `stop()`. Only the last member has no sibling behind it.
   With thread 1 fixed there is nothing to add, no member cycles the sim, and the asymmetry is gone.
   Write-up: [docs/vector_env/README.md](vector_env/README.md).

Also fixed upstream, 2026-08-14: **`Simulator._pre_remove_object` pruned the global init queue by
object NAME**, so in a vector env one member's `remove_object` evicted a sibling's freshly added
object of the same name and nothing ever initialised it. OG-lite now matches on identity.
`RealmVectorEnvironment._repair_init_queue()` is kept as a net, because `MODE=stock` /
`MODE=stockfix` still ship the stock `simulator.py`.

### Vectorized evaluation now works end to end

`realm/vector_eval.py` + `examples/04_vector_evaluate.py` run `repeats` rollouts in waves of
`num_envs`, sequential (non-batched) inference, same four artifacts as the single-env path.
Render-on-demand is implemented for vector envs too -- the render decision ORs across members because
`og.sim.render_on_step()` is a single global flag.

**Verified: 25 rollouts x 500 steps, task 0, Default, `DROID_robolab`, pi0.5, 4 envs -> SR 0.960
(24/25), TP 0.984**, against a single-env baseline of SR 1.000 at n=10. The one failure is coherent
(3 drops, stalled at `MOVE_CLOSE`, block never leaves the table). Sustained stepping is flat over 200
steps. Details and the throughput table: [docs/vector_env/README.md](vector_env/README.md).

**Capacity on one 46 GB L40S:** first scene 5204 MiB, each additional ~2728 MiB -> ~10 scenes
loadable with the pi0.5 server on the same card, ~14 with the policy served elsewhere. `play()` costs
more on top, so those are upper bounds.

**Next lever is batched inference, not more scenes.** Inference is sequential and O(N): ~+22 ms per
step per extra member, which overtakes the simulation term near 8 members and flattens per-member
throughput.

The remaining vectorization gaps (perturbations that cycle the sim, `reset_joints()` stepping the sim,
EE control in world vs scene frame, per-member tasks/perturbations) are unchanged and listed in
[docs/vector_env/README.md](vector_env/README.md).

## Closed: the incremental contact cache (was "open thread 2")

`gm.INCREMENTAL_CONTACT_CACHE=1` passes all four correctness criteria, with
`collisions_self=1, collisions_env=0` matching the stock container exactly.

**Measured under pi0.5** (interleaved, n=2/side, OG-lite both arms, gate on both) --
[section 9](perf/og391_step_profile.md):

| | fold OFF | fold ON | delta |
| --- | --: | --: | --: |
| `update_contact_cache` median | 23.99 ms | 0.070 ms | -99.7% |
| `add_contacts_from_physics_step` median | 0.322 ms | 0.725 ms | +125% |
| net contact-cache work per run | 16.91 s | 4.33 s | **-74%** |
| **`Simulator.step` median** | 76.90 ms | 52.84 ms | **-31%** |
| **`Simulator.step` total** | 48.95 s | 37.67 s | **-23%** |

The fold moves work out of the batched update and into the per-substep fold, and the net is a real
win. Every headline metric is **fully separated** -- the worst `on` run beats the best `off` run --
which is the test section 6's fork comparison failed, so n=2 suffices. It also removes the variance:
`off` swings 145.9% between identical runs, `on` 5.2%.

**Now ON by default** (`realm/sim_config.py`, 2026-08-13), together with the proximity gate, which
is pinned explicitly rather than inherited from the fork's default. No env vars needed; set
`REALM_INCREMENTAL_CONTACT_CACHE=0` / `REALM_PROXIMITY_GATE=0` to go back.

Correctness rests on OG-lite's `tests/test_contact_cache_equivalence.py`, which drives the real
methods with synthetic view data and asserts **bit-identical** matrices: **16/16 pass** across 5
seeds x {1, 2, 8} substeps plus the nothing-folded no-op. That is the check the earlier 2-step
runtime evidence could not give.

Two caveats worth carrying:

- The fold's benefit **scales with contact load**: -31% of `Simulator.step` median under pi0.5 but
  only -9% under `--model_type debug`, whose constant action never touches anything. Plan sweeps
  with the pi0.5 figure.
- `collisions_env` averaged 7.0 with the fold off and 10.0 with it on across 4 pi0.5 rollouts. That
  is **not** attributable to the fold -- rollouts are nondeterministic within a condition too
  (`inc0_r1` and `inc0_r2` diverge), and the unit test rules out a matrix difference. A paired check
  would need determinism this stack does not have.

Still untested: `REALM_PROXIMITY_GATE=0`. **`gm.CONTACT_REPORTING_PATTERNS` deliberately left off** --
it is the bigger lever but excluded links go invisible to every contact query, so a bad pattern
silently zeroes `collisions_env` and `object_drops`. It needs a validated pattern first.

**GPU physics does not work** -- see [section 10b](perf/og391_step_profile.md). Two upstream
device-consistency gaps, both firing before REALM code runs. `REALM_GPU_DYNAMICS` is wired and left
in place for when upstream fixes it.

## Done: pre-port vs ported phase benchmark

Jobs **190216 / 190217 / 190218**, one profiler in both checkouts, identical eval args. Full table
and caveats: [section 8](perf/og391_step_profile.md).

| | og111 + OG-lite | og391 stock | og391 + OG-lite |
| --- | --: | --: | --: |
| cold start | 278.3 s | 251.0 s | 257.5 s |
| **rollout** (3 resets + 390 steps) | 309.7 s | 160.1 s | **90.3 s** |
| reset, median/repeat | **2.88 s** | 6.43 s | 9.20 s |

- **The port is 1.9x faster at rollout, 3.4x with OG-lite.**
- Cold start improves only ~10%, and is a wash: the Isaac import got **3.7x slower** (12.7 -> 47.6 s)
  while env creation got faster (265.7 -> 203.4 s).
- **Reset regressed 2.2-3.2x** -- the one place the port is clearly worse. At 25 repeats that is
  +89 s (stock) / +158 s (OG-lite) of per-repeat overhead vs 1.1.1. **Not investigated; this is the
  most concrete open performance item.**
- Only cold start and rollout wall are comparable across stacks: 1.1.1's `--og_lite` routes blind
  steps through `step_blind()`, bypassing `RealmEnv.step` and skipping `_non_physics_step` entirely.

```bash
sbatch scripts/clara/interactive/sbatch_phase_ref_og111.sh
OGLITE=0 LABEL=og391_stock  sbatch scripts/clara/interactive/sbatch_phase_ref_og391.sh
OGLITE=1 LABEL=og391_oglite sbatch scripts/clara/interactive/sbatch_phase_ref_og391.sh
python scripts/clara/interactive/compare_phases.py ~/projects/REALM/logs/phase_ref
```

## Gotchas that have already cost time

1. **Never wrap an in-container command in `bash -lc`.** Apptainer binds `$HOME`, so a *login* shell
   re-sources the host `~/.bashrc`, prepends `~/miniconda3/bin` to PATH and shadows the container's
   conda env: you get host Python 3.12 and `ModuleNotFoundError: No module named 'omnigibson'`. Use
   `bash -c`, or call `python` directly. Distinct from the `apptainer exec` vs `run` trap, and it
   bites even with `run`.
2. **`gm` lies in the stock container.** `getattr(gm, "PROXIMITY_GATE_ENABLED")` returns a truthy
   `{'_read': set()}` for undefined macros rather than raising, so macro checks there are
   meaningless. Check the live source instead:
   `python -c "import inspect, omnigibson.utils.usd_utils as uu; print('PROXIMITY_GATE' in inspect.getsource(uu))"`.
3. **Set the container's working directory explicitly** (`apptainer --pwd /app`). Otherwise the job
   inherits the submit directory and can import a *different* REALM checkout than the one it bound.
4. **`atexit` never fires under Isaac.** `og.shutdown()` -> `SimulationApp.close()` takes the process
   down hard, so `atexit` handlers and `finally` blocks are skipped. Three phase jobs and four A/B
   runs completed, exited 0 and wrote **nothing at all** -- no crash to find, just silence. Hook
   `og.shutdown`; both profilers here do, plus a checkpoint every 400 samples.
5. **Exit code 0 is not sufficient evidence.** The OG-lite failure mode asserts inside an Isaac
   callback and then segfaults. `scripts/clara/interactive/check_run.py` encodes the real criteria: no
   `Traceback` / `Segmentation fault` / `row mismatch` in the log, **and** all four artifacts written
   with a populated data row.
6. **Registry removal is not stage removal.** `scene.remove_object()` ends in
   `delete_or_deactivate_prim()`, which may *deactivate*. A deactivated prim still passes
   `IsValid()`; only `IsActive()` tells you whether it still renders.
7. **Never `conda run` without `--no-capture-output`** (only relevant if you go back to a `conda run`
   workflow; `rr` does not use one). It buffers all output until exit, so killing the process
   destroys the entire log.
8. **Run-to-run variance reached 17%** with identical code. n>=3 per side, interleaved, before
   believing any single-digit difference. Compare stepping time, never wall clock -- startup is 64%
   of wall.
9. **A GPU Slurm gives you is not necessarily empty.** Check
   `nvidia-smi --query-compute-apps=...` on a fresh allocation before trusting any timing.
10. A YAML trailing comma makes `False,` parse as the truthy **string** `"False,"`. This silently
   turned on gravity compensation for every DROID variant once.

## Still deferred (agreed, not forgotten)

- 7 near-identical `DROID*.yaml` configs, 60-70% duplicated.
- `is_grasping`'s `0.45` threshold looks like a typo for `0.045`.
- `evaluate()` is a 322-line function; splitting it was held back while the robolab benchmark
  numbers were being validated. That reason has expired.
- `n_pre_obs_renders=2` has never been verified as sufficient (only that 3 was unjustified).
- `MISSING_PERTURBATIONS` / `SUPPORTED_TASK_TYPES` / `SKILL_COMPATIBILITY_MATRIX` in
  `env_dynamic.py` are unused, but encode design intent, so they were left alone.
