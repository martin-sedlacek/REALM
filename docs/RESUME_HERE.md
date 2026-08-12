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
| **vectorized environments** | **root-caused and fixed** -- [docs/vector_env/README.md](vector_env/README.md) |
| **OG-lite incremental contact cache** | **correctness verified; timings still open** -- [docs/perf/og391_step_profile.md](perf/og391_step_profile.md) |
| **pre-port vs ported phase benchmark** | **in flight** -- see below |

## Running anything at all: the Apptainer harness

The old workflow was `docker exec realm_stock` / `docker exec realm_oglite`. Clara has no Docker.
Both conditions are now one image plus a bind, wrapped in `tmp/interactive/rr`:

```bash
MODE=stock  ./tmp/interactive/rr python -u examples/02_evaluate.py ...   # image's own 3.9.1
MODE=oglite ./tmp/interactive/rr python -u examples/02_evaluate.py ...   # OG-lite fork bound in
ALLOC=<jobid> ./tmp/interactive/go <logname> ./tmp/interactive/<script>.sh   # run + tee + EXIT marker
```

`tmp/` is gitignored. That is exactly how `tmp/fork_ab_profile.py` was lost when the last machine
went away, so **copy anything worth keeping out of `tmp/` before the allocation ends.**

| file | what |
| --- | --- |
| `tmp/interactive/rr` | container wrapper, `MODE=stock\|oglite` |
| `tmp/interactive/go` | run a script in the held allocation, tee to `tmp/interactive/logs/<name>.log` |
| `tmp/interactive/show_macros.py` | prove a flag actually reached `gm` before spending a run on it |
| `tmp/interactive/check_run.py` | the four REALM pass criteria, so "exit 0" can't be mistaken for a pass |
| `tmp/interactive/t1_scene_probe.py` | per-member scene dump: names, z distribution, stage prims, fixes either side |
| `tmp/interactive/profile_step.py` | contact-cache / `_non_physics_step` timing (replaces the lost `fork_ab_profile.py`) |
| `tmp/interactive/profile_phases.py` | cold start / reset / step phases, **portable across 1.1.1 and 3.9.1** |
| `tmp/interactive/t2_ab_contact.sh` + `analyze_ab.py` | interleaved A/B of the incremental contact cache |
| `tmp/interactive/sbatch_phase_ref_og{111,391}.sh` + `compare_phases.py` | the pre-port vs ported benchmark |

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

Still open, and both found while measuring the above:

1. **`reset()` re-adds the removed chair, in every scene including scene 0.** `Scene.reset(hard=True)`
   restores from `_initial_file`, which is captured at the end of `Scene.initialize()` -- before
   `apply_scene_fixes_from_cfg` ever runs. `n_objects` goes 127 -> 128 and
   `straight_chair_pmpwwi_0` returns to `active=True`. Because scene 0 is affected too, **this very
   likely also happens in the single-env production path**, where `reset()` runs once per repeat.
   Not yet confirmed there. Candidate fix: `scene.update_initial_file()` after applying the fixes.
   Worth checking before trusting any result that depends on `to_remove`.
2. **Why scenes 1..N-2 do not recover on reset but the last one does.** Pre-fix, scene_3 came back
   to `above_50m=1` (just the pinned table) after warmup while scenes 1 and 2 stayed at 70. Moot for
   the fix, but it points at per-scene state being clobbered by the global play/stop that
   `Simulator.import_scene` runs for every import.

The other vectorization gaps (perturbations that cycle the sim, `reset_joints()` stepping the sim,
EE control in world vs scene frame, `evaluate()` still single-env) are unchanged and listed in
[docs/vector_env/README.md](vector_env/README.md).

## Closed: the incremental contact cache runs (was "open thread 2")

`gm.INCREMENTAL_CONTACT_CACHE=1` has now been exercised for the first time and **passes** all four
criteria, with `collisions_self=1, collisions_env=0` matching the stock container exactly. Flag
reaching `gm` was confirmed before booting the sim, so it is not a null test.

**Timings are still unmeasured.** `tmp/interactive/t2_ab_contact.sh` is written and the pi0.5 server
recipe is in it; run `N=3 ./tmp/interactive/t2_ab_contact.sh` with the server up, then
`analyze_ab.py`. Two things that harness already encodes:

- **Do not A/B with `--model_type debug`.** It returns a *constant* action, so the gripper never
  touches anything and the contact matrix never leaves its cheap ~23-28 ms mode -- the regime where
  the fold matters least. Use pi0.5.
- `--render_on_demand` does **not** confound it: `update_contact_cache()` runs before the `blind`
  early-out in `_non_physics_step`, so a blind step still pays the full contact cache.

`REALM_PROXIMITY_GATE=0` is still untested.

## In flight: pre-port vs ported phase benchmark

Reference numbers for cold start / reset / per-step on the **pre-port** stack (REALM@dev + OG-lite,
OmniGibson 1.1.1, `realm-dm.sif`) against the ported one, same eval arguments, same profiler:

```bash
sbatch tmp/interactive/sbatch_phase_ref_og111.sh                                   # 1.1.1 + OG-lite
OGLITE=0 LABEL=og391_stock  sbatch tmp/interactive/sbatch_phase_ref_og391.sh       # 3.9.1 stock
OGLITE=1 LABEL=og391_oglite sbatch tmp/interactive/sbatch_phase_ref_og391.sh       # 3.9.1 + fork
python tmp/interactive/compare_phases.py /mnt/home_lustre/sedlam56/projects/REALM/logs/phase_ref
```

Jobs 190213 / 190214 / 190215 (190212 died: it resolved `realm` from whichever directory `sbatch`
was submitted from rather than the repo bound at `/app` -- both scripts now pass
`apptainer --pwd /app`).

Caveats to carry into the writeup: all three landed on the same node and ran concurrently, which
controls for machine state but inflates absolute numbers; and `--model_type debug` means constant
actions and no gripper contact, so per-step figures are a floor rather than a pi0.5 rollout cost.

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
4. **Exit code 0 is not sufficient evidence.** The OG-lite failure mode asserts inside an Isaac
   callback and then segfaults. `tmp/interactive/check_run.py` encodes the real criteria: no
   `Traceback` / `Segmentation fault` / `row mismatch` in the log, **and** all four artifacts written
   with a populated data row.
5. **Registry removal is not stage removal.** `scene.remove_object()` ends in
   `delete_or_deactivate_prim()`, which may *deactivate*. A deactivated prim still passes
   `IsValid()`; only `IsActive()` tells you whether it still renders.
6. **Never `conda run` without `--no-capture-output`** (only relevant if you go back to a `conda run`
   workflow; `rr` does not use one). It buffers all output until exit, so killing the process
   destroys the entire log.
7. **Run-to-run variance reached 17%** with identical code. n>=3 per side, interleaved, before
   believing any single-digit difference. Compare stepping time, never wall clock -- startup is 64%
   of wall.
8. **A GPU Slurm gives you is not necessarily empty.** Check
   `nvidia-smi --query-compute-apps=...` on a fresh allocation before trusting any timing.
9. A YAML trailing comma makes `False,` parse as the truthy **string** `"False,"`. This silently
   turned on gravity compensation for every DROID variant once.

## Still deferred (agreed, not forgotten)

- 7 near-identical `DROID*.yaml` configs, 60-70% duplicated.
- `is_grasping`'s `0.45` threshold looks like a typo for `0.045`.
- `evaluate()` is a 322-line function; splitting it was held back while the robolab benchmark
  numbers were being validated. That reason has expired.
- `n_pre_obs_renders=2` has never been verified as sufficient (only that 3 was unjustified).
- `MISSING_PERTURBATIONS` / `SUPPORTED_TASK_TYPES` / `SKILL_COMPATIBILITY_MATRIX` in
  `env_dynamic.py` are unused, but encode design intent, so they were left alone.
