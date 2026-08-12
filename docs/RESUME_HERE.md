# Resume notes -- 2026-08-12

Written as a handoff: this machine is being retired mid-investigation. Single entry point for
picking the work back up elsewhere. Detail lives in the two linked documents; this file is the map
and the open-thread list.

## Where things stand

| thread | state |
| --- | --- |
| og391 port (OmniGibson 1.1.1 -> 3.9.1) | done, pi0.5 completes tasks, 16/16 perturbations pass |
| robolab asset (`droid_robolab`, `_v2`) | done, SR 0.750 / TP 0.850 under OG-lite |
| Apptainer sif + dataset on clara | done, checksums matched |
| repo cleanup + file splits | done, see below |
| **vectorized environments** | **works, one open bug** -- [docs/vector_env/README.md](vector_env/README.md) |
| **OG-lite contact cache** | **half verified** -- [docs/perf/og391_step_profile.md](perf/og391_step_profile.md) |

## Open thread 1: vector env scene fixes apply only to scene 0

Full writeup with evidence frames: [docs/vector_env/README.md](vector_env/README.md).

4 environments load, tile, render distinct observations and step together correctly. But
`apply_scene_fixes_from_cfg()` seems to take effect only in scene 0: in scenes 1..N-1 the breakfast
table is never pinned and the chair that should be deleted is still present, so the task objects end
up on the rug. Frames committed under `docs/vector_env/frames/`.

**First action on the new machine** -- one cheap run that discriminates between the two leading
hypotheses (globally-numbered object names vs. my batched stop/play):

```bash
docker exec realm_stock bash -lc 'cd /app && conda run --no-capture-output -n behavior \
  python -u examples/03_vector_first_frames.py --num_envs 1 --task_id 0'
```

If the table is correct at `num_envs=1`, the batching is implicated. If it is already wrong, the
naming hypothesis is. Then run the per-scene object-name dump given in the vector_env doc.

## Open thread 2: the incremental contact cache has never been exercised

`e30899f` (OG-lite's fix for the proximity-gate row mismatch) **is verified** -- a 2-step eval exits
0, no assert, all four artifacts with a populated row, and it took the gated branch rather than the
wildcard fast path.

`gm.INCREMENTAL_CONTACT_CACHE` is still **off everywhere**, so the incremental fold contributes
nothing. The run that would exercise it never happened (I killed the agent partway):

```bash
docker exec -e REALM_INCREMENTAL_CONTACT_CACHE=1 realm_oglite bash -lc 'cd /app && \
  conda run --no-capture-output -n behavior python -u examples/02_evaluate.py \
  --task_id 0 --perturbation_id 0 --repeats 1 --max_steps 2 --model_name debug \
  --model_type debug --port 8000 --experiment_name oglite_verify --run_id inc_on \
  --log_dir /app/logs/oglite_verify'
```

Correctness first, then timings via `tmp/fork_ab_profile.py` with the flag on vs off. The contact
cache is ~50% of stepping and ~98% of `_non_physics_step`, so a working fold should be obvious rather
than lost in noise. `REALM_PROXIMITY_GATE=0` is the escape hatch if the gate misbehaves; also untested.

## Gotchas that already cost time here

1. **Never `conda run` without `--no-capture-output`.** It buffers all output until exit; killing the
   process destroys the entire log. This wiped one 9-minute investigation.
2. **Exit code 0 is not sufficient evidence.** The OG-lite failure mode asserts inside an Isaac
   callback and then segfaults. Always also check the log for `Traceback` / `Segmentation fault` /
   the specific assert, *and* that all four artifacts exist with a populated data row.
3. **Verify which OmniGibson is live** before trusting any comparison: `md5sum` the container's
   `/behavior-src/OmniGibson/omnigibson/utils/usd_utils.py` against the host OG-lite checkout.
   `realm_stock` has no bind mount there and runs the image's own copy; `realm_oglite` does.
4. **Run-to-run variance reached 17%** with identical code. n>=3 per side, interleaved, before
   believing any single-digit difference. Compare stepping time, never wall clock -- startup is 64%
   of wall and swamps everything.
5. **GPU ceiling:** 4 scenes peaked at ~26 GB of 32 GB with a 16.6 GB policy server resident. Do not
   run two Isaac Sims concurrently.
6. A YAML trailing comma makes `False,` parse as the truthy **string** `"False,"`. This silently
   turned on gravity compensation for every DROID variant once.

## What changed in the repo recently

- `realm/helpers.py` (409 lines) split into `geometry.py` / `placement.py` / `categories.py` /
  `perturbations/v_aug.py`; `sim_config.py` merges the two simulator-config functions;
  `env_config.py` holds the 181-line config assembly; `env_dynamic.py` 712 -> ~430 lines.
  Verified: 29/29 bodies AST-identical, 18 geometry functions bit-identical over 200 random inputs
  each, full eval against stock exits 0.
- Vector env support (see thread 1).
- Dead code removed: `_panda_fk` and its constants, three stale `MAX_CAMERA_*_DEVIATION` duplicates
  (`v_view.py` re-declares them locally), `set_sim_config`'s unused `rendering_mode` parameter.

## Still deferred (agreed, not forgotten)

- 7 near-identical `DROID*.yaml` configs, 60-70% duplicated.
- `is_grasping`'s `0.45` threshold looks like a typo for `0.045`.
- `evaluate()` is a 322-line function; splitting it was held back deliberately while the robolab
  benchmark numbers were still being validated. That reason has now expired.
- `n_pre_obs_renders=2` has never been verified as sufficient (only that 3 was unjustified).
- `MISSING_PERTURBATIONS` / `SUPPORTED_TASK_TYPES` / `SKILL_COMPATIBILITY_MATRIX` in
  `env_dynamic.py` are unused, but encode design intent, so they were left alone rather than deleted.
