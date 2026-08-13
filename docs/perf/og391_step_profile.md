# Where the time goes in a REALM eval (OmniGibson 3.9.1)

Measured 10-12 Aug 2026 on the og391 port. Every number here was measured, not estimated; where a
figure is inferred or unreliable it says so. Reproduction harnesses and caveats are at the bottom.

## Setup

| | |
| --- | --- |
| Task | `task_id=1` `put_banana_into_box`, perturbation `Default` |
| Robot | `DROID_robolab` (13 DOF, mimic-joint 2F-85) |
| Policy | pi0.5 via openpi websocket server on `127.0.0.1:8000` |
| Shape | 3 repeats x 100 steps (300 evaluated steps, plus 3x30 warmup steps) |
| Rates | 15 Hz control, 120 Hz physics -> 8 physics substeps per control step |
| GPU | single RTX 5090, shared with the policy server |
| Image | `realm:og391`; OG-lite bind-mounted over `/behavior-src/OmniGibson` where stated |

## 1. Whole-eval budget

Wall clock 299.1 s for 3x100 steps:

| phase | cost | share of wall |
| --- | --- | --- |
| **one-off startup** (Isaac boot + scene import + robot load) | **192.8 s** | **64.4%** |
| 3 x (reset + warmup + 100 steps) | 96.1 s | 32.1% |
| logging, parquet writes, shutdown | 10.2 s | 3.4% |

`og.Environment.__init__` alone is 185.2 s of the 192.8 s startup. Startup is untouched by every
optimisation tried and is the single largest line item in a short eval.

**Implication for sweeps:** the fix for startup is amortisation, not optimisation. At 3 repeats you
pay 193 s to do 96 s of work; at 25 repeats the same 193 s covers ~800 s. Anything that spawns one
process per rollout (as `tests/test_perturbations_integrity.py` does, one Isaac boot per
perturbation) pays it every time.

## 2. Per-repeat fixed cost

| repeat | reset | warmup (30 steps) | rollout (100 steps) | total |
| --- | --- | --- | --- | --- |
| r0 (cold) | 6.71 s | 8.00 s | 21.33 s | 36.04 s |
| r1 | 5.25 s | 6.74 s | 17.68 s | 29.67 s |
| r2 | 5.54 s | 8.00 s | 16.84 s | 30.38 s |

- **Cold-reset premium is only +1.32 s** (6.71 s vs 5.39 s mean for later resets). The expensive part
  of a cold start is all in the 193 s startup, not in the first reset.
- **Steady state is ~5.4 s reset + ~7.4 s warmup = ~12.8 s of fixed cost per repeat**, before a
  single evaluated step runs. At `max_steps=100` that is 43% overhead per repeat; at the paper's 800
  steps it is ~9%.
- Warmup steps cost **more** than rollout steps (238.7 vs 186.1 ms) because warmup renders every
  step regardless of `render_on_demand`.

## 3. Inside a rollout step

Rendering every step, 300 rollout steps, mean 186.1 ms/step:

| component | ms/step | % of stepping |
| --- | --- | --- |
| `og.sim.step` | 181.6 | 97.5% |
| -- `_non_physics_step` | **99.3** | **53.3%** |
| -- physics substeps + in-step render | 82.2 | 44.1% |
| `VideoRecorder.add_frame` | 17.4 | 9.3% |
| policy inference (amortised over the chunk) | 10.0 | 5.4% |
| `env.get_obs` (observation readback) | 3.4 | 1.8% |
| -- `VisionSensor._get_obs` x3 cameras | 2.0 | 1.1% |
| `check_collisions` | 3.0 | 1.6% |
| task progression | 0.8 | 0.4% |
| `check_grasp_condition` | 0.8 | 0.4% |
| `env._pre_step` (action -> controllers) | 0.3 | 0.1% |
| `extract_from_obs`, `get_ee_pose` | 0.07 | 0.0% |

Notes:

- **Skipping a render does not skip physics.** A blind step still runs all 8 substeps. The render is
  only ~22.8 ms of the 82.2 ms combined `_sim_context.step(render=True)` call, which caps any
  render-side optimisation at ~12% of stepping.
- Physics alone is ~59.4 ms/step, derived from `_sim_context.step(render=False)` on blind steps.
  That figure comes from 8 separate un-batched calls so it likely over-states pure physics, making
  22.8 ms a *lower bound* on the render's marginal cost.
- Observation readback is negligible (1.8%) despite two 1280x720 cameras. This was a surprise and
  rules out the obvious "GPU->CPU copy" theory.

## 4. `_non_physics_step` is almost entirely the contact cache

| | stock 3.9.1 | OG-lite |
| --- | --- | --- |
| `_non_physics_step` | 52.65 s | 44.62 s |
| -- `RigidContactAPI.update_contact_cache` | **51.59 s (98.0%)** | **42.46 s (95.2%)** |
| -- object init + systems + object-state propagation | 1.06 s (2.0%) | 2.16 s (4.8%) |

Since `_non_physics_step` is ~53% of stepping, **the contact cache alone is roughly half of all step
time** -- more than physics and rendering combined, ~30x the observation readback.

It is also bimodal: ~23-28 ms on most steps, with a **~300 ms spike on ~28% of steps**. The spike is
render-independent (it persists on steps that render nothing at all), and consistent with
contact-cache cost scaling with the number of contact pairs, which jumps the moment the gripper
touches an object.

**Caveat:** `update_contact_cache` was sampled 407 times against 395 `_non_physics_step` calls -- it
is also invoked by `og.sim.step_physics()` and during reset -- so the share above is inflated by
roughly 3% and per-step pairing between the two series is invalid. Correcting for that still leaves
~96%. The conclusion holds; a precise spike decomposition needs proper call nesting.

### Ruled out by measurement

Culling object states and visual updates (`OBJECT_STATE_UPDATE_WHITELIST=["ToggledOn"]` +
`ENABLE_VISUAL_UPDATES=False`) changes `_non_physics_step` from 29.79 s to 30.14 s -- **no effect**.
Those flags gate only the 2-5% slice, which is why they cannot help. They are set in
`realm/eval.py` as of `e3b1337` anyway; they are harmless, not useful.

## 5. The render skip (`--render_on_demand`)

OG-lite, 3x100, render-every-step (A) vs render-skip (B). Default is **on** as of `1960f10`.

| | A: every step | B: skip |
| --- | --- | --- |
| wall | 299.1 s | 288.5 s (-3.5%) |
| 3 repeats (reset+warmup+rollout) | 105.5 s | 94.8 s (**-10%**) |
| per-step mean | 186.2 ms | 174.2 ms |
| per-step **median** | 140.2 ms | **79.4 ms (-43%)** |
| in-step render | 22.8 ms/step | 2.7 ms/step |
| explicit `sim.render()` passes | 0 | 3.4 ms/step |
| `VideoRecorder.add_frame` | 17.4 ms/step | 2.2 ms/step |

- 264 of 300 control steps (88%) skipped the in-step render. Net render passes 300 -> 108 (-64%).
- **Quote the median, not the mean.** Mean delta -11.9 ms has a bootstrap 95% CI of
  [-9.5, +33.5] ms and is not significant. Median delta -58.7 ms, CI [41.1, 67.0],
  Mann-Whitney p = 4.2e-10.
- The single biggest saving is **not** the render: it is `add_frame` (-4.55 s), because B records 39
  frames instead of 300. That is a quality trade, not free speed.
- The theoretical ceiling if rendering were free is 12.2% of stepping; the skip captured ~75% of it.
  The optimisation is near-maximal -- there simply is not much on the render side.

Two consequences of the default being on:

1. Recorded video drops to ~1 frame per action chunk (~39 frames per 300 steps).
2. Trajectories are not bit-identical to render-every-step, so results are **not strictly comparable**
   to baselines recorded before `1960f10` -- including `logs/ab_oglite` (stock, SR 0.33, n=6) and the
   robolab OG-lite verification (SR 0.75, n=4), both taken with rendering every step.

`n_pre_obs_renders` was reduced 3 -> 2 in `1960f10`. Two is the count OmniGibson's own
`Simulator.step()` documents as needed for a stage change to reach the rendered image. **Nobody has
verified 2 is sufficient** -- only that 3 was never justified. The exposure is a stale wrist image on
the render step following up to 8 blind steps. The cheap check: capture RGB after 1/2/3/4 passes and
measure pixel delta against a converged frame.

## 6. OG-lite vs stock 3.9.1: not established

Rendering every step, interleaved runs, stepping time for 390 steps:

```
stock:  92.15 s,  94.03 s,  78.87 s    mean 88.35 s   spread 17.2%
lite :  81.93 s,  81.98 s,  82.77 s    mean 82.23 s   spread  1.0%
```

The mean gap is 6.12 s (6.9%) but the worst within-condition spread is 15.16 s -- 2.5x larger --
and the **fastest stock run beat every OG-lite run**. Welch's t gives p ~ 0.2. Per-step medians
overlap too (stock 163.3/156.5/148.0 vs lite 150.2/147.6/146.1). An earlier 2-sample pair suggested
~12% and did not survive the third sample; an even earlier measurement had OG-lite 3% *slower*.

**Direction is suggestive (~5-7% favouring OG-lite), magnitude is not resolvable at this sample
size.** Resolving 6% against 17% noise needs n>=8 per side, or a lower-variance protocol.

What *does* look robust is the **variance**: OG-lite held 1.0% spread across runs where stock swung
17.2%. With n=3 that could be luck, but it would matter for sweep planning even if the mean gain is
nil. Stock's outlier run was faster *everywhere* (startup 203 s vs 230 s), pointing at machine state
-- clocks, thermals, contention with the policy server -- rather than code.

### Why a large fork win was never likely

**REALM calls none of OG-lite's features.** `git grep` over `realm/` and `examples/`:

| feature | REALM references |
| --- | --- |
| `step_blind()` | 0 |
| `CONTACT_REPORTING_PATTERNS` | 0 |
| `OBJECT_STATE_UPDATE_WHITELIST` | 1 (set in `eval.py`, measured worth ~0) |
| `ENABLE_VISUAL_UPDATES` | 1 (set in `eval.py`, measured worth ~0) |
| `RENDER_ON_STEP` | 1, and it is a **comment** |

The render skip uses stock 3.9.1's `og.sim.render_on_step()` context, not OG-lite. The fork differs
from stock in 4 files (~168 lines: `simulator.py`, `macros.py`, `envs/env_base.py`,
`prims/rigid_prim.py`) plus a missing `learning/` package. Crucially **`utils/usd_utils.py` has 0
changed lines**, and that is where `update_contact_cache` -- half of all step time -- lives.

### Repeat with the render skip on (the current default)

The runs above used `render_on_demand=False`, i.e. they never exercised the skip path. Repeating with
it on does not change the conclusion. Stepping time for 390 steps:

```
ROD off   stock:  92.15 s,  94.03 s,  78.87 s    mean 88.35 s
          lite :  81.93 s,  81.98 s,  82.77 s    mean 82.23 s
ROD on    stock: 102.50 s,  84.60 s               mean 93.55 s
          lite :  96.50 s                         (n=1, see below)
```

OG-lite's single valid ROD run (96.5 s) lands *between* the two stock ROD runs, so the fork ordering
does not even reproduce in sign here. Enabling the skip did not lower stepping time either -- it went
nominally up in both forks, which is consistent with section 5: the in-step render is only ~12% of
stepping, so the skip's win is concentrated in `add_frame` and per-repeat cost, not in physics.

The second OG-lite ROD run **segfaulted (exit 139) before startup finished** -- 0 completed control
steps, `startup` never recorded. Its 193.8 s wall clock is a crash, not a fast run, and must not be
read as a sample. n=1 for that cell.

That crash is attributable. OG-lite's `omnigibson/utils/usd_utils.py` was rewritten at 17:07:24 by
`04fc69b` ("Trim the non-physics step: incremental contact cache + proximity gate") and `6c51667`.
The run started ~17:08:17 -- after the edit -- so it loaded the new contact cache and died on its own
validation assert:

```
AssertionError: RigidContactAPI contact-view row mismatch.
Expected 54 dynamic rows, got 271 rows. Missing rows (0): []. Extra rows (217): [...]
```

The proximity gate's row filter disagrees with the contact view actually built for the scene: it
expects to track 54 dynamic bodies but the view carries 271, the extras being static scene furniture
(`armchair_*/base_link`, `bookcase_*/base_link`, ...). Runs that started *before* 17:07 (`lite_rod1`,
`stock_rod1`) are unaffected, and the stock container never loads OG-lite at all -- which is why only
this one cell crashed.

This is the section-7 lever #1 being built upstream, so the numbers in this document describe the
contact cache *before* that work. They will need retaking once the gate passes its own assert.

**Standing conclusion: no OG-lite speedup is established, with or without the render skip.**

> **Superseded 2026-08-13 -- see [section 8](#8-pre-port-og-111-vs-ported-391-the-reference-benchmark).**
> Everything in this section was measured *before the proximity gate worked*: `04fc69b` introduced it
> and broke REALM outright, and `e30899f` fixed it only on 2026-08-12. With a working gate, OG-lite
> cuts `og.sim.step` median by 55% and `_non_physics_step` median by 68% against stock in the same
> image. The "not established" verdict applies to the pre-gate fork, not to the current one.

### State of the OG-lite contact-cache work (2026-08-12, handoff)

Three OG-lite commits land the lever this document ranks first, and two of them had never been run
against REALM when this was written:

| commit | what | runtime status |
| --- | --- | --- |
| `04fc69b` | incremental contact cache + proximity gate | broke REALM (row mismatch, above) |
| `6c51667` | gate cache safe for multi-scene vector envs | -- |
| `e30899f` | fix the row mismatch: explicit sensor path list + list-of-lists filters when rows are gated, wildcard kept when nothing is gated | **VERIFIED 2026-08-12** |

`e30899f` is confirmed working. A 2-step REALM eval on task 0 in `realm_oglite` exited 0 with zero
`row mismatch` / `Traceback` / `Segmentation fault` hits and wrote all four artifacts with a populated
data row. Since the assert used to fire inside `simulator.play() -> update_handles() -> initialize_view`,
before any step could run, a completed rollout is proof the gated view builds. It exercised the new
branch rather than the wildcard fast path: this same task previously failed with `Expected 54 dynamic
rows, got 271`, so 217 bodies are gated out here. Its `collisions_self=1, collisions_env=0` matched the
stock-container row for the same task, so the gated matrix did not silently drop contacts -- though on
a 2-step run that is weak evidence.

Still unverified: `REALM_INCREMENTAL_CONTACT_CACHE=1` (run 2 above) never ran, so the incremental fold
remains unexercised and unmeasured. `REALM_PROXIMITY_GATE=0` was never needed, so the fallback is also
untested.

### Update 2026-08-13 (Clara, L40S job 190155): the incremental fold is VERIFIED CORRECT

Run 2 above has now been executed. `gm.INCREMENTAL_CONTACT_CACHE=1` **passes**, so the incremental
fold is no longer unexercised. Harness: `tmp/interactive/` (`rr` for the container, `t2_inc_on.sh`
for the run, `check_run.py` for the criteria).

It was confirmed to be a real test before the simulator booted — the flag reaching `gm` is not
something to assume:

```
gm.INCREMENTAL_CONTACT_CACHE = True    gm.PROXIMITY_GATE_ENABLED = True
gm.PROXIMITY_GATE_RADIUS    = 1.5      gm.CONTACT_REPORTING_PATTERNS = None
usd_utils has incremental fold branch: True    has proximity gate: True
```

Pass on all four criteria: exit 0; zero hits for `row mismatch` / `Traceback` /
`Segmentation fault` / `AssertionError` / `core dumped`; all four artifacts written; one populated
data row each. `collisions_self=1, collisions_env=0` — **identical to the `gate_on` reference row
and to the stock container** for this task, so the fold is not silently dropping contacts. On a
2-step run that is still weak evidence for equivalence; a long rollout has not been done.

**Timings remain unmeasured** — correctness only says the fold runs, not that it is faster.

Two notes for whoever does the timing A/B:

1. **Do not use `--model_type debug` for it.** The debug client returns a *constant* action
   (`np.zeros(8)` for joint control, `realm/inference/client.py:33`), so the gripper never touches
   anything and the contact matrix never leaves the cheap ~23-28 ms mode. Since the whole point is
   the ~300 ms spikes that appear when the gripper contacts an object, a debug A/B measures the one
   regime where the fold matters least. Use pi0.5.
2. **`--render_on_demand` does not confound it.** `update_contact_cache()` is called before the
   `blind` early-out in `Simulator._non_physics_step`, so a blind step still pays the full contact
   cache. The ROD default being on does not reduce the call count.

`tmp/fork_ab_profile.py` did not survive the old machine; it is recreated as
`tmp/interactive/profile_step.py` (+ `analyze_ab.py`, `t2_ab_contact.sh`).

Its author's note: the 23 unit tests do not touch `initialize_view`, which needs a live PhysX view, so
a REALM run is the only thing that confirms the gated path builds. Fallback if it regresses is
`gm.PROXIMITY_GATE_ENABLED = False`.

Separately, `gm.INCREMENTAL_CONTACT_CACHE` is **off** everywhere, so the incremental fold contributes
nothing until something sets it. REALM now has both knobs in `realm/sim_config.py`, driven by env
vars so conditions can be A/B'd without editing code:

```bash
# 1. does the row-gated view build at all (the runtime check unit tests cannot do)
docker exec realm_oglite bash -lc 'cd /app && conda run --no-capture-output -n behavior \
  python -u examples/02_evaluate.py --task_id 0 --perturbation_id 0 --repeats 1 --max_steps 2 \
  --model_name debug --model_type debug --port 8000 --experiment_name oglite_verify \
  --run_id gate_on --log_dir /app/logs/oglite_verify'

# 2. same, with the incremental fold on
docker exec -e REALM_INCREMENTAL_CONTACT_CACHE=1 realm_oglite bash -lc '...same, --run_id inc_on...'

# 3. only if 1 fails: confirm the fallback
docker exec -e REALM_PROXIMITY_GATE=0 realm_oglite bash -lc '...'
```

Pass requires all of: exit 0, no `row mismatch` / `Traceback` / `Segmentation fault` in the log, and
all four artifacts written with a populated data row. Exit 0 alone is not sufficient -- the failure
mode is an assert inside an Isaac callback followed by a segfault.

For timings rather than correctness, `tmp/fork_ab_profile.py` already patches `update_contact_cache`
and `_non_physics_step`, so it measures exactly this code. Since the contact cache is ~50% of
stepping and ~98% of `_non_physics_step`, a working incremental fold should be visible immediately
rather than lost in the 17% run-to-run noise that sank the fork-level comparison.

Both macros are no-ops in the stock container: its `gm` is a `MacroDict` that accepts unknown keys,
and neither macro is defined or read there (verified).

## 7. Levers, ranked by measured size

1. **Contact cache -- ~50% of stepping.** `gm.CONTACT_REPORTING_PATTERNS` is the intended tool and is
   still `None`, so every rigid link in the scene reports contacts into an O(RxC) matrix. REALM only
   ever queries robot links vs everything (`check_collisions`) and the two finger pads vs the
   manipulated object (`is_grasping`).
   **Risk:** excluded links become invisible to *every* contact query -- `Touching`, `ContactBodies`,
   assisted grasping, and REALM's own collision and drop metrics. A too-narrow pattern silently
   zeroes `collisions_env` and `object_drops` rather than failing. Validate any pattern by re-running
   the 4-repeat robolab check and confirming SR/TP *and* the collision counts.
2. **Startup -- 64% of wall.** Amortise: more repeats per process, avoid one-process-per-rollout.
3. **External camera resolution -- ~23% of stepping is gated on it.** `camera_config.yaml` renders
   1280x720; `client.py` then `resize_with_pad`s to 224x224, and since 16:9 pads inside that square
   the policy sees ~224x126, about 3% of the rendered pixels. That resolution drives the in-step
   render (12.2%), `add_frame` (9.3%) and readback (1.8%).
   **Not measured, and not free:** it changes aliasing in what the policy sees, so it needs an A/B on
   success rate, not just speed, and it degrades the recorded video. Note the inverse problem: the
   wrist cameras are 128x128 (OmniGibson's default) and get *upscaled* to 224, so the wrist view is
   under-resolved while the external one is ~18x over-resolved.
4. **Dead camera -- <1%.** `droid_robolab` mounts two cameras on `base_link`; only `Camera:1`
   (`wrist_camera_flipped`) is consumed, but `Camera:0` is rendered and read back every step.
   `Robot.__init__` accepts `exclude_sensor_names`. **Trap:** `ROBOT_OBS_PROFILES` hardcodes
   `wrist_camera_idx=1`; if excluding a sensor renumbers the survivor to `Camera:0` the lookup falls
   through to its "first camera" fallback silently. Fix both together.

## Reproduction

Harnesses live in `tmp/` (gitignored, so they may be gone):

- `tmp/fork_ab_profile.py <label> [rod]` -- fork A/B plus `_non_physics_step` internals. Pass `rod`
  to enable `render_on_demand`. Run once per container, diff the JSONs.
- `tmp/timed_phase_breakdown.py` / `tmp/analyze_timing.py` -- phase-tagged breakdown (every sample
  carries `r<N>.{reset,warmup,rollout}`), which is what produced sections 1-3 and 5.

Containers: `realm_oglite` (OG-lite bind-mounted) and `realm_stock` (image's own OmniGibson). Verify
which source is live before trusting a comparison -- compare the in-container `simulator.py` md5
against the host checkout; a bind mount existing does not prove the import resolves through it.

### Instrumentation gotchas

1. `og.sim` is `None` until `og.Environment.__init__` runs, and `Simulator` is defined inside a
   factory function so it has no importable class. Patch the singleton's **bound methods** from a
   wrapper around `Environment.__init__`.
2. `Simulator.step()` renders *internally* via `_sim_context.step(render=True)`. Patching
   `og.sim.render` catches only *explicit* render calls. To separate physics from the in-step render
   you must patch `og.sim._sim_context.step` and key on its `render=` argument.
3. Write raw per-sample data to JSON **before** printing any summary. A formatting bug must not
   destroy a 6-minute measurement (this happened once).
4. Startup dominates wall clock, so compare **stepping time**, never wall clock, when judging a
   step-loop change.
5. Run-to-run variance on this machine reached 17% with identical code. Interleave conditions and
   take n>=3 per side before believing any single-digit difference.

## 8. Pre-port (OG 1.1.1) vs ported (3.9.1): the reference benchmark

Measured 2026-08-13, Clara, jobs **190216 / 190217 / 190218**. One profiler
(`tmp/interactive/profile_phases.py`) staged into *both* checkouts, patching only what exists in
both. Identical eval arguments everywhere: task 0, perturbation 0, 3 repeats x 100 steps
(390 control steps including warmup), horizon 8, robot DROID, `--model_type debug`,
`rendering_mode rt`.

| | og111 + OG-lite (1.1.1) | og391 stock (3.9.1) | og391 + OG-lite (3.9.1) |
| --- | --: | --: | --: |
| wall | 588.0 s | 411.1 s | **347.8 s** |
| -- isaac import | 12.7 s | 47.6 s | 46.3 s |
| -- **cold start** (to first env ready) | 278.3 s | 251.0 s | 257.5 s |
| -- **rollout** (3 resets + 390 steps + logging) | 309.7 s | 160.1 s | **90.3 s** |
| reset, median per repeat | **2.88 s** | 6.43 s | 9.20 s |
| `og.sim.step` total / median | 70.1 s / 105.6 ms | 115.7 s / 237.9 ms | 42.3 s / 107.4 ms |
| explicit `og.sim.render` count / total | 1294 / 211.0 s | 402 / 10.0 s | 402 / 9.4 s |
| `_non_physics_step` median / total | 0.72 ms / 0.6 s | 134.2 ms / 74.5 s | 42.7 ms / 16.5 s |
| step + render work | 281.1 s | 125.7 s | **51.7 s** |

### What is actually comparable across the two stacks

**Only cold start and rollout wall.** Everything else is measured differently on each side because
the two stacks split the work differently:

- 1.1.1's `--og_lite` path calls `env.omnigibson_env.step_blind()` for blind steps
  (`realm/eval.py:491`), which bypasses `RealmEnvironmentDynamic.step` **and skips
  `_non_physics_step` entirely**. Hence `RealmEnv.step` fires 90 times on 1.1.1 against 390 on
  3.9.1, and the 1.1.1 `_non_physics_step` median of 0.72 ms is not the same quantity as 3.9.1's.
- 1.1.1 renders through explicit `og.sim.render()` (1294 calls); 3.9.1 renders inside
  `_sim_context.step(render=True)` and uses explicit renders only to flush before an observation.

So read the per-method rows as *within-stack* diagnostics, not as a 1.1.1-vs-3.9.1 ranking.

### The port is faster at rollout and slower at reset

- **Rollout: 309.7 s -> 160.1 s, a 1.9x speedup**, or **3.4x (90.3 s) with OG-lite**, for the same
  390 steps. Combined step+render work drops 281.1 s -> 125.7 s -> 51.7 s.
- **Cold start improves only ~10%** (278.3 -> 251.0 s), and it is a wash of two opposite moves: the
  Isaac import got **3.7x slower** (12.7 -> 47.6 s, pip-installed Isaac Sim 5.1 vs the baked 4.x
  image) while env creation got faster (265.7 -> 203.4 s). Startup is still the single largest line
  item in a short eval on either stack.
- **Reset regressed 2.2x-3.2x**: 2.88 s -> 6.43 s stock -> 9.20 s with OG-lite. At the paper's 25
  repeats that is +89 s (stock) or +158 s (OG-lite) of pure per-repeat overhead versus 1.1.1. This
  is the one place the port is clearly worse and it has not been investigated. n=3 resets per run.

### OG-lite on 3.9.1 is now a large, clean win -- this supersedes section 6

Stock vs OG-lite here are the **same code paths in the same image**, differing only by the bind, so
this comparison does not suffer the caveats above:

| | stock 3.9.1 | OG-lite 3.9.1 | delta |
| --- | --: | --: | --: |
| rollout | 160.1 s | 90.3 s | **-44%** |
| `og.sim.step` median | 237.9 ms | 107.4 ms | **-55%** |
| `og.sim.step` total | 115.7 s | 42.3 s | -63% |
| `_non_physics_step` median | 134.2 ms | 42.7 ms | **-68%** |
| `_non_physics_step` total | 74.5 s | 16.5 s | -78% |

Section 6's standing conclusion -- "no OG-lite speedup is established" -- was measured **before the
proximity gate worked**. `04fc69b` introduced it and broke REALM outright (the row-mismatch assert);
`e30899f` fixed it and was only verified on 2026-08-12. These are the first numbers taken with a
working gate, and the effect is 8-10x larger than the 5-7% that section 6 could not resolve, showing
up in two independent metrics. **Note this is the proximity gate alone**:
`gm.INCREMENTAL_CONTACT_CACHE` was at its default (off) in all three runs.

It also lands exactly where section 4 predicted: `_non_physics_step`, which is ~98% contact cache.

### Caveats -- do not quote these as settled

1. **n=1 per condition.** The effect sizes are far larger than the 17% run-to-run variance section 6
   measured, but nobody has replicated them.
2. **All three jobs ran concurrently on l40s-05.** That controls for machine state -- the drift that
   sank section 6 -- but shared CPU and memory bandwidth inflate all three absolute numbers.
3. **`--model_type debug` returns a constant action**, so the gripper never contacts anything and the
   contact matrix never enters the ~300 ms spike regime of section 4. Per-step figures are a
   **floor**, not a pi0.5 rollout cost. The workload is identical across conditions, which is what
   makes the comparison internally fair.
4. Part of the 1.1.1 render cost is configuration, not stack: 1.1.1 ran `n_pre_obs_renders=3` against
   the port's 2. But per-render cost also differs 6.6x (163 ms vs 23-25 ms), which is not config.

Reproduce:

```bash
sbatch tmp/interactive/sbatch_phase_ref_og111.sh
OGLITE=0 LABEL=og391_stock  sbatch tmp/interactive/sbatch_phase_ref_og391.sh
OGLITE=1 LABEL=og391_oglite sbatch tmp/interactive/sbatch_phase_ref_og391.sh
python tmp/interactive/compare_phases.py ~/projects/REALM/logs/phase_ref
```

**Instrumentation trap that cost three jobs:** `examples/02_evaluate.py` ends with `og.shutdown()`,
and Isaac's `SimulationApp.close()` takes the process down hard -- `atexit` handlers and `finally`
blocks do **not** run. A first round of these jobs exited 0, ran to completion and wrote no results
at all. Hook `og.shutdown`, not `atexit`. Both profilers here now do, plus a periodic checkpoint
every 400 samples.

## 9. The incremental contact cache: measured, and it works

Interleaved A/B on the held allocation, 2026-08-13. OG-lite 3.9.1 both sides, differing only by
`REALM_INCREMENTAL_CONTACT_CACHE`. **pi0.5**, not `debug` -- the debug client returns a constant
action, so the gripper never contacts anything and the cache never leaves its cheap mode, which is
the regime where the fold matters least. Task 0, perturbation 0, 2 repeats x 300 steps, horizon 8,
proximity gate on (default) in every run. Order: off, on, off, on. n=2 per side.

| | fold OFF | fold ON | delta |
| --- | --: | --: | --: |
| `update_contact_cache` median | 23.99 ms | **0.070 ms** | **-99.7%** |
| `update_contact_cache` total | 15.29 s | 0.078 s | -99.5% |
| `add_contacts_from_physics_step` median | 0.322 ms | 0.725 ms | **+125%** |
| `add_contacts_from_physics_step` total | 1.62 s | 4.25 s | +163% |
| `_non_physics_step` median | 24.34 ms | **0.351 ms** | **-98.6%** |
| `_non_physics_step` total | 16.26 s | 0.443 s | -97.3% |
| **`Simulator.step` median** | 76.90 ms | **52.84 ms** | **-31.3%** |
| **`Simulator.step` total** | 48.95 s | **37.67 s** | **-23.0%** |

### The work moves, and the net is a real win

The 99.7% drop in `update_contact_cache` is not free -- the fold does the same work per physics
substep instead of in one batch at the end, which is why `add_contacts_from_physics_step` gets 2.3x
more expensive. Netting the two, per run:

| | contact-cache work |
| --- | --: |
| OFF: `update` 15.29 s + `add` 1.62 s | **16.91 s** |
| ON: flush 0.08 s + `add` 4.25 s | **4.33 s** |

**-74% of all contact-cache time**, which lands as **-23% of total `Simulator.step` time**.

The accounting closes independently: `_sim_context.step(render=False)` -- the substeps, where the
fold runs -- rises 5.97 -> 6.36 ms (+0.39 ms x 8 substeps = +3.1 ms per control step), matching the
+0.40 ms x 8 rise in `add_contacts_from_physics_step`. `_sim_context.step(render=True)` is unchanged
(+1.0%, not resolved), as it should be: the fold does not touch rendering.

### Why n=2 is enough here, unlike section 6

Section 6 failed because the **fastest stock run beat every OG-lite run** -- the ordering did not
survive run selection. That is the test to apply, not spread-vs-gap, and `analyze_ab.py` now runs it
directly. Every headline metric here is **fully separated**:

| metric | worst ON | best OFF | separated |
| --- | --: | --: | --- |
| `update_contact_cache` median | 0.071 ms | 6.49 ms | 91x, yes |
| `_non_physics_step` median | 0.359 ms | 6.81 ms | 19x, yes |
| `Simulator.step` median | 53.42 ms | 65.41 ms | yes |
| `Simulator.step` total | 37.84 s | 42.02 s | yes |

No pairing of runs reverses the conclusion.

### The fold also removes the variance

The `off` condition swings hugely between two nominally identical runs -- `update_contact_cache`
median 6.49 ms vs 41.50 ms, a 145.9% spread. That is section 4's bimodality: the cost tracks the
number of contact pairs, which jumps the moment the gripper touches something, so it is
trajectory-dependent. The `on` condition is nearly constant (0.071 / 0.068 ms, 5.2% spread). For
sweep planning that predictability may matter as much as the mean.

### Caveats

1. **n=2 per side.** Separated at every pairing, but not replicated further.
2. **One `off` run terminated early** -- 605 `Simulator.step` calls against 662 for the other three,
   a pi0.5 rollout ending early. Its totals are therefore *lower*, which flatters the `off`
   condition, so the comparison is conservative rather than inflated.
3. Measured with the proximity gate **on** in both arms. The fold's benefit on top of a gated
   contact matrix is what is reported; it has not been measured with `REALM_PROXIMITY_GATE=0`.

**Recommendation:** `REALM_INCREMENTAL_CONTACT_CACHE=1` is worth turning on by default for OG-lite
runs, subject to a correctness check on a long rollout -- the 2-step equivalence check in the section
above is weak evidence that the folded matrix matches the batched one.

```bash
N=2 ./tmp/interactive/t2_ab_contact.sh          # needs the pi0.5 server on :8000
python tmp/interactive/analyze_ab.py tmp/interactive/prof
```
