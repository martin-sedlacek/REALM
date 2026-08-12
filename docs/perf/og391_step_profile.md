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
