# Test coverage

**What the test suite does and does not establish.** Read this before treating any green run —
a CI tick, a `make test-suite` pass, an "ALL TASKS PASSED INTEGRITY CHECK!" — as evidence.

How to *run* the tests is [Running the test suite](Running-the-Test-Suite). This page is the
honest ledger of what running them proves.

> **The one-sentence version.** The suite establishes that **the simulation builds, steps, and
> writes well-formed artifacts** for every task and every perturbation. It establishes **nothing
> about whether a policy can do the tasks**, because every test drives `--model_type debug`, whose
> constant action never closes the gripper.

---

## The headline gap: no success condition is ever evaluated

`realm/environments/task_progression.py` defines **22 success conditions**. The suite exercises
**none** of them.

Every test in the suite except `test_pi0_integration` runs `--model_type debug`. That client
returns a hardcoded `np.zeros(8)` on every call, whose last element becomes gripper = −1 (open).
The arm is commanded to hold still and **the gripper never closes**. So every rollout the suite
produces stops at the first rung of the ladder:

```
task_progression  0.0
stage             REACH
binary_SR         0.0
```

Measured 2026-08-16 on this branch: the `debug` runs under `logs/pert_integrity_test_tmp/`,
`logs/matrix_verdict/` and a fresh Quick-Start smoke run are **uniformly** `stage=REACH`,
`task_progression=0.0`.

`test_pi0_integration` is the only test that drives a real policy — and it needs a live openpi
server on `:8000`, which no automated run has. It **SKIPs**. Measured: `SKIP: preconditions not
met`, 6.9 s, in job 191441's results.

> **Do not over-read this.** It is a statement about the **suite**, not about REALM. Under a real
> policy the ladder does advance: in the pi0.5 control runs on this cluster on 2026-08-16
> (`logs/ctl_gripper/`), 24 of 45 rollouts reached `SUCCESS`, through `GRASP`, `LIFT_SLIGHT` and
> `MOVE_CLOSE`. Across the whole log tree, 9,839 of 42,561 recorded rollouts are `SUCCESS`. The
> gap is that **the automated tests never look at any of that.**

## The second gap: the metric formulas never run

Every artifact-producing test uses `--max_steps 1`. At one control step, `eval.py`'s metric block
takes the false branch of `len(qpos_joints) > 4`, and six metrics are written as **literal `0.0`
without their formulas ever executing**:

`joint_vel_var`, `joint_acc_var`, `joint_jerk`, `joint_path_length`, `cart_path_length`,
`cart_jerk`.

Verified by reading the artifacts, not the code — every report row under
`logs/pert_integrity_test_tmp/` carries `0.0` in all six. **A change that broke any of those six
formulas would pass the entire suite.**

`--repeats 1` on most tests means the **per-repeat reset path is never entered** either.
`test_perturbations_integrity` is the exception at `--repeats 3`, and is the only place the suite
exercises reset.

## The third gap: a no-op passes

`V-AUG`'s entry in `realm/environments/perturbations/registry.py` is **literally the same function
as `Default`** — the no-op — because its augmentation happens in the observation path rather than
in the scene. That is correct by design.

But it means `test_perturbations_integrity` reports `V-AUG: PASS` for a perturbation that changed
nothing in the scene, using the same criteria it applies to the fifteen that do. **The test cannot
tell a working perturbation from one that silently became a no-op.** This is not hypothetical: a
perturbation that passed every numeric check while being a complete no-op is a defect this project
has already had.

The same shape applies to `V-SC` on `push_switch`, `open_drawer` and `close_drawer`, which declare
**zero distractors** — `V-SC` has nothing to re-place or re-model there, so it is inert while still
paying for a full stopped-simulator reset.

## The fourth gap: `--no_render` does not mean "no rendering"

`test_integrity` and `test_single_task` pass `--no_render`. That drops the **external** sensors
only — `env_config.py` adds those under `if not env.no_rendering`, while the robot's wrist camera
is part of the robot and keeps rendering.

So `extract_from_obs` takes its "external sensors are missing" fallback and hands the recorder a
synthetic black 128×128 for the base image, next to a **real** wrist view. Measured on the gate's
own artifact (task 0, 2026-08-16): the recorded 128×256 frame is max=2, mean=0.002 on the base half
and max=214, mean=100 on the wrist half.

**So those tests prove the wrist camera rendered. They prove nothing about the external camera —
which is the view the policies are actually conditioned on.** `test_vector_integrity` runs with
rendering on and is the one to use for that.

---

## Coverage by tier

### Tier 1 — static checks (CI, `make check`)

| Covered | |
|---|---|
| Rubric ↔ checker cross-reference | `test_task_progression_rubrics` |
| Task-type literal/config consistency | `test_task_type_literals` |
| Perturbation, cell and robot-definition contracts | four host pytest modules |
| Lint | ruff `F401`/`F811` only — see `.ruff.toml` |

| **Not covered** | |
|---|---|
| The simulator | entirely. No scene, no rollout, no artifact |
| GPU-tier suite entries | everything needing the container |
| Semantics of any config | that a YAML *parses* is not that its keys are read |
| Types, style, complexity | the ruleset is two dead-code rules and nothing else |

Tier 1 is expected green. Reproduce its locked tool environment with `uv sync --locked`, then run
`uv run make check`. Its narrow lint scope and static contracts are useful gates, not broad proof of
runtime correctness.

### Tier 2 — the GPU suite (`make test-smoke` / `test-suite` / `test-matrix`)

| Covered | By |
|---|---|
| All 10 tasks build, step and write 4 artifacts | `test_integrity` |
| All 16 perturbations apply without crashing, over 3 repeats | `test_perturbations_integrity` |
| The vector path at `num_envs=2`, tasks and perturbations | `test_vector_integrity_*` |
| Cross-member object placement, `unitsResolve` | `test_scene_object_placement` |
| Joint-reset scheduling | `test_joint_reset_batching` |
| Drawer tasks load and run | `test_single_task_drawer`, `test_vector_integrity_drawers` |

| **Not covered** | Why it matters |
|---|---|
| Every success condition (22 of 22) | see above — nothing about grasping, placing or scoring |
| Six of the metric formulas | see above — all read `0.0` at `--max_steps 1` |
| The external camera under `--no_render` | see above — the view policies actually use |
| `VB-MOBJ`'s remove/add branch | the expensive path; only the rescale branch is reached |
| Drawer reset **outcome** | that the drawer *loads* is tested; that it *ends up open* is not |
| `--resume` | no test covers it, and the sweep drivers depend on it |
| `openpi` and `dreamzero` model types | only `debug` is ever constructed |
| Robot **definitions** | The canonical `droid_mounted` definition is covered; `ur` remains outside the default DROID gates. |
| 10 of the 13 robot **configs** | no test passes `--robot` explicitly except through those two defaults; the EE-control configs, the PD-control variants, `DROID_no_wrist_cam` and all three `UR5*` are untouched |
| `MODE` agreement | no test asserts `stock` and `oglite` produce the same result |
| A policy server | `test_pi0_integration` SKIPs without one |

### The `MODE` trap, which is a coverage gap wearing a disguise

`realm_og391_v2.sif` carries six of the seven OmniGibson patches. **The missing one is the up-axis
fix**, which lives only in the OG-lite fork. Without it, referencing a layer whose `upAxis`
disagrees with the stage's makes Kit append `xformOp:rotateX:unitsResolve` to the referencing prim,
which no OmniGibson pose setter can see — and it is materialised only for the **first** reference
to the asset.

Measured on `impact_drawer`'s `cabinet.usd` at `num_envs=2`: **scene 0's cabinet lay on its back**
with its drawers jammed at 0.229 m of a 0.300 m range, while scene 1's stood upright.

**The drawer tests pass anyway.** 10/10 on `test_integrity`, 2/2 on the vector drawer cells, both
at `MODE=stock`. Nothing in `tests/` notices, because almost every check is "the artifacts exist
with the right row count" and the `debug` policy never touches the drawer.

`test_scene_object_placement` exists precisely because of this, and is the only test that looks at
the **scene** rather than at the artifacts. Measured 2026-08-16, same code, same task, different
bind:

| `MODE` | `test_scene_object_placement` | `test_integrity` (10 tasks) | `test_vector_integrity_drawers` |
|---|---|---|---|
| `stock` | **FAIL** (329.2 s; 428.5 s on a re-run) | PASS | PASS |
| `oglite` | **PASS** (334.5 s) | — | — |

> **That failing run exited 0.** Measured on job 191496, 2026-08-16: `test_scene_object_placement`
> reported `FAIL` with `exit=0`. If the suite gated on exit codes it would have called that a pass.
> It gates on the test's own printed verdict line instead, which is why it did not — and it is the
> single clearest demonstration in this repository of why "exit code 0 proves nothing" is a rule
> and not a caution.

**That row is the whole point of this page.** The two tests that exercise the drawer *pass* under
the bind where the drawer is physically wrong; the one test that looks at the scene is the only one
that notices. Do not "fix" it by loosening its tolerance.

So: `make test-suite` re-runs the OG-lite-sensitive cells under `MODE=oglite` as a second
invocation, and `run_suite.py` records `mode` **per result** rather than as a header field. **Always
say which mode a result came from.**

---

## What a green run is worth

| Signal | Means | Does not mean |
|---|---|---|
| tier 1 green | the repo is syntactically intact and the suite's wiring resolves | anything about the simulator |
| tier 2 green | the simulation builds, steps and writes well-formed artifacts for every task and perturbation | that a policy can do the tasks, that the scene is physically correct, or that any metric is right |
| `test_pi0_integration` SKIP | no policy server was available | that it would have passed |

**There is deliberately no CI badge in the README.** A "passing" badge here would be read as the
middle column of that table when it only earns the first.

## See also

- [Running the test suite](Running-the-Test-Suite) — how to run each tier
- [Known issues and gotchas](Known-Issues-and-Gotchas) — why exit code 0 proves nothing
- [Tasks and perturbations](Tasks-and-Perturbations) — `V-SC`'s inert tasks, the unusable camera views
