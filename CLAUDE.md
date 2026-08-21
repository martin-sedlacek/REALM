# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. Rewritten 2026-08-19
against the current tree (the previous version described the pre-OG-3.9.1 layout). The wiki/
directory is the maintained operator documentation — prefer pointing at it over restating it.

## What REALM is

A simulation benchmark for evaluating generalization of robotic manipulation policies (VLA models
such as Pi0/Pi0.5 via openpi, and DreamZero). Ten manipulation tasks are evaluated against 16
perturbation types (visual, semantic, behavioral) on OmniGibson **3.9.1** / IsaacSim. Everything
sim-side runs inside a container (`.docker/realm_og391.Dockerfile` / `realm_og391.def`); nothing
here is pip-installable — code runs with the repo on `PYTHONPATH`.

## The one rule that governs all changes

**REALM is a benchmark: a change that moves a number is a bug, even when the old number was
wrong.** Concretely:
- Refactors must preserve behavior bit-for-bit, including **RNG draw order** — removing or
  reordering a single `np.random`/`random` call shifts every draw after it. Dead draws are kept
  and commented rather than deleted.
- Known number-moving bugs are flagged `KNOWN ISSUE` in place and fixed only in a gated batch
  behind a `VERSION` bump. The 1.0.0 batch (2026-08-19, see CHANGE_LEDGER.md) shipped the last
  gated set — `b_hobj`'s discarded scale factors, `vb_pose`'s compounding `init_poses` drift, the
  V-AUG range disagreement — so B-HOBJ / push-task VB-POSE / V-AUG numbers recorded before 1.0.0
  are not comparable and are being recomputed.
- Repeats deliberately share one seed stream (`sim_config.set_sim_config`, seed 1234); they are
  not reseeded per rollout.

## Layout (top level)

```
realm/
├── eval.py            single-env evaluation entry (SUPPORTED_TASKS / SUPPORTED_PERTURBATIONS
│                      are top-level literals here; tests AST-parse them — keep them literal)
├── vector_eval.py     N rollouts stepped together in one simulator, waves of num_envs
├── rollout.py         everything the two paths must agree on: metrics, RenderSchedule,
│                      Rollout, gripper conventions, result rows, artifact writing
├── sim_config.py      OmniGibson macros + seeding (before env build) and carb render modes
├── paths.py           run_log_dir(): the <root>/<experiment>/<model>[/<run_id>] convention
├── realm_logging.py   CSV reports, consolidated parquets, VideoRecorder
├── geometry.py        pure pose/rotation math (no omnigibson/torch — host-importable)
├── placement.py       collision-free placement (rejection sampling); place_within()
├── categories.py      object-category catalogue helpers
├── inference/         InferenceClient facade + one adapter per model type (client.py),
│                      DreamZero client, ZMQ base, extract_from_obs + ROBOT_OBS_PROFILES (utils)
├── environments/
│   ├── env_base.py            scoring reference + contact/grasp predicates
│   ├── task_progression.py    stage rubric checkers (TaskProgressionMixin)
│   ├── joint_reset.py         drawer joint reset, batched for vector envs
│   ├── env_config.py          OmniGibson config assembly from the YAML layers
│   ├── scene_setup.py         post-load scene/robot fixes (SceneSetupMixin)
│   ├── env_dynamic.py         RealmEnvironmentDynamic: construction + reset + step
│   ├── env_vector.py          RealmVectorEnvironment: N scenes, global sim ops done ONCE
│   ├── vec_init_queue.py      init-queue repair for object-replacing perturbations
│   └── perturbations/         one module per perturbation + registry.py + _helpers.py
├── robots/
│   ├── definitions/           RobotDefinition YAMLs (droid, droid_mounted, droid_robolab,
│   │                          droid_robolab_v2, ur) — OG 3.9.1 selects robots by `model`,
│   │                          there are no per-robot Python classes; WidowX uses stock vx300s
│   ├── controller_registry.py registers the four custom controllers + default configs
│   ├── droid_joint_controller.py / custom_joint_controller.py   joint PD (impedance / plain)
│   ├── droid_ee_controller.py   cartesian EE control; SUPPORTED_MODES = absolute_pose,
│   │                            pose_delta_ori — the other declared modes fail at construction
│   ├── droid_gripper_controller.py, gains.py
│   └── robot_ik/              dm_control/dm_robotics differential IK (normalised-velocity
│                              unit system — read robot_ik_solver.py's module docstring first)
└── config/                    tasks (REALM_DROID10, IMPACT, other), scenes, robots, objects, env

examples/   01_pi0_eval.py (hardcoded), 02_evaluate.py (the CLI), 03_vector_first_frames.py,
            04_vector_evaluate.py (vectorized CLI)
tests/      script-style tests + run_suite.py driver (see Testing below)
scripts/    clara/ (SLURM + lib/{common,server,apptainer}.sh), debug/ (hand-driven scripts),
            debug_probes/, karolina/, cluster_evals/, container launchers
docs/       code_archaeology.md (long-form evidence behind terse docstrings), CHANGE_LEDGER.md
            (root), vector_env/, perf/, evaluation_paths.md
wiki/       the operator docs: Quick-Start, Running-Evaluations, Robots-and-Configs,
            Running-the-Test-Suite, Cluster-and-Parallel-Runs, Known-Issues-and-Gotchas
```

## Evaluation pipeline

`realm/eval.py::evaluate()` (single env) and `realm/vector_eval.py::evaluate_vectorized()`
(waves of `num_envs`) write identical artifacts under a `realm.paths.run_log_dir()` directory:

```
<log_root>/<experiment_name>/<model_name>[/<run_id>]/
├── reports/{task}_{perturbation}.csv     rewritten after every rollout (resume reads it)
├── qpos/{task}.parquet                   one row per (perturbation, repeat)
├── actions/{task}.parquet
└── videos/{task}.parquet                 encoded mp4 bytes
```

Shared semantics live in `realm/rollout.py`; anything the two paths must agree on goes there,
never copy-adapted. Render-on-demand (default ON) renders only the steps whose observation feeds
inference; pass `--no-render_on_demand` when the recorded video matters.

## Inference clients

`realm/inference/client.py`: `InferenceClient(model_type, port, host)` dispatches to one adapter
per model type. **Registered:** `debug` (canned actions, no server), `openpi` (Pi0 family,
224x224 padded, websocket), `dreamzero` (320x180, requires `--multi-view` AND the robot-frame EE
pose). **Present but deliberately disabled:** `GR00T`, `GR00T_N16`, `molmoact` — re-enabling is
one line in `ADAPTERS`. Gripper conventions are keyed by the same strings in `realm/rollout.py`
(`GRIPPER_OPEN_ABOVE_HALF` / `_BELOW_HALF`); an unknown `model_type` raises rather than guessing.

Observations are keyed by `robot.name` (NOT always `"DROID"`), and the wrist-camera key is
resolved per robot through `ROBOT_OBS_PROFILES` in `realm/inference/utils.py` — update that table
when adding a robot asset. Actions are **absolute joint positions** (7) + gripper; models emit
gripper in (0,1), the environment expects (1,-1) with the 0.5 threshold.

## Perturbations

IDs 0–15: Default, V-AUG, V-VIEW, V-SC, V-LIGHT, S-PROP, S-LANG, S-MO, S-AFF, S-INT, B-HOBJ,
SB-NOUN, SB-VRB, VB-POSE, VB-MOBJ, VSB-NOBJ. Each is one module in
`realm/environments/perturbations/`, registered in `registry.PERTURBATION_FNS`, applied during
`reset()`, with a contract docstring stating what it mutates on the env.

Sim-state discipline: perturbations never call `og.sim.stop/play/step` directly — they use the
`_helpers.py` wrappers, which no-op/defer in a vector env so `RealmVectorEnvironment.reset()` can
do each global operation exactly once. Only add/remove/replace needs a stopped sim
(`NEEDS_STOPPED_SIM`); pose writes go through `vb_pose._place` on a live sim.

`task_type` is a closed namespace (put, pick, rotate, push, stack, open_drawer, close_drawer) —
pinned by two host-runnable tests: `tests/test_task_type_literals.py` (every compared literal is
declared) and `tests/test_perturbation_task_types.py` (SB-VRB's matrix/verb tables agree with the
configs). Run both after touching task types.

## Testing

Two tiers (see the Makefile header): tier 1 is container-free (`uv sync --locked`, then
`uv run make check`; expected GREEN); tier 2 needs the container/GPU and is driven by
`tests/run_suite.py` (`make test-smoke` / `test-suite` against a Slurm allocation). The tests are
**script-style with printed verdicts** — do NOT run `pytest tests/` (collection boots Isaac); the
four real pytest modules (`test_perturbation_task_types`, `test_cell_classification`,
`test_robot_base_column`, `test_robot_definition_parity` — all host-safe, ast/yaml based) are run
directly by filename. Exit codes are never
trusted: Isaac exits 0 on unhandled exceptions and segfaults at teardown on passing runs.

Changes made off-cluster (this machine has no GPU; OmniGibson cannot run locally) need their
container-side verification steps written down somewhere before they are forgotten. There was a
`TODO_CLARA.MD` at the root for this; it was deleted 2026-08-21 once its queue was empty (the
2026-08-19/20 sweep passed, the `DROID_robolab_v2` blocker was fixed and verified, and vectorized
SB-VRB came back 5/5). Recreate it, or start a fresh queue, when off-cluster work next accumulates —
an empty checklist in the tree is worse than none, because it reads as "nothing to verify". Durable
findings do NOT belong in that queue: they go to `docs/code_archaeology.md` (evidence),
`CHANGE_LEDGER.md` (a change and its revert), or `wiki/Known-Issues-and-Gotchas.md` (traps).

## Clusters

`scripts/clara/` submits everything; the three sourced libs in `scripts/clara/lib/` are the
extension points (new server type → `server.sh`; new bind/env → `apptainer.sh`). Required env:
`REALM_SIF`, `REALM_DATA_PATH`. `make put_clara` / `get_logs_clara` rsync code and logs. The
lighting fix is ON by default (`REALM_LIGHT_FIX=1`); OG-lite binds via `--og_lite` / MODE=oglite.
Details: wiki/Cluster-and-Parallel-Runs.md and wiki/Running-Evaluations.md.

## Developer workflows

**Add a task:** config YAML under `realm/config/tasks/REALM_DROID10/<task>/default.yaml`; stage
sequence in `task_progressions.yaml`; checkers in `task_progression.py` if a new stage type;
register in `eval.py::SUPPORTED_TASKS` (keep it a literal list). The task_type tests must pass.

**Add a perturbation:** one module in `perturbations/` taking the env; register in
`registry.PERTURBATION_FNS`; add to `eval.py::SUPPORTED_PERTURBATIONS`; use the `_helpers`
sim-state wrappers; write the what-it-mutates docstring; call `rebase_after_play` if it
adds/removes objects.

**Add a model:** one adapter class in `realm/inference/client.py` + an `ADAPTERS` entry; add the
model_type to the gripper-convention tables in `realm/rollout.py`; a new server launch branch
goes in `scripts/clara/lib/server.sh`.

**Add a robot:** definition YAML under `realm/robots/definitions/`; config under
`realm/config/robots/`; a `ROBOT_OBS_PROFILES` entry in `realm/inference/utils.py`; control
frequency in `sim_config.set_sim_config`.

## Conventions worth knowing before editing

- Positions written into object cfgs are **scene-frame** (vector-env scenes are tiled ~25 m
  apart; world-frame writes only look right in scene 0). `bounding_box` in a cfg is an EXTENT.
- Rubric dicts must be deep-copied per env (`TASK_PROGRESSIONS` is module-level and mutated).
- `og.sim.stop/play/step/render` are GLOBAL across scenes — batch them in vector paths.
- Docstrings carry contracts; long-form incident evidence lives in `docs/code_archaeology.md`
  and `CHANGE_LEDGER.md` — add new postmortems there, not as 40-line docstrings.
- `VERSION` at root is the release source of truth (`.github/workflows/release.yml` tags merges
  to main). The lint gate (`make lint`, F401/F811 only) and tier-1 tests are kept at zero/green.
