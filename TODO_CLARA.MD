# TODO_CLARA — container/GPU verification queue

Everything below verifies work done off-cluster (no local GPU; OmniGibson cannot run on the
laptop). Items are ordered so the cheap checks run first and gate the expensive ones. Tick items
off and delete them as they pass; a failure is worth pasting back verbatim (the verdict lines
plus the first traceback), not summarising.

Sync first: `make put_clara`.

---

## SWEEP OF 2026-08-19/20 IS DONE — all of items 1–5b passed

Covered `b9a6fb3..HEAD`: perturbations refactor, Phase-0 cleanup, robots fail-fast + IK docs,
inference-client adapters, logging dedup, paths consolidation, the drawer-fix merge, and the 1.0.0
number-moving batch. Run on allocation 193102 (**h200-02**, not l40s — l40s had all 5 usable nodes at
8/8 GPUs, `l40s-06` draining and 240 jobs queued), `MODE=oglite` throughout.

| item | what | result |
|---|---|---|
| 1 | login node: task-type literals, rubrics, `run_suite --only local`, 4 host pytest modules | PASS (37 pytest passed — 37 not 36+skip, because `test_against_the_real_captured_log`'s log exists here) |
| 2 | `tests/test_robot_ik_units.py` in-container, first-ever run | PASS, 13 checks incl. `RobotArm.name` |
| 3 | `test_integrity` (10 tasks, 30 cells) | PASS |
| 3 | `test_perturbations_integrity` (16 perturbations, 48 cells) | FAIL on B-HOBJ → fixed in `db3a4bc` → **re-run PASS, 48/48** |
| 5 | EE-control fail-fast, `--robot DROID_ee_control` | PASS — `absolute_pose` constructs, effector prefix is `franka` not a bound method |
| 5b | `t10_bhobj_props` task 0 (6 resets) + task 8 joint path (4 resets) | PASS both — no compounding, not frozen |
| 5b | VB-POSE push non-accumulation | PASS — `init_poses` drift exactly 0.0 across 4 resets |

Item 4 (SB-VRB drawer refusal) was left unrun: it is statically pinned by
`tests/test_perturbation_task_types.py` and upstream already exercised the cells live. Item 6 (A/B
number-neutrality) was left unrun: superseded by the offline equivalence harness, and B-HOBJ /
push-VB-POSE / V-AUG now differ by design since 1.0.0 so a diff there proves nothing.

**One bug found, fixed and re-verified (`db3a4bc`):** `b_hobj.py` wrote the scaled mass without a
`float()` cast. `min(np.float64, MASS_CLIP_KG)` returns whichever OPERAND is smaller keeping its type,
so below the 2 kg clip an np.float64 reached `set_masses` and raised
`RuntimeError: Index put requires the source and destination dtypes match` then SIGSEGV — i.e. the
dtype of the write depended on the draw. The joint setters three lines below already had the cast with
a comment naming the hazard. No number moves (Isaac stores masses float32 either way). Only reachable
against a live Isaac, so no host-side check could have caught it.

**Lesson worth keeping:** `run_suite`'s table used to print merged rows from earlier invocations
against the same `--out` indistinguishably, under a bare `PASS=n`. `make check` (2 container-free
tests) therefore printed `PASS=9` including a stale `test_perturbations_integrity PASS` that
contradicted a FAIL measured on the same tree 20 minutes earlier. Gating was never wrong (`mine`
decides status and the XML); the *printed* line was. Merged rows are now marked `*` with split counts.

---

## 1. BLOCKER — `--robot DROID_robolab_v2` cannot construct a robot

Found 2026-08-19 while running item 5b. **This is not the vector-path issue previously recorded; that
attribution was wrong.**

```
ValueError: 'panda_joint1' is not in list      (robot.py:2816, arm_control_idx)
```

The robot's articulation comes up without its arm joints. `DROID_robolab_v2` resolves to
`model: droid_robolab_v2` → `droid_robolab_v2_mounted.usd`, and that asset fails on the **single-env**
path too — measured with `t10_bhobj_props.py`, which builds a plain `RealmEnvironmentDynamic` with no
vector env and defaults to this robot.

Why it looked vector-specific before: every run cited as evidence that single-env was fine
(`t4_default_check.sh`, and all 78 cells of item 3) silently used `--robot DROID`, the **stock** robot,
because that is the default in `examples/02_evaluate.py` and in `DEFAULT_ROBOT` in all three integrity
drivers. So `in_vec_env=True` was never the discriminator.

Practically: `sbatch_eval_pi05.sh` and the interactive probes default to `--robot DROID_robolab_v2`,
which currently cannot build. **`--robot DROID_robolab_v2_bare` works** and is the escape hatch.

Ruled OUT by measurement (`logs/mountedasset/rootlink.log`) — do not re-tread:
`entity_prim.py:241`'s `else "base_link"` root-link fallback. The mounted asset yields ONE clean
candidate (`table`) and every joint in it has both `body0` and `body1`, so the fallback never fires for
it. The asset that DOES depend on that fallback is stock `droid_mounted.usd`, and it works.

Next check: does `ArticulationView(f"{prim_path}/table")` (`entity_prim.py:141`) resolve to the
articulation `PhysicsArticulationRootAPI` is authored on? `table` is a root link no config names
(`arm_link_names` lists `panda_link0..7`).

Owner decision also open: whether `droid_robolab_v2` should keep pointing at the mounted asset while it
cannot build, or the two model names should swap so the default is the one that works.

## 2. Backlog (not this sweep)

- **Controller base-class extraction** is now UNBLOCKED — item 2 passed on the cluster. Wants a
  same-allocation `test-smoke` before and after when it lands.
- **Vectorized SB-VRB remains UNVERIFIED end to end** (adds a "receiver" other members' scenes lack —
  a case V-SC never exercises). `python tests/test_vector_integrity.py` perturbations entry, or the t9
  interactive probe with `--perturbation SB-VRB`.
- The 2-drawer + `target_drawer_loc="bottom"` case now raises a clear ValueError instead of
  UnboundLocalError — only relevant if a 2-drawer cabinet asset ever enters the drawer tasks.

## 3. Gotchas for whoever runs this next

- **VB-POSE logs nothing** — no `print`, no `og.log`. The old instruction "check the log: each reset's
  switch position within ±0.075/±0.15 m of the same base pose" cannot be followed. Use
  `logs/todo_clara/vbpose_switch_probe.py`, which asserts the invariant the 1.0.0 `.clone()` fix
  actually changed (`env.init_poses` bit-identical across resets) plus a variance check so it cannot
  pass vacuously on a frozen offset.
- **`rc=139` means nothing.** Isaac segfaults at teardown after passing work. Both t10 probes exited
  139 having printed `PASSED`. Read the verdict lines, never the exit code — that is why
  `run_suite` records `exit_code` but never gates on it.
- **`ruff` is in neither the login python nor the container.** It lives in the `mlspaces` and
  `molmoact_dev` conda envs:
  `make check RUFF=/home/sedlam56/miniconda3/envs/mlspaces/bin/ruff PYTHON=/home/sedlam56/miniconda3/envs/behavior/bin/python`
- **`--out` accumulates across invocations.** Point `--out` somewhere per-sweep, or read the `*` marks.
