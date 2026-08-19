# TODO_CLARA — container/GPU verification queue

Everything below verifies work done off-cluster (no local GPU; OmniGibson cannot run on the
laptop). Items are ordered so the cheap checks run first and gate the expensive ones. Tick items
off and delete them as they pass; a failure is worth pasting back verbatim (the verdict lines
plus the first traceback), not summarising.

Covers commits `b9a6fb3..HEAD` on `port-to-og391`: perturbations refactor, Phase-0 cleanup,
robots fail-fast + IK docs, inference-client adapters, logging dedup, paths consolidation. All of
it was verified offline with old-vs-new equivalence harnesses (bit-identical outputs incl. RNG
stream state), so the runs below are confirmation sweeps, not first contact — anything red is a
real finding.

Sync first: `make put_clara`.

## 1. Login node — no GPU, no container (~5 s total)

```bash
python3 tests/test_task_type_literals.py          # expect: PASSED
python3 tests/test_task_progression_rubrics.py    # expect: PASSED (was committed-red; now green)
python3 tests/run_suite.py --only local --strict  # expect: both entries PASS
# pytest is absent from the login python; run the task-types test wherever pytest exists:
pytest tests/test_perturbation_task_types.py -q   # expect: 7 passed
```

## 2. Container, CPU is fine — the new IK/gains unit test (~1 min)

New file, first-ever run — if it fails, suspect the TEST before the solver, and paste the
full output back so it can be iterated.

```bash
# inside the container (rr, or apptainer_eval on an allocation):
python -u tests/test_robot_ik_units.py
# expect: "PASSED -- IK conversion algebra and gain preparation behave as pinned"
```

This pins the solver's conversion algebra BEFORE the planned controller base-class extraction.
Once it passes on the cluster, say so — the robots dedup is parked on exactly this.

## 3. GPU smoke — the perturbations refactor (the main event, ~30–60 min)

```bash
sbatch scripts/clara/tests/run_pert_integrity_test.sh   # all 16 perturbations on task 0
sbatch scripts/clara/tests/run_integrity_test.sh        # tasks 0–7 under Default
# or equivalently, against a held allocation:
ALLOC=<jobid> make test-smoke
```

Expected: every cell PASS with the exact-row artifact checks. The perturbation modules were
refactored wholesale (sb_vrb/v_sc/vb_pose/placement decomposed, helpers extracted) with RNG order
preserved — a placement or instruction that differs from a pre-refactor run on the same seed
would falsify the offline equivalence and is a stop-the-line finding.

## 4. SB-VRB on the drawer tasks — NEW BEHAVIOR, eyeball it (~15 min)

Upstream fix `74f19b0` (merged into the refactor) makes SB-VRB actually fire on tasks 8/9 for the
first time — before it, the matrix keys never matched and the cells silently no-op'd. So this is
not a regression check; it is the first look at live behavior:

```bash
sbatch scripts/clara/run_debug_single.sh --task_id 8 --perturbation_id 12 --model_type debug --experiment_name sbvrb_drawer_probe
sbatch scripts/clara/run_debug_single.sh --task_id 9 --perturbation_id 12 --model_type debug --experiment_name sbvrb_drawer_probe
```

Check the log for: `task_type` flipping open_drawer<->close_drawer, a "New instruction: open/close
the ..." line (must NOT read "open drawer the ..."), no KeyError from the COMPATIBILITY_MATRIX
guard, and the rollout scoring against the flipped rubric.

## 5. EE-control smoke — the fail-fast must not reject a valid config (~15 min)

Construction now rejects the four broken IK modes; `absolute_pose` and `pose_delta_ori` must
still construct and run. Also confirms the `RobotArm.name` property fix (effector prefixes were
`<bound method ...>` strings before — cosmetic, but this is the first run with it corrected).

```bash
sbatch scripts/clara/run_debug_single.sh --task_id 0 --perturbation_id 0 --model_type debug \
    --robot DROID_ee_control --experiment_name ee_failfast_probe
```

Expect a normal debug rollout (the debug adapter returns a fixed EE pose). An
`AssertionError: Unsupported ik mode ...` here would mean a config names a mode I mis-classified
— paste the config name and the assert text.

## 6. Optional — direct A/B number-neutrality proof (~30 min)

Only if you want belt-and-braces on top of the offline equivalence: same seed, same task, debug
model, pre- vs post-refactor, diff the reports.

```bash
git worktree add ../REALM_pre 643d7c9        # last commit before the refactor commits
# run the SAME command in both checkouts (debug model, 2 repeats, any task/pert pair you like):
#   python examples/02_evaluate.py --task_id 0 --perturbation_id 13 --repeats 2 --max_steps 100 \
#       --model_type debug --model_name debug --port 1 --experiment_name ab_check --log_dir <distinct dirs>
diff <old>/reports/put_green_block_into_bowl_VB-POSE.csv <new>/reports/put_green_block_into_bowl_VB-POSE.csv
# expect: identical rows. VB-POSE/V-SC/SB-VRB are the perturbations worth A/B-ing (placement RNG).
git worktree remove ../REALM_pre
```

## 7. Backlog (not this sweep)

- **Vectorized SB-VRB remains UNVERIFIED end to end** (adds a "receiver" other members' scenes
  lack — a case V-SC never exercises). When vector work resumes:
  `python tests/test_vector_integrity.py` perturbations entry, or the t9 interactive probe with
  `--perturbation SB-VRB`.
- **Controller base-class extraction** starts only after item 2 passes on the cluster, and wants
  a same-alloc `test-smoke` before/after when it lands.
- The 2-drawer + `target_drawer_loc="bottom"` case now raises a clear ValueError instead of
  UnboundLocalError — only relevant if a 2-drawer cabinet asset ever enters the drawer tasks.
