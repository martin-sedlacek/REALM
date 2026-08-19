# TODO_CLARA — container/GPU verification queue

Everything below verifies work done off-cluster (no local GPU; OmniGibson cannot run on the
laptop). Items are ordered so the cheap checks run first and gate the expensive ones. Tick items
off and delete them as they pass; a failure is worth pasting back verbatim (the verdict lines
plus the first traceback), not summarising.

Covers commits `b9a6fb3..HEAD` on `port-to-og391`: perturbations refactor, Phase-0 cleanup,
robots fail-fast + IK docs, inference-client adapters, logging dedup, paths consolidation, and
the merge of origin's drawer-fix series (`3bc4c75` — SB-VRB now deliberately refuses the drawer
tasks; the refusal block merged cleanly into the decomposed sb_vrb.py). All of it was verified
offline with old-vs-new equivalence harnesses (bit-identical outputs incl. RNG stream state), so
the runs below are confirmation sweeps, not first contact — anything red is a real finding.

Sync first: `make put_clara`.

## 1. Login node — no GPU, no container (~5 s total)

```bash
python3 tests/test_task_type_literals.py          # expect: PASSED
python3 tests/test_task_progression_rubrics.py    # expect: PASSED (was committed-red; now green)
python3 tests/run_suite.py --only local --strict  # expect: both entries PASS
# pytest is absent from the login python; run the 4 host pytest modules wherever pytest exists
# (`make check` does this automatically when pytest is importable):
pytest tests/test_perturbation_task_types.py tests/test_cell_classification.py \
       tests/test_robot_base_column.py tests/test_robot_definition_parity.py -q
# expect: 36 passed, 1 skipped -- already confirmed on the laptop post-merge, 2026-08-19
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

## 4. SB-VRB on the drawer tasks — confirm the refusal survived the merge (~15 min, optional)

The policy REVERSED upstream while the refactor was in flight: `eef7f14` makes SB-VRB refuse the
drawer tasks outright (NotImplementedError before any env mutation — the receiver it would inject
has nowhere to go and was being dropped from the sky), and `e622b72`/`fa28d94` teach the matrix
harness that this refusal is not a crash. Upstream already exercised these cells live; what the
merge added is my decomposed sb_vrb.py carrying the refusal block, so the only question is
whether the merged file still raises FIRST — statically it does (the guard is the first statement
of `sb_vrb()`), and `pytest tests/test_perturbation_task_types.py` pins the UNSUPPORTED set.
Only if you want a live confirmation:

```bash
sbatch scripts/clara/run_debug_single.sh --task_id 8 --perturbation_id 12 --model_type debug --experiment_name sbvrb_drawer_refusal
```

Check the log for the loud `NotImplementedError: SB-VRB does not support task_type 'open_drawer'`
(with the "do not 'fix' this by making it a no-op" text), raised BEFORE any placement/instruction
lines — and no `Failed to place object 'receiver'` anywhere.

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
