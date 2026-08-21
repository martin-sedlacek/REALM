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

## 1. RESOLVED 2026-08-21 — `--robot DROID_robolab_v2` was a DANGLING JOINT in the asset

`ValueError: 'panda_joint1' is not in list` (robot.py:2816, arm_control_idx). Cause, in
`droid_robolab_v2_mounted.usd`:

```
/panda/table/panda_table_joint   (PhysicsFixedJoint)
    physics:body0 -> /panda/droid_mounted/droid_mounted/base_link    MISSING
    physics:body1 -> /panda/droid_mounted/panda_link0                MISSING
```

No `/panda/droid_mounted` prim exists — both targets are leftovers from an earlier hierarchy, so the
joint bolted nothing to nothing and `table` floated free. `entity_prim.py:229` then derives a link name
from body1 with `pathString.split("/")[-1]`, a **basename**: the dangling path still ends in
`panda_link0`, so the arm's real root entered `joint_children` anyway, `valid_root_links = links -
joint_children` left only `table`, and `ArticulationView("/panda/table")` was built on a free-floating
body with no arm joints beneath it.

Two edits, BOTH required:
- `bf1e416` — removed a duplicate `PhysicsArticulationRootAPI`/`PhysxArticulationAPI` from
  `/panda/table` (with the joint repaired this would re-split the table into its own articulation).
- `b90febe` — repointed the joint: `body0 -> /panda/table`, `body1 -> /panda/panda_link0`.

Cumulative diff vs the pristine asset: **2 changed prim specs out of 1031**. No attribute, mass or pose
touched. Pristine copy is in git at `6154f19`.

Verified: smoke 193774, `--robot DROID_robolab_v2` (the default, no override), task 0 / VB-POSE / pi0.5
/ vec 4 → `DONE 2 rollouts in 731.5s -- SR=0.000 TP=0.100`, no construction error. Then the full
VB-POSE benchmark (193796-193825, 30 jobs) ran on the mounted asset; task 0 gives SR=0.800 (pi0),
0.800 (pi0-FAST), 0.680 (pi0.5), so the asset is sound under load.

The owner decision is closed too: `droid_robolab_v2` keeps pointing at the mounted asset, which now
builds. `DROID_robolab_v2_bare` remains available.

### Correcting the record on this item

The old entry's "Ruled OUT" paragraph was **accurate but aimed at the wrong quantity**, and that is why
it sent the investigation the wrong way for three cluster runs. "The mounted asset yields ONE clean
candidate (`table`)" is TRUE — `valid_root_links == ['table']`, measured again 2026-08-21. It is a
statement about root-LINK candidates. The `ArticulationRootAPI` count is a *different* quantity (the
asset had two: `/panda` and `/panda/table`), and on 2026-08-21 I briefly "retracted" the old line by
conflating the two. That retraction was itself wrong; the line was right.

"Every joint has both `body0` and `body1`" was also true — and was the trap. Both relationships were
*present*; nobody checked whether their targets **resolve**. Presence is not validity.

**Method that actually worked, use it first next time:** replicate the consuming code's inference
(`entity_prim.py:203-241`) offline with pxr on a login node — no Isaac, no GPU, seconds per asset. It
prints `valid_root_links`, `root_link_name`, dangling targets and link reachability directly, instead of
inferring them from USD topology. Kept as `scripts/debug_probes/inspect_articulation_roots.py`;
the two repair scripts are `fix_mounted_articulation_root.py` and `fix_mounted_table_joint.py`.
A host `usd-core` (`pip install usd-core`) is enough — the container is not needed.

Worth a cheap follow-up: `/panda/table` has `physics:mass=0.0` alongside `PhysicsRigidBodyAPI`.
Pre-existing, untouched by the fix, and it did not stop the benchmark — but a zero-mass dynamic rigid
body is odd and is the first suspect if the table ever behaves strangely.

## 2. Backlog (not this sweep)

- **Controller base-class extraction** is now UNBLOCKED — item 2 passed on the cluster. Wants a
  same-allocation `test-smoke` before and after when it lands.
- ~~**Vectorized SB-VRB remains UNVERIFIED end to end**~~ **VERIFIED 2026-08-21, 5/5 PASS.**
  `tests/test_vector_integrity.py --cells 0,2,4,6,7 :SB-VRB --num_envs 2 --repeats 2 --max_steps 5
  --robot DROID_robolab_v2`, on held alloc 193981 (l40s-03): put_green_block_into_bowl, rotate_marker,
  pick_spoon, stack_cubes, push_switch all `PASS: 2 rollouts, all artifacts present`; `0 refused,
  0 known-broken, 0 failed`. Log: `logs/todo_clara/sbvrb_vector.log`.
  Those five cells are one per distinct `task_type` SB-VRB reaches (put/rotate/pick/stack/push).
  `num_envs=2` is the cheapest count that can see the bug this item was about: scene 0's origin IS the
  world origin, so a scene-frame/world-frame confusion in the added receiver is invisible with 1 env.
  Tasks 8/9 excluded on purpose — SB-VRB is declared not-implemented there
  (`EXPECTED_NOT_IMPLEMENTED`), so refusal is correct and is statically pinned already.
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
  Confirmed still true 2026-08-21, and that invocation gives `tier 1 clean` (lint PASS + 37 pytest).
  Plain `make check` reports ruff/pytest missing and exits 1 — that is PATH, not a regression. Read this
  bullet before concluding "no ruff available on this machine".
- **`--out` accumulates across invocations.** Point `--out` somewhere per-sweep, or read the `*` marks.
