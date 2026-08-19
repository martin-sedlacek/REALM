# Code archaeology: the long-form evidence behind terse docstrings

Dated measurements and dead-branch histories that used to live as 20-40-line docstrings on
one-line functions. The code keeps a short contract plus a pointer here; this file keeps the
receipts. Each section is referenced from exactly one place in the code.

## Second-camera selection for drawer tasks (`realm/rollout.py::wants_base_im_second`)

The DROID policies take one exterior view, so one of REALM's two exterior cameras has to be
picked, and the code has always *intended* to pick the second one for the drawer tasks. It never
did, and why it should is not recoverable:

* The comparison read `task_type == "open_close_drawer"` -- a string no task config declares, in
  this tree or the pre-port 1.1.1 one. `git log -S` in the 1.1.1 repo puts that string in
  `6e895fd init`, the project's first commit, and finds no `task_type: "open_close_drawer"` ever:
  it is not a rename casualty, it simply never matched anything. So the second camera was NEVER
  selected, for any task, in any run.
* `base_im_second` is None on every run without `--multi-view`, and the openpi path does
  `resize_with_pad(img_to_use, 224, 224)` on the selection -- so fixing the string alone would
  have turned a silent no-op into a TypeError on exactly the two tasks it was meant to help.
  Hence the live predicate also requires `base_im_second is not None`.

THE INTENT IS CLEAR, THE JUSTIFICATION IS NOT (2026-08-16). Nothing in the repo says why a drawer
task wants the second view, and the camera configs do not obviously support it: `open_drawer` is
the one task whose cam1 is a hand-picked pose (`CP3`) rather than an episode extrinsic, with cam2
left at `default`, so sending the policy cam2 moves it OFF the pose someone chose for this task;
`close_drawer` uses the same standard `ep_001042_cam1/cam2` pair as `put_green_block_into_bowl`
and is not distinctive at all. Selecting camera 2 is the defensible reading of a dead branch, but
it is not a measured improvement and has never been run.

Scope, checked rather than assumed: only the molmoact and openpi paths of `InferenceClient`
consult `use_base_im_second` at all -- dreamzero asserts both images and sends both regardless.
The only launchers that pass `--multi-view` unconditionally (`scripts/parallel_sweep_launcher.sh`,
`scripts/dreamzero_sweep_driver.sh`) both run dreamzero, so no harness path that exists today
changes behaviour; this bites the first time someone runs openpi with `--multi-view` on a drawer
task.

## The finger-closure threshold (`realm/environments/env_base.py::_finger_closure_threshold`)

The original grasp test compared a bare literal `0.45` against the raw finger joint value -- no
units, no normalisation by joint range -- so what it meant depended entirely on the asset:

* On `droid.usd` the finger joints are PRISMATIC in metres over [0, 0.05]; 0.45 is 9x the entire
  travel and the test is VACUOUSLY TRUE.
* On the robolab 2F-85 the same proprio indices are REVOLUTE in radians over [0, 0.7854]; 0.45
  lands mid-range and the test becomes "less than ~57% closed" -- which a real grasp violates.
  Measured 2026-08-11 (job 189066): with both pads on the block and the block lifted,
  finger_joint sits at 0.507-0.528, so this rejected 78/78 genuine grasp steps and the asset
  could never score a GRASP. Because recompute_task_progression breaks at the first unmet stage,
  that also froze LIFT/MOVE/PLACE on rollouts that visibly completed the task.

The live code scales by the robot's own open->closed range: `open + 9.0 * (closed - open)`. That
reproduces 0.45 EXACTLY for droid.usd (9 * 0.05), so every historical result is bit-identical;
for robolab it becomes 7.07 rad and the test is vacuous there too, matching the behaviour the
stock asset has always had.

NOTE: 0.45 is very likely a typo for 0.045, i.e. "the fingers stopped short of full closure, so
an object is between them" (90% of droid.usd's travel), which would be a meaningful test rather
than a no-op. Deliberately NOT adopted: it would make the guard bite on droid.usd for the first
time and could move every historical REALM number. Decide that separately, with a measurement.

## Concurrent suite runs colliding on the log tree (`tests/_paths.py::scratch_log_root`)

Two `tests/run_suite.py` invocations running at the same time (typically one per allocation, to
compare two checkouts) write into the SAME `/logs/<name>` tree: the name is fixed per test, with
no discriminator. The parquets are appended to, so the run that finishes second sees both runs'
rows and `check_artifacts` reports `FAIL_ROWS(2!=1)`.

Measured 2026-08-16: a before/after comparison run concurrently on jobs 191494 and 191495 had its
`test_single_task_drawer` cell PASS on the tree that finished first (11:09:06) and FAIL on the
tree that finished second (11:10:18), with identical code on that path. It reads exactly like a
regression in the second tree and is not one. Do not "fix" this by relaxing the row count: the
exact-rows check is what made the collision visible at all.

The `/app/logs` half of the same docstring: in this checkout `logs` is a SYMLINK to an absolute
host path; `scripts/clara/interactive/rr` binds the checkout at /app and the log tree at /logs,
and does not bind the symlink's target -- so `/app/logs` resolves to nothing and the first
`os.makedirs()` dies with FileNotFoundError before a single task is evaluated (measured
2026-08-16). Under the retired `scripts/run_docker.sh` (`-v $(pwd):/app`), `logs` was a real
directory and `/app/logs` worked, which is why older code used it.
