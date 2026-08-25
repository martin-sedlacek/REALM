# Known issues and gotchas

Two kinds of thing here: **known limitations** that affect what your results mean, and **traps** that
will cost you an afternoon if you meet them cold. Everything on this page is recorded because it
already happened to someone.

Items marked **known and accepted** are deliberate calls about scope, not oversights. They are
documented so you can decide whether they affect your use, not as a to-do list.

---

## Limitations that affect results

### Two tasks have unusable camera views — *observed, parked, not diagnosed*

**Task 6 (`stack_cubes`) renders essentially nothing but sky** — its spawn region sits roughly a
metre beyond the nearest wall, putting the camera outside the room.

**Task 2 (`rotate_marker`) also gives an unusable external view, but not for the same reason.** Its
spawn region is *inside* the wall envelope and the scene does render an interior; the region is
simply short of any floor surface.

The dangerous part is that **every artifact and metric check still passes** on both. The run
completes, videos are written, reports are produced, nothing warns. A vision-conditioned policy
evaluated on task 6 in this state is being scored on pictures of the sky.

**Look at the frames before reporting numbers from tasks 2 or 6.**

Deliberately *not* labelled accepted: the project's own note says these may simply differ from the
pre-port configuration rather than be a port bug, and asks for them to be eyeballed in the GUI before
anything is changed. That check has not happened, so the cause is unconfirmed.

### `SB-NOUN` degenerates on some tasks — *known and accepted*

`SB-NOUN` re-targets the instruction at a different object already in the scene. On some tasks it
frequently re-draws the *original* object, giving a degenerate no-op instruction — around **16% on
task 0 and 66% on task 6**.

**Treat those two figures as indicative, not as measured rates.** Each is a single 25-rollout chain
and the draws within a chain are not independent, since the original object can be re-drawn. The
uncertainty is wide and the two are not cleanly comparable. A separate note in the same project puts
the general no-op rate at "~1/5 of resets"; the gap is sample size, not a contradiction.

Where it degenerates, the perturbation is weaker than its name suggests. Factor that into any
per-perturbation comparison rather than reading `SB-NOUN` as uniformly hard.

### `V-SC` is inert on three tasks, and over-subscribed on the rest

Two separate things, both worth knowing before averaging `V-SC` across tasks.

**It does nothing on `push_switch`, `open_drawer` and `close_drawer`.** `V-SC` re-places and
re-models the distractors a task *already declares*; it does not spawn new ones. Those three tasks
declare **zero** distractors, so there is nothing to clutter with — while the perturbation still
pays for a full stopped-simulator reset. A `V-SC` average over all ten tasks is averaging in three
near-no-ops. *(Read from the task configs and `v_sc.py`; not measured at runtime.)*

**Where it does act, the spawn region is over-subscribed** — *known and accepted*. On the task with
the most distractors, roughly **two objects per environment per reset** fail collision-free placement
after the attempt budget is exhausted and are dropped in from above. So `V-SC` carries some
falling-object dynamics that are not really part of its intent.

Older documentation describing `V-SC` as "spawns 3 random distractors" is wrong on the count, on the
per-task variation, and on the mechanism.

### Vector results before the rubric fix have invalid success rates

Environments in a wave shared one progression dictionary, making progression an **OR across
members**. The tell is identical `task_progression_timestamps` across members. See
[Logs and outputs](Logging). A success rate of 0.960 over 25 rollouts, from that
period, is **explicitly retracted**.

### Reset got slower on the port

Moving from OmniGibson 1.1.1 to 3.9.1 made rollout stepping substantially faster but made **reset
2.2–3.2× slower**. If your workload is many short rollouts, that regression can dominate. See
[Performance and scaling](Performance-and-Scaling).

### Videos are sparse unless you ask otherwise

`--render_on_demand` defaults **on**, giving roughly one recorded frame per action chunk. See
[Running evaluations](Running-Evaluations).

---

## Documentation and scripts that are stale

The repo has two eras: pre-port (OmniGibson 1.1.1) and post-port (3.9.1). Several user-facing
artifacts were never updated.

### The setup dataset path remains stale

`./setup.sh --docker --dataset` builds from `.docker/realm.Dockerfile`, and
`./setup.sh --apptainer` from `.docker/realm.def`. Both recipes use `.docker/constraints.txt`.
The container recipe names are current, but `setup.sh` downloads the dataset through a
`micromamba` environment that does not exist in the 3.9.1 image, and writes path variables into your
`~/.bashrc`.

Use [Installation](Installation) instead.

### `scripts/download_dataset.sh` does not exist

Two scripts (`scripts/run_apptainer.sh`, `scripts/eval.sh`) print instructions to run it. It is not
in the repo.

### `scripts/eval.sh` invokes a CLI that does not exist

It runs `python -u realm/eval.py` with flags. `realm/eval.py` is a library module with no `__main__`
and no argument parser. Some of the flags it passes (`--model`, a value-taking `--multi-view`) do not
exist on any parser in the repo.

### Other stale flags and names

- `--spp` and `--og_lite` are passed by some scripts and exist on **no** Python parser; argparse
  aborts on them.
- `--model_type` values `pi0`, `pi0_FAST` and `hamster` appear in older docs. Only
  `openpi`, `dreamzero` and `debug` construct.
- `--robot UR5_aligned` appears in older docs. The real config is `UR5_aligned_pd_control`.
- The repo README's task table spells task 0 `put_green_block_in_bowl`; the real identifier is
  `put_green_block_into_bowl`.
- The README roadmap lists vectorized environments as not done. They shipped.
- The second camera on drawer tasks is gated on a task type string that no task config ever sets, so
  that branch is unreachable. Treat it as a latent bug, not a mechanism.

### Build the container off Lustre

`apptainer build --fakeroot` fails on Lustre trying to change ownership inside the image rootfs.
Build on local disk and move the resulting validated SIF to the cluster.

---

## Traps

Each of these has cost real time.

### Exit code 0 proves nothing

Isaac's shutdown hard-exits with status 0, so an unhandled exception yields a `COMPLETED` job that
wrote nothing. **`atexit` and `finally` blocks never run.** Exit 139 (SIGSEGV) is equally
uninformative — Isaac can segfault at teardown after a verdict has printed. Check artifacts, not
status.

### `bash -lc` inside the container breaks the environment

Apptainer binds your home directory, so a *login* shell re-sources the host `~/.bashrc`, which can
prepend a host Python to `PATH` and shadow the container's conda environment. The symptom is
`ModuleNotFoundError: No module named 'omnigibson'`. Use `bash -c`, or call `python` directly.

### `apptainer shell` and `exec` skip the runscript

The runscript is what activates the conda environment. Use `apptainer run`.

### Undefined macros are truthy

In the stock image, reading an undefined `gm` macro returns a truthy object rather than raising or
returning `None`. So `getattr(gm, "SOMETHING", False)` checks are meaningless — they pass whether or
not the macro exists. Inspect the source instead.

### `og.log.info()` is invisible

The simulator pins the root logger to `WARNING` during setup. Info-level logging from your own code
will silently vanish.

### A trailing comma in YAML makes a truthy string

`use_cc_compensation: False,` parses as the **string** `"False,"`, which is truthy. This has silently
enabled a feature that was believed off. Check for stray commas when a boolean config seems ignored.

### Numba's cache needs a writable directory

OmniGibson JIT-compiles with caching enabled, which writes next to the source. The image is
read-only, so `NUMBA_CACHE_DIR` must point somewhere writable that survives — the harness binds it
under the log filesystem. The failure is `cannot cache function '_quat_multiply': no locator
available` at import time, and it does not reproduce under Docker.

### `VB-POSE` logs nothing at all

No `print`, no `og.log`. Instructions of the form "check the log: each reset's switch position is
within ±0.075/±0.15 m of the same base pose" cannot be followed, because there is no such log line to
check. Verify it by asserting on state instead — `env.init_poses` bit-identical across resets is the
invariant the 1.0.0 `.clone()` fix actually changed — and pair that with a variance check, or the
assertion passes vacuously on a frozen offset.

### Startup dominates short runs

Roughly **64% of wall time** on a small evaluation is startup, not rollout. Comparing wall-clock time
between configurations mostly compares startup. Compare stepping time, and amortise by doing more
repeats per process rather than more processes.

## See also

- [Installation](Installation)
- [Running evaluations](Running-Evaluations)
- [Performance and scaling](Performance-and-Scaling)
