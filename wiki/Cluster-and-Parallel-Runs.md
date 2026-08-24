# Cluster and parallel runs

Two independent ways to get more throughput, which compose:

- **Vectorization** — N environments inside one process, sharing one simulator.
- **Sweeping** — one process per matrix cell, many processes across a cluster.

> **Only one cluster route currently works.** The `scripts/clara/interactive/` harness (`rr`, `go`,
> and the `sbatch_*.sh` launchers) is maintained and runs against the 3.9.1 image. The older
> pipelines under `scripts/cluster_evals/`, `scripts/karolina/` and `scripts/eval.sh` still activate
> a `micromamba` environment that does not exist in this image, and will fail on first contact. See
> [Known issues](Known-Issues-and-Gotchas).

## Interactive: hold an allocation, run against it

```sh
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
```

Check the GPU is actually idle before using it — an allocation is not a guarantee that nothing else
is resident:

```sh
srun --jobid=<ID> --overlap nvidia-smi \
     --query-compute-apps=pid,used_memory,name --format=csv
```

**`rr` starts the container wherever you invoke it — it does not `srun` onto the allocation.** Reach
it through one:

```sh
srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr python -u ...
```

`go` does that for you, and additionally tees output to a log, records the command and allocation,
and appends an explicit exit marker:

```sh
ALLOC=<jobid> ./scripts/clara/interactive/go <logname> ./scripts/clara/interactive/<script>.sh
```

`go` takes a **script file**, not an inline command string — multi-line commands passed through
`srun` have had their newlines collapsed into one mangled command that ran silently. It also refuses
if the allocation is not RUNNING, and if the script has lost its executable bit (which `sed -i`
does), because the resulting `srun` failure looks nothing like the real cause.

That explicit exit marker matters more than usual here — see below.

## Batch: `sbatch_eval_pi05.sh`

The maintained end-to-end launcher. It allocates, starts its own policy server on a port derived
from the job ID so parallel jobs do not collide, waits for the server, runs the evaluation, and then
**verifies that real artifacts were produced**.

```sh
VEC=4 PERT_ID=0 MAX_STEPS=800 REPEATS=25 RUN_ID=def_vec4 \
  sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
```

Configured entirely by environment variable. `VEC` is the path selector, not just a count:

- `VEC >= 1` → `examples/04_vector_evaluate.py --num_envs $VEC`
- `VEC = 0` → `examples/02_evaluate.py`

Other variables include `TASK_ID`, `PERT_ID`, `REPEATS`, `MAX_STEPS`, `HORIZON`, `ROBOT`,
`MODEL_NAME`, `EXPERIMENT`, `RUN_ID`, `PORT`, `SERVER_WAIT` and `CKPT`.

> **Check two defaults before using it.**
>
> - **`ROBOT` defaults to `DROID_mounted`**, not the `DROID` documented everywhere else. Pass
>   `ROBOT=DROID` unless you specifically want the RoboLab gripper — and note that if its definitions
>   are not registered, the job fails and **still exits 0**.
> - **It hard-requires the OG-lite fork** and aborts if it is absent, so it is not usable
>   unmodified outside a setup that has OG-lite checked out.
>
> Its preflight checks the image, dataset, OG-lite and checkpoint — but **on the compute node, after
> SLURM has accepted the job**, not before submission. They save you a wasted run, not a wasted
> queue wait.

### ⚠ Exit code 0 does not mean the run succeeded

Isaac's shutdown path hard-exits with status 0. An unhandled Python exception therefore still yields
a job SLURM records as `COMPLETED`, having written nothing. This has happened more than once in this
project — a job died on an assertion, produced no output, and was recorded as successful.

Two corollaries:

- **`atexit` and `finally` blocks do not run.** Do not put cleanup or result-writing there.
- **Exit code 139 (SIGSEGV) is equally uninformative** — Isaac can segfault at teardown *after* a
  verdict has already printed.

The only trustworthy gate is checking the artifacts. `scripts/clara/interactive/check_run.py` does
this two ways, and the second matters most here:

- given a results directory and `--repeats N`, it verifies that N rollouts actually landed
  (`--newer-than` guards against counting a previous run's files);
- given a log file as its optional positional argument, it **scans for crash markers** — which is how
  you catch the exception that a status of 0 hid.

Without `--repeats` it cannot know the expected count, so pass it.

> **The log argument only works on a log that carries an `### EXIT_CODE=` marker.** That marker is
> written by `go` and by `sbatch_eval_pi05.sh`, and by nothing else. Hand `check_run.py` a log you
> captured yourself with `srun ... > run.log` and its log scan reports
> `[warn] no EXIT_CODE marker found (run may still be in flight)` and the **whole verdict is FAIL**,
> however clean the run was and however complete the artifacts. Confirmed 2026-08-16 against a
> perfectly good `debug` run: artifacts `pass`, log `FAIL`, exit 1. This is deliberate — an unmarked
> log might be a truncated one; the release gate treats that as a failure — but it is
> indistinguishable from "your run was fine, you just did not use `go`".
>
> So: produce the log with `go`, or **omit the log argument** and check the artifacts alone, which
> gives a clean `VERDICT: PASS` and exit 0.

## Sweeping the matrix

`scripts/cluster_evals/run_evals_for_ckpt.sh` fans the 10 × 16 matrix out into one process per cell:

```sh
--task_ids 0,4,8 --perturbation_ids 3-7
```

Both accept comma-separated values and `a-b` ranges. **Omit either and you get all of it** — they
default to `0-9` and `0-15`. Cells whose outputs already exist are skipped, so the script is
re-runnable after a partial failure.

Ports are derived per cell so that concurrent jobs do not collide.

> This driver reads the task and perturbation lists out of `realm/eval.py` at runtime rather than
> keeping its own copy, so the IDs cannot drift from the code. But note the caveat at the top of this
> page about the older pipelines — check which launcher a sweep script ultimately calls before
> trusting it.

## Vectorization

`examples/04_vector_evaluate.py --num_envs N` builds N environments in one process and runs the
requested `--repeats` in waves of N.

**Choosing N.** On a single L40S, somewhere around **4 environments** is the useful operating point.
The project's own measurements disagree above that: one batch found 8 members meaningfully worse in
aggregate, another found 8 better under a more aggressive configuration, and 16 was never shown to be
economic. Treat 4 as the safe default and measure before going higher. Whatever you pick, say which
measurement you are relying on.

**Four perturbations need a stopped simulator:** `V-SC`, `VB-MOBJ`, `VSB-NOBJ` and `SB-VRB` — the
ones that add or remove objects. Stopping the simulator is a **global** operation, not a
per-environment one, so the vector environment handles it centrally: it checks whether any member of
the wave needs a stopped sim and, if so, batches **one** `stop()`/`play()` cycle around the whole
wave instead of cycling per member.

**So these do vectorize.** They are the expensive resets, not excluded ones. Everything else —
including `VB-POSE` and `V-VIEW` — only writes poses, works on a live sim, and deliberately never
triggers a cycle.

The validated OG391 SIF includes the scene z-offset and other OG-lite fixes, so vector environments
run in the default `MODE=stock`. Use `MODE=oglite` only to test a host fork checkout.

See also the vector-run caveat in [Logs and outputs](Logs-Outputs-and-Viewer): results recorded
before the per-environment rubric fix have invalid success rates.

## A SLURM detail worth copying rather than rediscovering

`sbatch` ships the script's *text* to a spool directory, so `${BASH_SOURCE[0]}` cannot be used to
locate sibling files like `lib/paths.sh`. The current launchers carry a short locator that tries
`BASH_SOURCE`, then the job's recorded command, then the submit directory. Copy that block into any
new launcher instead of reinventing it.

## See also

- [Running evaluations](Running-Evaluations)
- [Performance and scaling](Performance-and-Scaling)
- [Known issues and gotchas](Known-Issues-and-Gotchas)
