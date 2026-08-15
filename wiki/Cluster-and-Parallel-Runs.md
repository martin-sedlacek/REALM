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

Then `rr` runs inside the container on that allocation, and `go` wraps a script with logging:

```sh
ALLOC=<jobid> ./scripts/clara/interactive/go <logname> ./scripts/clara/interactive/<script>.sh
```

`go` tees output to a log and appends the exit code, which matters more than usual here — see below.

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
`MODEL_NAME`, `EXPERIMENT`, `RUN_ID`, `PORT`, `SERVER_WAIT` and `CKPT`. It preflights the image, the
dataset and the checkpoint before submitting work.

### ⚠ Exit code 0 does not mean the run succeeded

Isaac's shutdown path hard-exits with status 0. An unhandled Python exception therefore still yields
a job SLURM records as `COMPLETED`, having written nothing. This has happened more than once in this
project — a job died on an assertion, produced no output, and was recorded as successful.

Two corollaries:

- **`atexit` and `finally` blocks do not run.** Do not put cleanup or result-writing there.
- **Exit code 139 (SIGSEGV) is equally uninformative** — Isaac can segfault at teardown *after* a
  verdict has already printed.

The only trustworthy gate is checking the artifacts. `scripts/clara/interactive/check_run.py` does
this: given a results directory it verifies the expected number of rollouts landed, and
`--newer-than` guards against counting a previous run's files.

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

**Four perturbations are not safe vectorized:** `VB-POSE`, `VB-MOBJ`, `VSB-NOBJ` and `SB-VRB`. They
stop and restart the simulator, which is a **global** operation, not a per-environment one. Run those
with `VEC=0`.

**Vector environments historically required `MODE=oglite`**, because the scene z-offset fix lived
only in the fork. That fix is now in both build recipes, so a rebuilt image — or `MODE=stockfix` with
a current patch directory — should remove the requirement. **A rebuilt image has never been
verified**, and the vectorized script's own docstring still asserts the OG-lite requirement. Until
someone verifies it, `MODE=oglite` is the conservative choice.

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
