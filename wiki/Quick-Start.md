# Quick start

This page takes you from a working [installation](Installation) to a real evaluation, in steps
that each verify something before the next one depends on it.

Everything runs **inside the container**. The wrapper that puts you there is
`scripts/clara/interactive/rr`.

> **`scripts/clara/` is the development cluster's harness, not a portable installer.** The commands
> below are exactly what is run in practice, which makes them trustworthy — but the `salloc` line,
> the partition and GPU names, and the paths resolved by `scripts/clara/lib/paths.sh` are
> site-specific. Adapt them for your machine. See the note at the end of
> [Installation](Installation).

## 0. Hold an allocation

```sh
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
```

Note the job ID it prints — you need it for every command below.

> ### `rr` does not put you on the node. `srun` does.
>
> `rr` starts the container **wherever it is invoked**. It does not allocate and it does not `srun`.
> Run it bare on a login node and you get a container with no GPU, which fails in confusing ways
> rather than obviously.
>
> So every `rr` invocation goes **through `srun` onto the allocation you hold**:
>
> ```sh
> srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr python -u ...
> ```
>
> The repo's `go` wrapper does this for you and adds logging and an explicit exit marker:
> `ALLOC=<ID> ./scripts/clara/interactive/go <logname> <script> [args...]`. It runs a **script
> file**, not an inline command string — passing multi-line commands through `srun` has mangled them
> before, which is why `go` exists.
>
> The commands below show the `srun` form explicitly. Drop the prefix only if you already have an
> interactive shell on the node.

Check the GPU is actually free before using it — holding an allocation does not guarantee nothing
else is resident on the card:

```sh
srun --jobid=<ID> --overlap nvidia-smi \
     --query-compute-apps=pid,used_memory,name --format=csv
```

## 1. Check paths before anything else

```sh
bash -c 'source scripts/clara/lib/paths.sh; realm_paths_show'
```

The three that `rr` actually checks before it will start are `REALM_SIF`, `REALM_DATA` and
`REALM_LOGS`. **Those three must say `ok`**; if one is missing `rr` refuses anyway, and this just
tells you *which*, immediately, instead of after a container spin-up.

> **`REALM_APPDATA` reads `MISSING` on a fresh checkout, and that is fine.** It is
> `$REALM_ROOT/data/cache`, the per-checkout Kit/USD shader cache, and `rr` creates it on first run
> — it is an output, not a prerequisite. Verified 2026-08-16 on a clean worktree: seven lines `ok`,
> `REALM_APPDATA` `MISSING`, and step 2 below then ran fine and wrote all four artifacts. Do not go
> hunting for it.

`(cwd)` is the first line and is informational, not a path check.

## 2. A run that needs no policy server

The `debug` model type returns a constant action, so this exercises the whole simulation and logging
path without a network dependency. Keep it tiny.

```sh
srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 \
    --repeats 1 --max_steps 20 \
    --model_type debug --model_name debug --port 8000 \
    --experiment_name smoke --run_id first --log_dir /logs
```

`--port` is required even for `debug`, which never connects. `MODE` is not set here because
**`MODE=stock` is the default** — the image's own OmniGibson.

If that produced a run directory under `/logs/smoke/debug/first`, the install is good.

## 3. A real evaluation

Now you need a policy server. **REALM does not ship one** — it is a client. The server is a separate
process serving your policy over a websocket, and REALM only needs its host and port.

For π0.5 the repo has a launcher, `scripts/clara/interactive/pi05_server.sh`. Read it before running
it: it `cd`s into an **openpi checkout outside this repository** and defaults to a checkpoint under a
specific user's home directory, so it will exit immediately on any other machine. Treat it as a
worked example of the shape:

```sh
PORT=8000 CKPT=/path/to/your/checkpoint ./scripts/clara/interactive/pi05_server.sh
```

Serving π0.5 yourself means an openpi checkout, a checkpoint directory containing `params/`, and
`scripts/serve_policy.py` from openpi — not from REALM. It takes roughly 70 seconds to come up and
about 12 GB of VRAM at the memory fraction that launcher sets, which matters because the simulator
needs the rest of the card.

Wait for the server to be listening before starting the eval. Every batch launcher in the repo does a
socket preflight for exactly this reason: the client **blocks forever retrying** rather than failing,
so an eval against a dead port looks like a hang, not an error.

Single environment:

```sh
srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 \
    --repeats 25 --max_steps 800 --horizon 8 \
    --model_type openpi --model_name checkpoints_pi05_droid_jointpos \
    --host 127.0.0.1 --port 8000 \
    --experiment_name pi05 --run_id single --log_dir /logs
```

Vectorized — note this is a **different script** with a slightly different flag set:

```sh
srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr \
  python -u examples/04_vector_evaluate.py \
    --num_envs 4 --repeats 25 --max_steps 800 --horizon 8 \
    --task_id 0 --perturbation_id 0 \
    --model_type openpi --model_name checkpoints_pi05_droid_jointpos \
    --host 127.0.0.1 --port 8000 \
    --experiment_name pi05 --run_id vec --log_dir /logs
```

With `--num_envs 4 --repeats 25`, the 25 rollouts run in waves of 4.

> **Do not copy a single-env command line onto the vectorized script.** `examples/04_vector_evaluate.py`
> has no `--resume` and no `--no_render`; `examples/02_evaluate.py` has both. Their `--log_dir`
> defaults also differ. See [Running evaluations](Running-Evaluations).

> **Four perturbations need a stopped simulator** — `V-SC`, `VB-MOBJ`, `VSB-NOBJ` and `SB-VRB`,
> because they add or remove objects. They still run vectorized: the vector environment batches
> **one** stop/play cycle across the whole wave. They are simply the expensive ones to reset.

## Or just submit a batch job

`sbatch_eval_pi05.sh` does the whole thing — allocates, starts its own policy server on a
non-colliding port, waits for it, runs the eval, and then **checks that real artifacts were produced**
before reporting success:

```sh
VEC=4 PERT_ID=0 MAX_STEPS=800 REPEATS=25 RUN_ID=def_vec4 ROBOT=DROID \
  sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
```

It is configured entirely through environment variables, and `VEC` selects the path: `VEC>=1` runs
the vectorized script with that many environments, `VEC=0` runs the single-env script.

> **Two of its defaults will surprise you, which is why `ROBOT=DROID` is set explicitly above.**
>
> - **It defaults to `ROBOT=DROID_mounted`**, not the `DROID` that every flag table on this wiki
>   documents. That robot needs `scripts/install_robot_definitions.py` to have been run, and if it
>   has not, the job fails with `... is not a registered robot` — **and still exits 0, so SLURM
>   reports COMPLETED.**
> - **It hard-requires the OG-lite fork** and aborts if it is not found. OG-lite is not part of a
>   normal install, so this launcher is not a route an outside user can take unmodified.
>
> Its four preflight checks run **on the compute node, after SLURM has accepted the job** — they
> catch a misconfiguration early in the run, not before submission.

> **A SLURM exit code of 0 proves nothing here.** Isaac's shutdown call hard-exits with status 0, so
> an unhandled Python exception still produces a `COMPLETED` job that wrote no results. This has
> happened. Always check the artifacts — which is what that launcher's final gate does for you.

## Next

- [Tasks and perturbations](Tasks-and-Perturbations) — the 10 × 16 matrix
- [Running evaluations](Running-Evaluations) — every flag, and what `MODE` does
- [Cluster and parallel runs](Cluster-and-Parallel-Runs) — sweeping the matrix
- [Known issues and gotchas](Known-Issues-and-Gotchas) — read before debugging anything
