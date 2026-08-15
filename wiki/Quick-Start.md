# Quick start

This page takes you from a working [installation](Installation) to a real evaluation, in four steps
that each verify something before the next one depends on it.

Everything runs **inside the container**. The wrapper that puts you there is
`scripts/clara/interactive/rr`.

## 0. Hold an allocation

`rr` runs on an allocation you already hold — it does not allocate for you.

```sh
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
```

## 1. Check paths before anything else

```sh
bash -c 'source scripts/clara/lib/paths.sh; realm_paths_show'
```

Every line should say `ok`. If the image, dataset or log directory is missing, `rr` will refuse to
start anyway — this just tells you *which* one, immediately, instead of after a container spin-up.

## 2. A run that needs no policy server

The `debug` model type returns a constant action, so this exercises the whole simulation and logging
path without a network dependency. Keep it tiny.

```sh
./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 \
    --repeats 1 --max_steps 20 \
    --model_type debug --model_name debug --port 8000 \
    --experiment_name smoke --run_id first --log_dir /logs
```

`--port` is required even for `debug`, which never connects. `MODE` is not set here because
**`MODE=stock` is the default** — the image's own OmniGibson.

If that produced a run directory under `/logs/smoke/debug/first`, the install is good.

## 3. A vectorized smoke test

Before running a real vectorized evaluation, check that N environments build and render:

```sh
./scripts/clara/interactive/rr \
  python -u examples/03_vector_first_frames.py \
    --num_envs 4 --task_id 0 --out_dir /logs/vector_first_frames
```

This steps every environment once and writes one PNG per tile. Four images that look like four
plausible scenes means the vectorized path is wired up. It is much faster to debug here than inside
a real evaluation.

## 4. A real evaluation

Now you need a policy server. Start one — for π0.5 the repo has a launcher:

```sh
./scripts/clara/interactive/pi05_server.sh
```

It defaults to port `8000` and takes about 70 seconds to come up, using roughly 12 GB of VRAM. Wait
for it to be listening before starting the eval; every batch launcher in the repo does a socket
preflight for exactly this reason, because the client **blocks forever retrying** rather than failing
if nothing is there.

Single environment:

```sh
./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 \
    --repeats 25 --max_steps 800 --horizon 8 \
    --model_type openpi --model_name checkpoints_pi05_droid_jointpos \
    --host 127.0.0.1 --port 8000 \
    --experiment_name pi05 --run_id single --log_dir /logs
```

Vectorized — note this is a **different script** with a slightly different flag set:

```sh
./scripts/clara/interactive/rr \
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

> **Four perturbations are not safe vectorized** — `VB-POSE`, `VB-MOBJ`, `VSB-NOBJ` and `SB-VRB` stop
> and restart the simulator globally, which is not per-environment. Run those single-env.

## Or just submit a batch job

`sbatch_eval_pi05.sh` does the whole thing — allocates, starts its own policy server on a
non-colliding port, waits for it, runs the eval, and then **checks that real artifacts were produced**
before reporting success:

```sh
VEC=4 PERT_ID=0 MAX_STEPS=800 REPEATS=25 RUN_ID=def_vec4 \
  sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
```

It is configured entirely through environment variables, and `VEC` selects the path: `VEC>=1` runs
the vectorized script with that many environments, `VEC=0` runs the single-env script.

> **A SLURM exit code of 0 proves nothing here.** Isaac's shutdown call hard-exits with status 0, so
> an unhandled Python exception still produces a `COMPLETED` job that wrote no results. This has
> happened. Always check the artifacts — which is what that launcher's final gate does for you.

## Next

- [Tasks and perturbations](Tasks-and-Perturbations) — the 10 × 16 matrix
- [Running evaluations](Running-Evaluations) — every flag, and what `MODE` does
- [Cluster and parallel runs](Cluster-and-Parallel-Runs) — sweeping the matrix
- [Known issues and gotchas](Known-Issues-and-Gotchas) — read before debugging anything
