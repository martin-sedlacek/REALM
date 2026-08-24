# Quick start

This page takes you from a working [installation](Installation) to a small evaluation without
assuming a particular cluster, scheduler, filesystem, or account.

## 1. Obtain a GPU shell

REALM requires an NVIDIA GPU. On a workstation, run the commands below directly. On a managed
cluster, first request an interactive GPU allocation using the command documented by your site, then
run them on the allocated compute node. Do not start the simulator on a login node.

Confirm that the GPU is visible and has enough free memory:

```sh
nvidia-smi
```

## 2. Configure the container paths

From the repository root:

```sh
export REALM_SIF=/path/to/realm.sif
export REALM_DATA_PATH=/path/to/realm/data
```

`REALM_DATA_PATH` is the parent directory containing `datasets/` and the writable `isaac-sim/`
cache directories. See [Installation](Installation) for the expected layout.

Open the container:

```sh
./scripts/run_apptainer.sh
```

The remaining commands on this page run inside that shell. The repository is mounted at `/app`.

## 3. Run a server-free smoke evaluation

The `debug` model returns a constant action, so this checks simulation, rendering, and logging
without a policy server:

```sh
python -u examples/02_evaluate.py \
  --task_id 0 --perturbation_id 0 \
  --repeats 1 --max_steps 20 \
  --model_type debug --model_name debug --port 8000 \
  --experiment_name smoke --run_id first --log_dir /app/logs
```

`--port` is required by the CLI even though the debug model never connects to it. A successful run
creates output under `logs/smoke/debug/first` in the checkout.

## 4. Run a policy evaluation

REALM is a policy client; it does not ship a policy server. Start a compatible server separately and
wait until its socket accepts connections. The client retries indefinitely when the endpoint is
unavailable, so a dead port otherwise looks like a hung evaluation.

Single environment:

```sh
python -u examples/02_evaluate.py \
  --task_id 0 --perturbation_id 0 \
  --repeats 25 --max_steps 800 --horizon 8 \
  --model_type openpi --model_name YOUR_MODEL_NAME \
  --host POLICY_HOST --port POLICY_PORT \
  --experiment_name evaluation --run_id single --log_dir /app/logs
```

Vectorized:

```sh
python -u examples/04_vector_evaluate.py \
  --num_envs 4 --repeats 25 --max_steps 800 --horizon 8 \
  --task_id 0 --perturbation_id 0 \
  --model_type openpi --model_name YOUR_MODEL_NAME \
  --host POLICY_HOST --port POLICY_PORT \
  --experiment_name evaluation --run_id vector --log_dir /app/logs
```

With `--num_envs 4 --repeats 25`, rollouts run in waves of four. Start at four environments on a
high-memory GPU and measure before increasing it.

The two entry points do not expose exactly the same flags. In particular, the vectorized script has
no `--resume` or `--no_render`; see [Running evaluations](Running-Evaluations).

## 5. Verify artifacts

Do not use the process or scheduler exit code as the sole success criterion. Isaac can terminate
with status zero after an unhandled exception and can segfault during teardown after a valid result
was already written. Check that the expected report, rollout files, and media exist and contain the
requested number of repeats.

## Next

- [Tasks and perturbations](Tasks-and-Perturbations)
- [Running evaluations](Running-Evaluations)
- [Cluster and parallel runs](Cluster-and-Parallel-Runs)
- [Known issues and gotchas](Known-Issues-and-Gotchas)
