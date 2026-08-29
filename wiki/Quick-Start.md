# Quick Start

This page assumes the container, dataset and robot definitions are installed. See
[Installation](Installation) first.

## Start on a GPU

REALM requires an NVIDIA GPU. On a cluster, enter an allocated compute node before starting the
container.

```sh
nvidia-smi
```

## Open the container

From the repository root:

```sh
export REALM_SIF=/path/to/realm.sif
export REALM_DATA_PATH=/path/to/realm/data
./scripts/run_apptainer.sh
```

`REALM_DATA_PATH` contains `datasets/` and the writable `isaac-sim/` cache directories. The
repository is mounted into the container at `/app`.

## Run a smoke evaluation

The debug model returns a constant action and does not need a policy server:

```sh
python -u examples/02_evaluate.py \
  --task_id 0 --perturbation_id 0 \
  --repeats 1 --max_steps 20 \
  --model_type debug --model_name debug --port 8000 \
  --experiment_name smoke --run_id first --log_dir /app/logs
```

`--port` is required by the CLI but is not used by the debug model. The run should create:

```text
logs/smoke/debug/first/
```

## Run with a policy server

REALM is a policy client. Start a compatible policy server separately, then pass its host and port:

```sh
python -u examples/02_evaluate.py \
  --task_id 0 --perturbation_id 0 \
  --repeats 25 --max_steps 800 --horizon 8 \
  --model_type openpi --model_name MODEL_NAME \
  --host POLICY_HOST --port POLICY_PORT \
  --experiment_name evaluation --run_id single --log_dir /app/logs
```

For vectorized evaluation:

```sh
python -u examples/04_vector_evaluate.py \
  --num_envs 4 --repeats 25 --max_steps 800 --horizon 8 \
  --task_id 0 --perturbation_id 0 \
  --model_type openpi --model_name MODEL_NAME \
  --host POLICY_HOST --port POLICY_PORT \
  --experiment_name evaluation --run_id vector --log_dir /app/logs
```

The policy client retries when the server is unavailable, so check the server before starting a run.

## Check the output

Do not trust only the process exit code. Check that the report contains the requested repeats and
that the expected actions, qpos and videos were written.

See [Logging](Logging) for the output format and [Running evaluations](Running-Evaluations) for all
arguments.
