# Cluster and Parallel Runs

REALM supports two forms of parallelism:

- **Vectorization** runs multiple environments in one simulator process.
- **Sweeping** runs task and perturbation pairs as separate jobs.

Cluster launch scripts are site-specific and are not included in the repository.

## Interactive runs

Request a GPU allocation using your cluster's normal workflow and enter the compute node. Then set
`REALM_SIF` and `REALM_DATA_PATH` and run `./scripts/run_apptainer.sh` as shown in
[Quick Start](Quick-Start).

Do not start the simulator on a login node. Check `nvidia-smi` before launching.

## Batch runs

A batch script should:

1. check the image, dataset, logs and checkpoint paths;
2. start the policy server when needed;
3. wait until the policy port is available;
4. run the single or vectorized evaluation script;
5. check the number of output rollouts.

Use a different port for each parallel policy server. Record the image checksum, repository commit,
task config, model name and full evaluation command.

Isaac can exit with status 0 after an exception and can segfault during shutdown after a successful
run. Check the report files and logs instead of using only the scheduler status.

## Sweeps

Task and perturbation IDs are defined in `realm/eval.py`. Read them from the code instead of keeping
a second list in the batch scripts.

Give each task and perturbation pair a unique run ID. A resumed sweep should skip a pair only when
all expected report rows already exist.

## Vectorized evaluation

```sh
python -u examples/04_vector_evaluate.py --num_envs 4 ...
```

`--repeats` are processed in waves of `--num_envs`. Four environments is a reasonable starting
point on a high-memory GPU. Measure memory and throughput before increasing it.

`V-SC`, `VB-MOBJ`, `VSB-NOBJ` and `SB-VRB` add or remove objects and need a simulator stop/play
cycle. The vector environment performs one cycle for the whole wave, so these perturbations still
support vectorized evaluation but have slower resets.

## See also

- [Running evaluations](Running-Evaluations)
- [Performance and scaling](Performance-and-Scaling)
