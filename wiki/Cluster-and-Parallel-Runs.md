# Cluster and parallel runs

REALM supports two independent forms of parallelism:

- **Vectorization:** multiple environments in one simulator process.
- **Sweeping:** multiple task/perturbation cells as independent scheduler jobs.

The repository intentionally does not ship site-specific scheduler wrappers. The guidance below is
scheduler-neutral; translate it to your cluster's partitions, accounts, storage, and module setup.

## Interactive runs

Request an interactive GPU allocation according to your site's documentation, enter its compute
node, and confirm GPU visibility with `nvidia-smi`. Then configure `REALM_SIF` and
`REALM_DATA_PATH` and launch `./scripts/run_apptainer.sh` as described in [Quick start](Quick-Start).

Never run the simulator directly on a login node. A scheduler allocation alone may not move the
current shell onto its compute node; verify the hostname and GPU visibility before launching.

## Batch jobs

A portable batch script should perform these steps:

1. Resolve the repository, `realm.sif`, dataset, cache, log, and checkpoint paths from explicit
   arguments or environment variables.
2. Verify those paths on the compute node.
3. Start the policy server on a job-specific port when the selected model requires one.
4. Wait until the server socket accepts connections.
5. Run `examples/02_evaluate.py` or `examples/04_vector_evaluate.py` inside the container.
6. Verify the expected artifacts and rollout count before declaring success.

Use a port derived from a scheduler job or array index when several evaluations share a node. Avoid
embedding usernames, account names, partitions, or absolute site paths in a reusable launcher.

### Exit codes are insufficient

Isaac's shutdown path can hard-exit with status zero after an unhandled exception. It can also
segfault during teardown after a valid verdict was printed. Therefore:

- do not rely on `atexit` or `finally` for essential result writing;
- scan logs for tracebacks and fatal markers;
- verify output modification times and the requested number of rollouts;
- make the artifact check determine the batch job's final exit status.

## Sweeping the matrix

The task and perturbation registries live in `realm/eval.py`. Build scheduler arrays from those
registries instead of maintaining a second hard-coded list. Each array cell should receive explicit
task and perturbation IDs and a unique run ID.

Make sweeps resumable by skipping only cells whose complete expected artifact set already exists.
An output directory by itself is not proof of completion.

## Vectorization

`examples/04_vector_evaluate.py --num_envs N` builds N environments in one process and runs the
requested `--repeats` in waves of N.

Around four environments is a conservative starting point on a high-memory datacenter GPU. Memory,
renderer mode, scene complexity, and policy-server placement all affect the useful value, so measure
your hardware before increasing it.

Four perturbations need a stopped simulator because they add or remove objects: `V-SC`, `VB-MOBJ`,
`VSB-NOBJ`, and `SB-VRB`. The vector environment batches one global stop/play cycle around the wave;
these perturbations are expensive to reset, but they remain vectorizable.

The release image includes the required OmniGibson fixes. A host OG-lite bind is only needed when
developing and testing changes to that separate fork.

## Scheduler portability notes

- Batch systems commonly spool a copy of the submitted script, so resolving the repository relative
  to the script file may not work. Pass the repository path explicitly or use the scheduler's submit
  directory variable.
- Quote bind paths and arguments; shared storage paths frequently expose weak shell handling.
- Keep caches on storage suited to many small files. Building an Apptainer image on a distributed
  filesystem can fail on ownership operations; build on local disk when your site recommends it.
- Record the image checksum, repository commit, task config, model identifier, and effective command
  with every run.

## See also

- [Running evaluations](Running-Evaluations)
- [Performance and scaling](Performance-and-Scaling)
- [Known issues and gotchas](Known-Issues-and-Gotchas)
