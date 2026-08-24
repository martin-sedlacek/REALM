# Interactive Apptainer harness for REALM on Clara

The workstation this project grew up on ran two Docker containers, `realm_stock` and `realm_oglite`.
Clara has no Docker. These scripts are the replacement: **one image plus a bind**, driven from a held
Slurm allocation so you iterate without paying `sbatch` queue time and Isaac cold starts.

They live here, tracked, on purpose. The previous generation of this harness lived in `tmp/` and was
lost with the machine. **Only artifacts belong in `tmp/`** -- `tmp/interactive/logs/` for run
logs, `tmp/interactive/prof/` for profiler JSON.

## Getting an allocation

```bash
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
srun --jobid=<ID> --overlap nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

That second line is not optional: a GPU Slurm hands you is not necessarily empty. Empty output is
what you want.

## The two entry points

| | |
| --- | --- |
| `rr` | runs a command inside the container. `MODE=stock` uses the image's own OmniGibson 3.9.1; `MODE=oglite` binds the `OG-lite_og391` fork over it. |
| `go` | runs a script inside the held allocation, tees to `tmp/interactive/logs/<name>.log`, appends an explicit `### EXIT_CODE=` marker. |

```bash
MODE=oglite ./scripts/clara/interactive/rr python -u examples/02_evaluate.py --task_id 0 ...
ALLOC=<ID> ./scripts/clara/interactive/go smoke ./scripts/clara/interactive/check_run.py --help
```

## Everything else

| file | what it is for |
| --- | --- |
| `show_macros.py` | prove a flag actually reached `gm` **before** spending a run on it |
| `check_run.py` | the REALM pass criteria; exit 0 alone is not one of them. `--repeats N` also demands the full rollout count and `--newer-than EPOCH` that the artifacts are *this* run's; `sbatch_eval_pi05.sh` gates on it and exits non-zero when it fails |
| `pi05_server.sh` | resident pi0.5 policy server on :8000 |

## Four traps these encode

1. **Never `bash -lc` inside the container.** Apptainer binds `$HOME`, so a login shell re-sources the
   host `~/.bashrc`, prepends `~/miniconda3/bin` to PATH and shadows the conda env: you get host
   Python and `ModuleNotFoundError: No module named 'omnigibson'`. Use `bash -c`, or call `python`.
2. **`atexit` never fires.** `og.shutdown()` -> `SimulationApp.close()` hard-exits, skipping `atexit`
   and `finally`. Persist required artifacts before shutdown.
3. **Set `--pwd /app`.** Otherwise the container inherits the submit directory and can import a
   different REALM checkout than the one it bound.
4. **`gm` lies in the stock image**: undefined macros return a truthy `{'_read': set()}` rather than
   raising. Check the live source, not `getattr`.
