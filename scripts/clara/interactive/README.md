# Interactive Apptainer harness for REALM on Clara

The workstation this project grew up on ran two Docker containers, `realm_stock` and `realm_oglite`.
Clara has no Docker. These scripts are the replacement: **one image plus a bind**, driven from a held
Slurm allocation so you iterate without paying `sbatch` queue time and Isaac cold starts.

They live here, tracked, on purpose. The previous generation of this harness lived in `tmp/` and was
lost with the machine (`docs/perf/og391_step_profile.md` still refers to the vanished
`tmp/fork_ab_profile.py`). **Only artifacts belong in `tmp/`** -- `tmp/interactive/logs/` for run
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
ALLOC=190155 ./scripts/clara/interactive/go inc_on ./scripts/clara/interactive/t2_inc_on.sh
```

## Everything else

| file | what it is for |
| --- | --- |
| `show_macros.py` | prove a flag actually reached `gm` **before** spending a run on it |
| `check_run.py` | the REALM pass criteria; exit 0 alone is not one of them. `--repeats N` also demands the full rollout count and `--newer-than EPOCH` that the artifacts are *this* run's; `sbatch_eval_pi05.sh` gates on it and exits non-zero when it fails |
| `t10_bhobj_props.py` | does B-HOBJ's mass / stiffness / damping / max-effort perturbation compound across resets? `--legacy` re-measures the pre-fix drift, `--add_articulated` reaches the joint half |
| `t11_eval_gate.sh` | does `sbatch_eval_pi05.sh` still refuse to call a crashed run a success? Host-only, seconds, no allocation -- run it after touching either that script or `check_run.py` |
| `t1_scene_probe.py` / `t1_probe.sh` | per-member scene dump for vector envs: names, z distribution, stage prims, state either side of the scene fixes |
| `t1_frames.sh` | the 4-env first-frame montage |
| `t2_inc_on.sh` | correctness gate for `gm.INCREMENTAL_CONTACT_CACHE` |
| `t2_ab_contact.sh` / `analyze_ab.py` | interleaved A/B of the incremental contact cache under pi0.5 |
| `profile_step.py` | contact-cache and `_non_physics_step` timing |
| `profile_phases.py` | cold start / reset / step, **portable across OG 1.1.1 and 3.9.1** |
| `sbatch_phase_ref_og{111,391}.sh` / `compare_phases.py` | pre-port vs ported reference benchmark |
| `pi05_server.sh` | resident pi0.5 policy server on :8000 |

## Four traps these encode

1. **Never `bash -lc` inside the container.** Apptainer binds `$HOME`, so a login shell re-sources the
   host `~/.bashrc`, prepends `~/miniconda3/bin` to PATH and shadows the conda env: you get host
   Python and `ModuleNotFoundError: No module named 'omnigibson'`. Use `bash -c`, or call `python`.
2. **`atexit` never fires.** `og.shutdown()` -> `SimulationApp.close()` hard-exits, skipping `atexit`
   and `finally`. Jobs complete, exit 0, and write nothing. Both profilers hook `og.shutdown` and
   checkpoint every 400 samples.
3. **Set `--pwd /app`.** Otherwise the container inherits the submit directory and can import a
   different REALM checkout than the one it bound.
4. **`gm` lies in the stock image**: undefined macros return a truthy `{'_read': set()}` rather than
   raising. Check the live source, not `getattr`.
