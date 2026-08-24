# Prompt: port the REALM cluster scripts to OmniGibson 3.9.1

Hand the section below to a coding agent running on the cluster. It is self-contained.

---

## Task

REALM has been ported from OmniGibson 1.1.1 (`stanfordvl/omnigibson:1.1.1`) to OmniGibson 3.9.1
(`stanfordvl/behavior:3.9.1`) on branch `port-to-og391`. The Python side, the Docker image and a new
Apptainer image are all done and verified on a workstation. **Only the cluster shell scripts are left.**
They still assume the 1.1.1 image and will fail immediately if submitted as-is.

Your job: update the cluster scripts so REALM evaluations run on this cluster against the new image.

### What already exists here

| Thing | Path |
| --- | --- |
| Repo | `/home/sedlam56/projects/REALM` (branch `port-to-og391`) |
| New Apptainer image | `/home/sedlam56/projects/REALM/realm_og391.sif` (13 GB) |
| New BEHAVIOR-1K 3.9.1 dataset | `/home/sedlam56/projects/REALM/data/datasets_og391` |
| Old 1.1.1 dataset — **do not touch or delete** | `/home/sedlam56/projects/REALM/data/datasets` |
| Image definition (reference for what is inside the sif) | `.docker/realm_og391.def` |
| Docker equivalent (reference) | `.docker/realm_og391.Dockerfile` |

### What changed in the image, and why the scripts break

1. **Python env**: conda env **`behavior`** (Python 3.11) replaces micromamba env `omnigibson` (3.10).
   Activate with `. /opt/conda/etc/profile.d/conda.sh && conda activate behavior`.
   `micromamba run -n omnigibson ...` no longer works — that env does not exist.
2. **OmniGibson source** lives at `/behavior-src/OmniGibson`, not `/omnigibson-src`.
3. **Dataset layout**: one `OMNIGIBSON_DATA_PATH=/data` (already set inside the image) replaces
   `OMNIGIBSON_DATASET_PATH` / `OMNIGIBSON_ASSET_PATH` / `GIBSON_DATASET_PATH` / `OMNIGIBSON_KEY_PATH`.
   Under `/data` the tree is `behavior-1k-assets/` + `omnigibson-robot-assets/` + `omnigibson.key`,
   **not** the old `assets/` + `og_dataset/`.
4. **`/isaac-sim` does not exist any more.** Isaac Sim 5.1 is installed as pip wheels inside the conda
   env. The image declares `/cache` as its cache volume (`OMNIGIBSON_APPDATA_PATH=/cache/appdata`).
5. **Robots are data, not classes.** REALM's robots are `RobotDefinition` YAMLs in
   `realm/robots/definitions/`, discovered by globbing `<OMNIGIBSON_DATA_PATH>/*/models/*/*.yaml`.
   They are already symlinked into `data/datasets_og391/omnigibson-robot-assets/models/` as
   `droid`, `droid_mounted`, `ur`, pointing at `/app/realm/robots/definitions/<name>` — these resolve
   only when the repo is bound at `/app`. `scripts/install_robot_definitions.py` recreates them if needed.

### Concrete work items

These were identified by inspecting the scripts and verifying against the sif. Re-verify rather than
trusting the list blindly.

1. **Replace micromamba activation with conda** in all 9 files that reference it:
   `scripts/eval.sh`, `scripts/clara/lib/apptainer.sh`,
   `scripts/clara/run_{integrity_test,pert_integrity_test,pi0_integration_test,eval_single_debug}.sh`,
   `scripts/clara/tests/run_pi0_integration_test.sh`, `scripts/cluster_evals/run_single_eval.sh`,
   `scripts/karolina/run_eval_single.sh`.
   Most funnel through `apptainer_eval()` in `scripts/clara/lib/apptainer.sh` — **start there.**
2. **Repoint the dataset bind.** Scripts bind `$REALM_DATA_PATH/datasets:/data`; that is now the 1.1.1
   tree. Use `datasets_og391`. Prefer a single overridable variable over hardcoding.
3. **Fix the Kit cache bind.** `--bind $REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit`
   targets a path that no longer exists; with `--writable-tmpfs` Apptainer may create it silently, so
   the shader cache is discarded every job (slow startup, no error). Find where Isaac Sim 5.1 / the
   image actually caches (start from `/cache` and `OMNIGIBSON_APPDATA_PATH`) and bind persistent
   storage there instead. Verify a second run is faster than the first.
4. **`scripts/run_apptainer.sh` uses `apptainer shell`, which bypasses `%runscript`**, so the conda env
   is NOT activated and you land on the base Python 3.13 with no omnigibson. Activate explicitly.
5. **`REALM_SIF`** must point at `realm_og391.sif`.
6. **OG-lite bind path**: `--bind ../OG-lite:/omnigibson-src` → `/behavior-src/OmniGibson`
   (6 scripts). Note OG-lite is still a 1.1.1 fork and will not work against 3.9.1 regardless — wire
   the path correctly but do not spend time trying to make `--og-lite` runs succeed.
7. **Runtime pip installs**: `apptainer_eval()` runs `pip install json_numpy zmq msgpack openai` every
   job. In this image `zmq`, `msgpack` and `openai` are already present; only **`json_numpy`** is
   missing. On a read-only sif these installs land in tmpfs and are thrown away each job. Either drop
   the redundant ones, or better, add `json_numpy` to `.docker/realm_og391.def` and remove the line
   (rebuilding the sif is a separate, larger task — coordinate before doing it).

### Constraints

- **Do not modify anything under `data/`.** The datasets are large and were transferred manually.
  In particular `data/datasets` (1.1.1) must stay intact — the old image still uses it.
- **Do not rebuild or overwrite `realm_og391.sif`** without asking; it was built and checksum-verified
  off-cluster.
- Edit scripts **in place** on branch `port-to-og391` (the branch is the 1.1.1-vs-3.9.1 switch), rather
  than adding parallel `*_og391.sh` variants — unless told otherwise.
- Do not change REALM Python source unless a script fix genuinely requires it. The Python port is done
  and validated; unexplained edits there are more likely to be regressions.

### How to verify

Cheap checks first, in order:

```bash
cd /home/sedlam56/projects/REALM

# 1. Image sanity (no GPU, no job needed)
apptainer test realm_og391.sif

# 2. Env + dataset + robot registry resolve
apptainer exec --nv \
  --bind $PWD:/app --bind $PWD/data/datasets_og391:/data \
  realm_og391.sif bash -lc '
    . /opt/conda/etc/profile.d/conda.sh && conda activate behavior
    python -c "
import omnigibson as og
from omnigibson.robots import REGISTERED_ROBOTS
import realm
print(og.__version__, og.macros.gm.DATA_PATH)
print([r for r in REGISTERED_ROBOTS if r in (\"droid\",\"droid_mounted\",\"ur\")])"'
```

Expected: `3.9.1 /data` and `['droid', 'droid_mounted', 'ur']`.

Then, on a GPU node, the real end-to-end check — every perturbation, 3 repeats, 1 step, no policy
server required (`model_type=debug`):

```bash
OMNIGIBSON_HEADLESS=1 python tests/test_perturbations_integrity.py --repeats 3 --max_steps 1
```

Expected: `ALL PERTURBATIONS PASSED INTEGRITY CHECK!` (16/16). This is the reference result — it passes
on the workstation, so any failure here is a cluster/script problem, not a REALM bug.

To eyeball rollouts (videos are stored as mp4 bytes inside parquet, one row per rollout, appended
across runs — so they are not directly viewable and the newest is the *last* row):

```bash
Use the recorded parquet videos directly when visually reviewing perturbation runs.
```

### Gotchas worth knowing before you start

- **Apptainer-only failure mode**: OmniGibson uses `@jit(cache=True)` and numba writes its cache next
  to the source file. A sif is read-only, so `import omnigibson` dies with
  `cannot cache function '_quat_multiply': no locator available` unless `NUMBA_CACHE_DIR` points
  somewhere writable. The image sets it to `/tmp/numba_cache` and the scripts already bind a per-job
  writable `/tmp`, so this should be fine — but if `/tmp` is a shared filesystem on this cluster,
  point `NUMBA_CACHE_DIR` at node-local scratch. This never reproduces under Docker.
- `apptainer shell` and `apptainer exec` do **not** run `%runscript`; only `apptainer run` does. Any
  script relying on the image to activate conda for it will silently get the wrong Python.
- The integrity test spawns a fresh Isaac Sim per perturbation (~3 min startup each), so a full sweep
  is ~1 hour. Budget SLURM time accordingly and prefer running it as a batch job, not interactively.

### Report back

State which scripts you changed and why, what you verified versus what you only reasoned about, and
anything you found that this list did not predict. If something is broken in a way that needs a
decision (e.g. the sif needs rebuilding), stop and ask rather than guessing.
