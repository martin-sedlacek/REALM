# Installation

**REALM is not pip-installed.** There is no `pyproject.toml`, no `requirements.txt` and no setup step
on the host. Every dependency — OmniGibson 3.9.1, Isaac Sim 5.1, PyTorch, the lot — is baked into a
single Apptainer image, and the repo is bind-mounted into it at `/app`. Python finds REALM because
the image sets `PYTHONPATH=/app`.

So "installing REALM" means three things:

1. get the container image,
2. get the BEHAVIOR dataset and assets,
3. register REALM's robot definitions into the dataset directory.

Plus a GPU. REALM does not run on CPU in any useful sense.

> **Heads up before you follow the repo README.** Its install path (`./setup.sh --docker --dataset`)
> **does not work on this branch** — it builds from `.docker/realm.Dockerfile` and `.docker/realm.def`,
> and neither file exists any more; only the `realm_og391` pair does. Two scripts also tell you to run
> `./scripts/download_dataset.sh`, which does not exist either. See
> [Known issues](Known-Issues-and-Gotchas). The path documented on this page is the one that is
> actually used.

## 1. The container image

The image is `realm_og391.sif`, roughly 13 GB. Build recipes are in the repo and kept in sync:

- `.docker/realm_og391.def` — Apptainer, `Bootstrap: docker`, from `stanfordvl/behavior:3.9.1`
- `.docker/realm_og391.Dockerfile` — the Docker counterpart
- `.docker/og391-constraints.txt` — the version pins that make the stack cohere

Both recipes do the same thing: start from the upstream BEHAVIOR 3.9.1 image, apply the seven patches
under `realm/misc/`, then install the robotics and logging dependencies plus the vendored
`openpi-client`. Each patch is followed by a check that fails the build if the patch applied nothing,
so a silently-unpatched image is not possible.

> **Rebuilding is currently blocked on Lustre.** `apptainer build --fakeroot` fails trying to change
> ownership inside the image rootfs — a filesystem limitation, not a recipe bug. The consequence
> matters: **a rebuilt image has never been verified**; only the bind-mount path has. Until that
> changes, the substitute for "an image with the patches in it" is `MODE=stockfix` (see
> [Running evaluations](Running-Evaluations)).

Sanity-check an image without a GPU or a job:

```sh
apptainer test realm_og391.sif
```

## 2. Dataset and assets

The dataset directory is bound into the container as `/data`, and the image sets
`OMNIGIBSON_DATA_PATH=/data`. In OmniGibson 3.9.1 that **single** variable replaces the four separate
path variables that 1.1.1 used — if you are porting older notes, that is the change to make.

The directory must contain:

```
behavior-1k-assets/          # scenes and objects
omnigibson-robot-assets/     # robot models
omnigibson.key               # BEHAVIOR decryption key
```

The underlying download is OmniGibson's own `python -m omnigibson.download_datasets`, and it requires
accepting the NVIDIA Omniverse EULA. `setup.sh` wraps it, but wraps it wrongly on this branch — it
activates a `micromamba` environment named `omnigibson`, which does not exist in the 3.9.1 image (the
image uses a conda environment named `behavior`, on Python 3.11).

REALM's `data/` is gitignored. On a shared cluster it is normal for `data/datasets` to be a symlink
into a shared store rather than a real directory.

## 3. Register the robot definitions

**Easy to miss, and the failure is confusing.** OmniGibson 3.9.1 discovers robots by globbing the
dataset directory for `<data>/*/models/<name>/<name>.yaml`. REALM's robot definitions live in the
repo, not in the dataset, so they have to be linked in:

```sh
python scripts/install_robot_definitions.py
```

Flags: `--copy` to copy instead of symlinking, `--data-path` to point at a dataset other than
`$OMNIGIBSON_DATA_PATH`.

These links are **not tracked in git**, so they do not come with a clone and they do not survive a
fresh dataset directory. On a machine where this has not been run, typically only `droid`,
`droid_mounted` and `ur` are registered — the RoboLab robots will not load. See
[Robots and configs](Robots-and-Configs).

## 4. Check that paths resolve

Every harness script resolves its paths through `scripts/clara/lib/paths.sh`, which derives them from
its own location rather than reading them from the environment. Print what everything resolved to,
and whether it exists:

```sh
bash -c 'source scripts/clara/lib/paths.sh; realm_paths_show'
```

This is the first thing to run when something behaves oddly, and it is cheap.

### Why the overrides have an `_OG391` suffix

`paths.sh` deliberately does **not** honour `REALM_ROOT`, `REALM_SIF` or `REALM_LOGS` from the
environment. On the machine this was developed on, the shell profile exports those names pointing at
a **pre-port OmniGibson 1.1.1 tree and image** — so honouring them would silently select the wrong
container and the wrong code, with no error.

If you need to override a path, use the suffixed name:

| Override | Selects |
|---|---|
| `REALM_SHARED_OG391` | the shared store the other defaults hang off |
| `REALM_SIF_OG391` | the container image |
| `REALM_DATA_OG391` | the dataset directory (→ `/data`) |
| `REALM_LOGS_OG391` | the log directory (→ `/logs`) |
| `REALM_APPDATA_OG391` | the cache directory (→ `/cache`) |
| `REALM_STOCK_PATCH_OG391` | the patched-files directory used by `MODE=stockfix` |
| `REALM_OGLITE_OG391` | the OG-lite fork used by `MODE=oglite` |

`REALM_ROOT` is always the checkout that `paths.sh` itself lives in. That is deliberate: it is what
makes **git worktrees** work. An earlier version named the main checkout absolutely, so a worktree's
scripts bound the *main* checkout at `/app` — two agents spent time testing fixes against code they
had not edited.

## 5. A GPU allocation

REALM needs a GPU. On a SLURM cluster, hold an allocation and run against it rather than allocating
per command:

```sh
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
```

Then see [Quick start](Quick-Start).

## Verifying the install

The strongest check that needs no policy server — it runs all 16 perturbations against one task.
Like everything else, it runs **inside the container**, via `rr`:

```sh
./scripts/clara/interactive/rr \
  python -u tests/test_perturbations_integrity.py --repeats 1 --max_steps 1
```

Success prints `ALL PERTURBATIONS PASSED INTEGRITY CHECK!`. It uses the `debug` model type, which
returns a constant action and needs nothing listening on a port.

## See also

- [Quick start](Quick-Start)
- [Running evaluations](Running-Evaluations) — `rr`, `MODE`, and the full flag surface
- [Known issues and gotchas](Known-Issues-and-Gotchas)
