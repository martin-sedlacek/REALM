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
`openpi-client`.

Two separate safeguards, worth keeping straight: the `%post` section runs under `set -e`, so a
**failed** `patch` aborts the build; and `%test` greps for a marker string from each of the seven
patches, catching one that applied but produced the wrong result. **The greps are in `%test`, so
`apptainer build --notest` skips them** — do not build with `--notest` and assume the image is
verified.

Build it **from the repository root** — the recipe copies patches and the vendored client in by
repo-relative path, so it will not build from anywhere else:

```sh
apptainer build realm_og391.sif .docker/realm_og391.def
```

> **Two caveats, and they are the reason there is no one-line install.**
>
> 1. **Build somewhere that is not Lustre.** On Lustre, `apptainer build --fakeroot` fails trying to
>    change ownership inside the image rootfs. That is a filesystem limitation, not a recipe bug —
>    build on local disk and move the resulting `.sif`.
> 2. **A rebuilt image has never been verified.** Only the bind-mount path has been exercised. The
>    recipes are believed correct and the patch checks are real, but nobody has yet confirmed that an
>    image built from them behaves identically to the one in use. Until someone does, the substitute
>    for "an image with the patches in it" is `MODE=stockfix` — see
>    [Running evaluations](Running-Evaluations).
>
> There is no published prebuilt image. If you are joining an existing deployment, get the `.sif`
> path from whoever runs it rather than rebuilding.

Sanity-check an image without a GPU or a job — it checks installed package versions and greps for
each patch marker:

```sh
apptainer test --bind /path/to/datasets:/data realm_og391.sif
```

`%test` deliberately does **not** import `omnigibson` (importing it asserts that the data path
exists), so a pass means "the image was built correctly", not "the simulator runs".

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

The script installs **all five** definitions in one pass and exits on the first failure — it is
all-or-nothing, so there is no partially-registered state to diagnose. The links are **not tracked in
git**, so they do not come with a clone and they do not survive a fresh dataset directory: until you
run this, **none** of REALM's robots are registered and even `--robot DROID` will fail with
`... is not a registered robot`. See [Robots and configs](Robots-and-Configs).

## 4. Check that paths resolve

Every harness script resolves its paths through `scripts/clara/lib/paths.sh`. Print what everything
resolved to, and whether it exists:

```sh
bash -c 'source scripts/clara/lib/paths.sh; realm_paths_show'
```

Each path line is marked `ok` or `MISSING` (the leading `(cwd)` line is informational). This is the
first thing to run when something behaves oddly, and it is cheap.

Only three of them are prerequisites — `REALM_SIF`, `REALM_DATA` and `REALM_LOGS`, which are what
`rr` refuses to start without. **`REALM_APPDATA` reading `MISSING` on a fresh checkout is normal:**
it is the per-checkout shader cache, and `rr` creates it on first run.

### Pointing it at your machine

Only `REALM_ROOT` is derived from the script's own location. **Everything else hangs off one shared
store, which is hardcoded to the original author's path.** So on any other machine the single most
useful override is:

```sh
export REALM_SHARED_OG391=/your/shared/store
```

with that directory laid out as `realm_og391.sif`, `data/datasets_og391/`, `logs/` and
`stock_patch/`. That one variable fixes the image, dataset, log, stock-patch and OG-lite lookups
together. Alternatively `realm_og391.sif`, `data/datasets/` and `logs/` inside the checkout itself
are tried first, so a self-contained clone also resolves without any variable set.

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
| `REALM_STOCK_PATCH_OG391` | the patched-files directory used by `MODE=stockfix` (`rr` also honours a bare `STOCK_PATCH`, since nothing in a shell profile sets that name) |
| `REALM_OGLITE_OG391` | the OG-lite fork used by `MODE=oglite` |

`REALM_ROOT` is always the checkout that `paths.sh` itself lives in. That is deliberate: it is what
makes **git worktrees** work. An earlier version named the main checkout absolutely, so a worktree's
scripts bound the *main* checkout at `/app` — meaning edits made in the worktree had no effect on the
run, silently, and fixes got tested against code that had never been changed.

## 5. A GPU allocation

REALM needs a GPU. On a SLURM cluster, hold an allocation and run against it rather than allocating
per command:

```sh
salloc --no-shell --job-name=realm-interactive --partition=l40s --nodes=1 \
       --cpus-per-task=32 --gres=gpu:L40S:1 --mem=120G --time=24:00:00
```

Then see [Quick start](Quick-Start).

> ### The harness under `scripts/clara/` is site-specific
>
> "clara" is the name of the cluster REALM is developed on. The scripts under
> `scripts/clara/interactive/` — including `rr`, which every other page on this wiki uses — encode
> that site's partition names, GPU types, shared-store layout and module conventions.
>
> **They are the working reference, not a portable installer.** On a different cluster you will need
> to adapt at least the `salloc` line, the paths in `scripts/clara/lib/paths.sh`, and anything that
> names an absolute location outside the repo. The *structure* transfers — bind the repo at `/app`,
> the dataset at `/data`, a log directory at `/logs`, and run through the image's runscript — and
> `rr` is short enough to read end to end before adapting it.
>
> There is currently no site-neutral launcher. If you write one, that is a welcome contribution.

## Verifying the install

The strongest check that needs no policy server — it runs all 16 perturbations against one task.
Like everything else, it runs **inside the container** and **on the allocation**: `rr` starts the
container wherever you invoke it, so it has to be reached through `srun`.

```sh
srun --jobid=<ID> --overlap ./scripts/clara/interactive/rr \
  python -u tests/test_perturbations_integrity.py --repeats 1 --max_steps 1
```

Success prints `ALL PERTURBATIONS PASSED INTEGRITY CHECK!`, preceded by one `<NAME>: PASS` line per
perturbation. It uses the `debug` model type, which returns a constant action and needs nothing
listening on a port.

> **Budget about 45 minutes, and do not leave it in the foreground of a shell you need back.**
> It runs each of the 16 perturbations in its own subprocess, so it pays a full Isaac boot sixteen
> times — the `--repeats 1 --max_steps 1` budget is not what costs. Measured 2026-08-16 on one
> L40S at `MODE=stock`: **16/16 PASS in ~43 min**. The first per-perturbation line does not appear
> for several minutes; that is the first boot, not a hang.
>
> If you want a cheaper install check, `make test-smoke` covers a different slice — one task end to
> end plus the scene check — in ~12 minutes. See [Running the test suite](Running-the-Test-Suite).

That is one test out of twelve. To run the rest — including the two static checks that need no GPU,
no container and no allocation at all — see
[Running the test suite](Running-the-Test-Suite):

```sh
make test-static                 # container-free, ~0.1 s
ALLOC=<jobid> make test          # the full suite
```

**Do not run `pytest tests/`.** Every file there is named `test_*.py` and none defines a collectable
test, so it collects zero items — after importing four modules that each boot a full Isaac instance.

## See also

- [Quick start](Quick-Start)
- [Running evaluations](Running-Evaluations) — `rr`, `MODE`, and the full flag surface
- [Running the test suite](Running-the-Test-Suite) — `make test` and what each tier needs
- [Known issues and gotchas](Known-Issues-and-Gotchas)
