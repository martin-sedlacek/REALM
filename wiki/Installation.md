# Installation

**REALM is not pip-installed.** Every runtime dependency — OmniGibson 3.9.1, Isaac Sim 5.1,
PyTorch, the lot — is baked into Docker or an Apptainer image, and the repo is mounted at `/app`.
Python finds REALM because the image sets `PYTHONPATH=/app`. The root `pyproject.toml` is a virtual
uv project containing only host-safe lint and static-test tools; it is not a runtime environment.

To prepare that optional host-check environment:

```sh
uv sync --locked
uv run make check
```

So "installing REALM" means three things:

1. get the container image,
2. get the BEHAVIOR dataset and assets,
3. register REALM's robot definitions into the dataset directory.

Plus a GPU. REALM does not run on CPU in any useful sense.

## 1. The container image

The image is `realm.sif`, roughly 13 GB. Build recipes are in the repo and kept in sync:

- `.docker/realm.def` — Apptainer, `Bootstrap: docker`, from `stanfordvl/behavior:3.9.1`
- `.docker/realm.Dockerfile` — the Docker counterpart
- `.docker/constraints.txt` — the version pins that make the stack cohere

Both recipes start from the upstream BEHAVIOR 3.9.1 image, install the staged OG-lite OmniGibson
package wholesale, then install the robotics and logging dependencies plus the vendored
`openpi-client`.

The build records the exact OG-lite commit and `%test` greps for each required semantic marker.
**The greps are in `%test`, so
`apptainer build --notest` skips them** — do not build with `--notest` and assume the image is
verified.

First stage OG-lite, then build from the repository root:

```sh
./scripts/stage_oglite_for_build.sh
apptainer build realm.sif .docker/realm.def
```

> **Two caveats, and they are the reason there is no one-line install.**
>
> 1. **Build somewhere that is not Lustre.** On Lustre, `apptainer build --fakeroot` fails trying to
>    change ownership inside the image rootfs. That is a filesystem limitation, not a recipe bug —
>    build on local disk and move the resulting `.sif`.
> 2. Use the validated `realm.sif` for benchmark runs. A host OG-lite bind is a development workflow
>    for testing fork changes without rebuilding.
>
> There is no published prebuilt image. If you are joining an existing deployment, get the `.sif`
> path from whoever runs it rather than rebuilding.

Sanity-check an image without a GPU or a job — it checks installed package versions and semantic
markers:

```sh
apptainer test --bind /path/to/datasets:/data realm.sif
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

OmniGibson 3.9.1 discovers robots by globbing the dataset directory for `<data>/*/models/<name>/<name>.yaml`. REALM's robot definitions live in the
repo, not in the dataset, so they have to be linked in with:

```sh
python scripts/install_robot_definitions.py
```

**It is enough to just run this once in the directory during initial installation.**

Optional flags: 

> `--copy` to copy instead of symlinking

>`--data-path` to point at a dataset other than
`$OMNIGIBSON_DATA_PATH`.

## 4. Check that paths resolve

The portable Apptainer launcher uses two explicit environment variables:

```sh
export REALM_SIF=/path/to/realm.sif
export REALM_DATA_PATH=/path/to/realm/data
./scripts/run_apptainer.sh
```

`REALM_DATA_PATH` is the parent directory containing `datasets/` and the writable `isaac-sim/`
cache directories. The launcher binds `REALM_DATA_PATH/datasets` to `/data`. `setup.sh --apptainer`
sets the same variables when it installs an image.

## (Optional) Verifying the installation

If you suspect there is something wrong with your installation, we recommend running the test suite to verify integrity.
With a GPU aviailable, you can check that whether all 16 perturbations pass an integrity check on one of the tasks. Open the
release container on a GPU node as described in [Quick start](Quick-Start), then run:

```sh
python -u tests/test_perturbations_integrity.py --repeats 1 --max_steps 1
```

Success prints `ALL PERTURBATIONS PASSED INTEGRITY CHECK!`, preceded by one `<NAME>: PASS` line per
perturbation. It uses the `debug` model type, which returns a constant action and needs nothing
listening on a port.

> ❗**Budget about 45 minutes for the test.**
> It runs each of the 16 perturbations in its own subprocess, so it pays a full Isaac col-start boot sixteen
> times — the `--repeats 1 --max_steps 1` budget is not what costs. Measured on one
> L40S GPU using the release image.

If you want a cheaper install check, `make test-smoke` covers a different slice — one task end to 
end plus the scene check — in ~12 minutes.

Remaining tests also include:

```sh
make test-static                 # container-free
python tests/run_suite.py --list # inspect the GPU/container suite
```

Also see [Quick start](Quick-Start) for next steps once the installation is verified.

## See also

- [Quick start](Quick-Start)
- [Running evaluations](Running-Evaluations)