#!/bin/bash

set -e

if [[ -z "${REALM_SIF:-}" ]]; then
  echo "REALM_SIF is not set."
  echo "Set it to the Singularity image to use, e.g.:"
  echo "  export REALM_SIF=\"/path/to/the/sif/file\""
  exit 1
fi

if [[ -z "${REALM_DATA_PATH:-}" ]]; then
  echo "REALM_DATA_PATH is not set."
  echo "Set it to the path where omnigibson dataset and IsaacSim cache located to use, e.g.:"
  echo "  export REALM_DATA_PATH=\"/path/to/the/realm/data\""
  echo "Run ./scripts/download_dataset.sh , if you haven't downloaded data yet"
  exit 1
fi


# Sanity-check the image against the port, because the failure otherwise is confusing and slow.
# OmniGibson 3.9.1 lives at /behavior-src in this port's image; the pre-port 1.1.1 image has it at
# /omnigibson-src. The shell profile on the dev machine still exports REALM_SIF pointing at the
# 1.1.1 image, so running this from the current checkout picks up the wrong container and dies several
# minutes later with "No module named 'omnigibson'" or a missing-path assert. Warn rather than
# fail: passing a deliberately different image is legitimate.
if ! apptainer exec "$REALM_SIF" test -d /behavior-src 2>/dev/null; then
    echo "WARNING: $REALM_SIF has no /behavior-src -- that looks like the PRE-PORT (1.1.1) image," >&2
    echo "         but this checkout requires the current REALM image. Set REALM_SIF to realm.sif if that" >&2
    echo "         was not deliberate." >&2
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

cd $REALM_ROOT
# Bound at /tmp below. A fresh checkout does not ship it, and apptainer refuses to start when a
# bind SOURCE is missing, so the launcher has to create it like every other bind target.
mkdir -p $REALM_ROOT/tmp
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/kit
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/ov
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/pip
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/glcache
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/computecache
mkdir -p $REALM_DATA_PATH/isaac-sim/logs
mkdir -p $REALM_DATA_PATH/isaac-sim/config
mkdir -p $REALM_DATA_PATH/isaac-sim/data
mkdir -p $REALM_DATA_PATH/isaac-sim/documents

# WHY `run` AND WHY AN EXPLICIT SHELL. Only the image's %runscript activates the conda env
# `behavior` (python 3.11) that OmniGibson is installed into, and ONLY `apptainer run` executes it.
#   * `apptainer shell` skips the runscript -- you land in conda env `base` on python 3.13, and the
#     first REALM command dies at `import omnigibson`. That is what this script used to do.
#   * `apptainer run` with NO arguments is not enough either: the runscript ends in
#     `exec /bin/bash --login`, and a LOGIN shell re-sources the HOST ~/.bashrc (apptainer binds
#     $HOME by default), so a host conda init silently wins -- measured handing back
#     /home/<user>/miniconda3/bin/python with the container env discarded.
# So: `run`, always with an explicit command, and the interactive default is a bash that reads no
# rc files. `--norc --noprofile` is what keeps the host's dotfiles out of the container.
#
# Anything passed to this script is run inside the container instead of the interactive shell:
#   ./scripts/run_apptainer.sh                        # interactive
#   ./scripts/run_apptainer.sh python -u examples/02_evaluate.py --task_id 0 ...
if [ $# -eq 0 ]; then
  set -- bash --norc --noprofile
fi

echo "Ready to launch singularity"
apptainer run \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind $(pwd):/app \
  --bind $REALM_DATA_PATH/datasets:/data \
  --bind $REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
  --bind $REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov \
  --bind $REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip \
  --bind $REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind $REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
  --bind $REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs \
  --bind $REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config \
  --bind $REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data \
  --bind $REALM_DATA_PATH/isaac-sim/documents:/root/Documents \
  --bind $REALM_ROOT/tmp:/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  $REALM_SIF \
  "$@"
