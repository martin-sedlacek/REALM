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


# Warn early when the selected image does not contain OmniGibson 3.9.1.
if ! apptainer exec "$REALM_SIF" test -d /behavior-src 2>/dev/null; then
    echo "WARNING: $REALM_SIF has no /behavior-src -- that looks like the PRE-PORT (1.1.1) image," >&2
    echo "         but this checkout requires the current REALM image. Set REALM_SIF to realm.sif if that" >&2
    echo "         was not deliberate." >&2
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

cd $REALM_ROOT
# Apptainer requires bind sources to exist.
mkdir -p $REALM_ROOT/tmp
# OMNIGIBSON_APPDATA_PATH=/cache/appdata in the 3.9.1 image, and Isaac is started with
# --portable-root under it. Unbound, that lands in apptainer's --writable-tmpfs overlay, which is
# capped by `sessiondir max size` in apptainer.conf (64 MiB on some hosts) -- the material-library
# cache then fills it and Kit raises OSError: [Errno 28] No space left on device from an async
# preload task. That exception is non-fatal, so the run continues, but tests/test_vector_integrity
# classifies any child log containing "Traceback" as a CRASH -- so a full-disk overlay reads as a
# simulator crash. scripts/run_docker.sh already binds this; keep the two launchers in step.
mkdir -p $REALM_DATA_PATH/cache
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/kit
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/ov
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/pip
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/glcache
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/computecache
mkdir -p $REALM_DATA_PATH/isaac-sim/logs
mkdir -p $REALM_DATA_PATH/isaac-sim/config
mkdir -p $REALM_DATA_PATH/isaac-sim/data
mkdir -p $REALM_DATA_PATH/isaac-sim/documents

# `run` activates the image environment; the explicit shell avoids host startup files.
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
  --bind $REALM_DATA_PATH/cache:/cache \
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
