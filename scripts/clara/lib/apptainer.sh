#!/bin/bash
# Apptainer execution wrappers for REALM evaluation.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/lib/apptainer.sh"
#
# Variables read from the calling script's environment:
#   REALM_ROOT            - repository root (= pwd at SLURM job start)
#   REALM_DATA_PATH       - base path for datasets and IsaacSim caches
#   REALM_SIF             - Apptainer image (.sif) path
#   SLURM_JOB_ID          - provided by SLURM
#   OG_LITE_BIND          - set by compute_og_lite_bind() from common.sh
#   EXTRA_APPTAINER_ARGS  - optional extra --bind / --env flags (default empty)

# Run an arbitrary command inside the REALM container with all standard
# IsaacSim / OmniGibson bind mounts and environment variables applied.
#
# Usage: apptainer_exec <command> [args...]
apptainer_exec() {
  apptainer exec \
    --userns \
    --nv \
    --writable-tmpfs \
    --bind "$REALM_ROOT":/app \
    --bind "$REALM_DATA_PATH"/datasets:/data \
    --bind "$REALM_DATA_PATH"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
    --bind "$REALM_DATA_PATH"/isaac-sim/cache/ov:/root/.cache/ov \
    --bind "$REALM_DATA_PATH"/isaac-sim/cache/pip:/root/.cache/pip \
    --bind "$REALM_DATA_PATH"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
    --bind "$REALM_DATA_PATH"/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
    --bind "$REALM_DATA_PATH"/isaac-sim/logs:/root/.nvidia-omniverse/logs \
    --bind "$REALM_DATA_PATH"/isaac-sim/config:/root/.nvidia-omniverse/config \
    --bind "$REALM_DATA_PATH"/isaac-sim/data:/root/.local/share/ov/data \
    --bind "$REALM_DATA_PATH"/isaac-sim/documents:/root/Documents \
    --bind "$REALM_ROOT"/tmp/"$SLURM_JOB_ID":/tmp \
    $OG_LITE_BIND \
    $EXTRA_APPTAINER_ARGS \
    --env TMPDIR=/tmp \
    --env OMNIGIBSON_HEADLESS=1 \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env MAMBA_CACHE_DIR="$REALM_ROOT"/mamba_cache/"$SLURM_JOB_ID" \
    --env PIP_CACHE_DIR="$REALM_ROOT"/pip_cache/"$SLURM_JOB_ID" \
    "$REALM_SIF" \
    "$@"
}

# Run a command string inside the omnigibson micromamba environment,
# preceded by the standard set of pip package installs.
#
# Variables in the command string must already be expanded by the caller
# (pass the string with double quotes so the shell expands them first).
#
# Usage: apptainer_eval "python examples/02_evaluate.py --arg val ..."
apptainer_eval() {
  apptainer_exec \
    micromamba run -n omnigibson bash -c "
      pip install json_numpy --quiet &&
      pip install zmq --quiet &&
      pip install msgpack --quiet &&
      pip install openai --quiet &&
      pip install /app/packages/openpi-client --quiet --force-reinstall --no-deps &&
      $1
    "
}

