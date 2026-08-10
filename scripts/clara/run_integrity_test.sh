#!/bin/bash
#SBATCH --job-name realm-integrity-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 00-01:00:00

#---------------------------------------------------------------------------------

REALM_ROOT=$(pwd)
OG_LITE=false
for arg in "$@"; do
  case "$arg" in --og-lite|--og_lite) OG_LITE=true ;; esac
done

cd $REALM_ROOT || exit
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"

OG_LITE_BIND=""
[ "$OG_LITE" = "true" ] && OG_LITE_BIND="--bind $REALM_ROOT/../OG-lite:/omnigibson-src"

echo "Running Task Integrity Test..."

apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd)":/app \
  $OG_LITE_BIND \
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
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env MAMBA_CACHE_DIR="$REALM_ROOT"/mamba_cache/"$SLURM_JOB_ID" \
  --env PIP_CACHE_DIR="$REALM_ROOT"/pip_cache/"$SLURM_JOB_ID" \
  $REALM_SIF \
  micromamba run -n omnigibson bash -c "
    pip install json_numpy --quiet &&
    pip install zmq --quiet &&
    pip install msgpack --quiet &&
    pip install openai --quiet &&
    python tests/test_integrity.py
  "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "Test finished successfully. Cleaning up..."
  rm -rf "$REALM_ROOT/tmp/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
else
  echo "Test failed (exit code $EXIT_CODE). Preserving temporary directories for debugging."
fi

exit $EXIT_CODE
