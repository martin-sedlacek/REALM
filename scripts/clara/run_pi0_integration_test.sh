#!/bin/bash
#SBATCH --job-name realm-pi0-integration-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 00-01:00:00

#---------------------------------------------------------------------------------

REALM_ROOT=$(pwd)
PORT=8000
POLICY_CONFIG="pi0_fast_droid_jointpos"
CHECKPOINT_PATH="gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos"

# Ensure OPENPI_ROOT and OPENPI_SIF are set in environment or provided here
if [[ -z "${OPENPI_ROOT:-}" ]]; then
    # Provide a default for CI/Cluster environments
    export OPENPI_ROOT="/home/sedlam56/projects/openpi"
fi
if [[ -z "${OPENPI_SIF:-}" ]]; then
    # Provide a default for CI/Cluster environments
    export OPENPI_SIF="/home/sedlam56/projects/openpi/apptainer/openpi.sif"
fi

echo "Spinning up Pi0-FAST Server on port $PORT..."

# Use setsid to run in a new session so it doesn't get killed when this shell exits
cd "$OPENPI_ROOT"
apptainer exec \
    --writable-tmpfs \
    --nv \
    --bind "$(pwd):/app" \
    --bind "$REALM_ROOT/python_cache":/app/.cache/xdg \
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    --env XDG_CACHE_HOME=/app/.cache/xdg \
    --env GIT_LFS_SKIP_SMUDGE=1 \
    "$OPENPI_SIF" uv run /app/scripts/serve_policy.py \
        --port=$PORT \
        policy:checkpoint \
        --policy.config="$POLICY_CONFIG" \
        --policy.dir="$CHECKPOINT_PATH" & SERVER_PID=$!

echo "Waiting for server to initialize (120s)..."
sleep 120

#---------------------------------------------------------------------------------

cd $REALM_ROOT || exit
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/logs/pi0_integration"

echo "Running Pi0-FAST Integration Test..."

apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd)":/app \
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
  --bind "$REALM_ROOT"/logs/pi0_integration:/app/logs \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  $REALM_SIF \
  micromamba run -n omnigibson python tests/test_pi0_integration.py

EXIT_CODE=$?

# Cleanup
echo "Cleaning up server process..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

if [ $EXIT_CODE -eq 0 ]; then
  echo "Integration test passed successfully."
  rm -rf "$REALM_ROOT/tmp/$SLURM_JOB_ID"
else
  echo "Integration test failed (exit code $EXIT_CODE)."
fi

exit $EXIT_CODE
