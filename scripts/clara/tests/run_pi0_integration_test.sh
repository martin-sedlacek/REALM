#!/bin/bash
#SBATCH --job-name realm-pi0-integration-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 00-01:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apptainer.sh"

PORT=8000
POLICY_CONFIG="pi05_full_droid_finetune"
CHECKPOINT_PATH="gs://openpi-assets/checkpoints/pi05_droid_jointpos"
OPENPI_ROOT="/home/sedlam56/projects/openpi"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --og_lite) OG_LITE=true; shift 1 ;;
    *) shift ;;
  esac
done

compute_og_lite_bind

#--- Start Pi0 server ----------------------------------------------------------

echo "Spinning up Pi05 Server on port $PORT..."

cd "$OPENPI_ROOT"
uv run scripts/serve_policy.py \
    --port=$PORT \
    policy:checkpoint \
    --policy.config=$POLICY_CONFIG \
    --policy.dir=$CHECKPOINT_PATH & SERVER_PID=$!

echo "Waiting for server to initialize (120s)..."
sleep 120

#--- Run integration test ------------------------------------------------------

cd "$REALM_ROOT" || exit
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/logs/pi0_integration"

EXTRA_APPTAINER_ARGS="--bind $REALM_ROOT/logs/pi0_integration:/app/logs"

echo "Running Pi0-FAST Integration Test..."

apptainer_exec micromamba run -n omnigibson python tests/test_pi0_integration.py

EXIT_CODE=$?

#--- Cleanup -------------------------------------------------------------------

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
