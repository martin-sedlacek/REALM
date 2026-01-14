#!/bin/bash
#SBATCH --job-name omnigibson-test
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time 00-08:00:00

#---------------------------------------------------------------------------------

DEBUG=false
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config) POLICY_CONFIG="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --policy_run_dir) POLICY_RUN_DIR="$2"; shift 2 ;;
    --base_port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_id) TASK_ID="$2"; shift 2 ;;
    --perturbation_id) PERTURBATION_ID="$2"; shift 2 ;;
    --run_id) RUN_ID="$2"; shift 2 ;;
    --debug) DEBUG=true; shift 1;;
    *) shift ;;
  esac
done

REALM_ROOT=$(pwd)

#---------------------------------------------------------------------------------

export HF_HOME=$REALM_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
[[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"

export XDG_CACHE_HOME=$REALM_ROOT/python_cache

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

POLICY_SCRIPT="${POLICY_RUN_DIR}/scripts/serve_policy.py" # TODO: assuming default openppi repo
port=$((BASE_PORT + PERTURBATION_ID + 100 * TASK_ID))

if [ "$DEBUG" = "false" ]; then
  cd "$POLICY_RUN_DIR"
  uv run "$POLICY_SCRIPT" \
    --port=$port policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$CHECKPOINT_PATH" & SERVER_PID=$!
  sleep 60
fi

#---------------------------------------------------------------------------------

cd $REALM_ROOT
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"

if [ "$DEBUG" = "true" ]; then
  MODEL_NAME="debug"
else
  CLEAN_PATH="${CHECKPOINT_PATH%/}"
  MODEL_NAME=$(basename "$(dirname "${CLEAN_PATH%/}")")_$(basename "${CLEAN_PATH%/}")
fi

apptainer exec \
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
  --bind $REALM_ROOT/tmp/$SLURM_JOB_ID:/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env MAMBA_CACHE_DIR=$REALM_ROOT/mamba_cache/$SLURM_JOB_ID \
  --env PIP_CACHE_DIR=$REALM_ROOT/pip_cache/$SLURM_JOB_ID \
  $REALM_SIF \
  micromamba run -n omnigibson python examples/02_eval_dynamic_scenes.py \
  --perturbation_id $PERTURBATION_ID \
  --task_id $TASK_ID \
  --repeats $REPEATS \
  --max_steps $MAX_STEPS \
  --model $MODEL_NAME \
  --port $port \
  --run_id $RUN_ID \
  --experiment_name $EXPERIMENT_NAME
