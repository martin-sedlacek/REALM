#!/bin/bash
#SBATCH --job-name omnigibson-test
#SBATCH --gpus 1
#SBATCH --mem 120G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 64
#SBATCH --time 00-04:30:00

#---------------------------------------------------------------------------------

REALM_ROOT=$(pwd)
RUN_ID=$(date +%Y%m%d_%H%M%S)
DEBUG=false
RENDERING_MODE="rt"
MULTI_VIEW_FLAG=""
RESUME_FLAG=""
TASK_CFG_PATH=""
NO_RENDER_FLAG=""
ROBOT_FLAG=""
BASE_PORT=8000

: "${REALM_SIF:?Set REALM_SIF to the REALM Apptainer image}"
: "${REALM_DATA_PATH:?Set REALM_DATA_PATH to the REALM data directory}"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config) POLICY_CONFIG="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --policy_run_dir) POLICY_RUN_DIR="$2"; shift 2 ;;
    --base_port|--base-port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_id) TASK_ID="$2"; shift 2 ;;
    --task_cfg_path) TASK_CFG_PATH="$2"; shift 2 ;;
    --perturbation_id) PERTURBATION_ID="$2"; shift 2 ;;
    --run_id) RUN_ID="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --debug) DEBUG=true; shift 1;;
    --rendering_mode) RENDERING_MODE="$2"; shift 2 ;;
    --multi-view) MULTI_VIEW_FLAG="--multi-view"; shift 1;;
    --resume) RESUME_FLAG="--resume"; shift 1;;
    --no_render) NO_RENDER_FLAG="--no_render"; shift 1;;
    --robot) ROBOT_FLAG="--robot $2"; shift 2 ;;
    *) shift ;;
  esac
done



#---------------------------------------------------------------------------------

export HF_HOME=$REALM_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
[[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"

export XDG_CACHE_HOME=$REALM_ROOT/python_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

port=$((BASE_PORT + PERTURBATION_ID + 100 * TASK_ID))

if [ "$DEBUG" = "false" ]; then
  if [ "$MODEL_TYPE" = "openpi" ]; then
    : "${OPENPI_SIF:?Set OPENPI_SIF to the OpenPI Apptainer image}"
    cd "$POLICY_RUN_DIR" || exit
    apptainer exec \
      --writable-tmpfs \
      --nv \
      --bind "$(pwd)":/app \
      --bind $CHECKPOINT_PATH:/checkpoint \
      --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
      --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
      --env GIT_LFS_SKIP_SMUDGE=1 \
      "$OPENPI_SIF" uv run /app/scripts/serve_policy.py \
        --port=$port \
        policy:checkpoint \
        --policy.config=$POLICY_CONFIG \
        --policy.dir=/checkpoint & SERVER_PID=$!
    sleep 120
  elif [ "$MODEL_TYPE" = "molmoact" ]; then
    : "${MOLMOACT_SIF:?Set MOLMOACT_SIF to the MolmoAct Apptainer image}"
    cd "$POLICY_RUN_DIR" || exit
    apptainer exec \
      --writable-tmpfs \
      --nv \
      --bind "$(pwd)":/app \
      --bind $CHECKPOINT_PATH:/checkpoint \
      "$MOLMOACT_SIF" /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate && pip install tyro && pip install /app/packages/openpi-client && python /app/inference/run_molmoact_server.py --port=${port}"
    sleep 120
  elif [ "$MODEL_TYPE" == "GR00T" ]; then
    cd "$POLICY_RUN_DIR" || exit
    uv run scripts/serve_gr00t.py \
      --port=$port \
      --model_path $CHECKPOINT_PATH \
      --data-config droid_joint_pos & SERVER_PID=$!
    sleep 120
  fi
fi

#---------------------------------------------------------------------------------

cd $REALM_ROOT || exit
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"

if [ "$DEBUG" = "true" ]; then
  MODEL_NAME="debug"
elif [ "$MODEL_TYPE" = "molmoact" ]; then
  MODEL_NAME="molmoact"
else
  CLEAN_PATH="${CHECKPOINT_PATH%/}"
  MODEL_NAME=$(basename "$(dirname "${CLEAN_PATH%/}")")_$(basename "${CLEAN_PATH%/}")
fi

if [ -n "$TASK_CFG_PATH" ]; then
  TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH"
else
  TASK_CFG_ARG=""
fi

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
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env MAMBA_CACHE_DIR="$REALM_ROOT"/mamba_cache/"$SLURM_JOB_ID" \
  --env PIP_CACHE_DIR="$REALM_ROOT"/pip_cache/"$SLURM_JOB_ID" \
  $REALM_SIF \
  micromamba run -n omnigibson python examples/02_evaluate.py \
  --perturbation_id $PERTURBATION_ID \
  --task_id $TASK_ID \
  $TASK_CFG_ARG \
  --repeats $REPEATS \
  --max_steps $MAX_STEPS \
  --model_name $MODEL_NAME \
  --model_type $MODEL_TYPE \
  --port $port \
  --run_id $RUN_ID \
  --experiment_name $EXPERIMENT_NAME \
  --rendering_mode $RENDERING_MODE \
  $MULTI_VIEW_FLAG \
  $RESUME_FLAG \
  $NO_RENDER_FLAG \
  $ROBOT_FLAG

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "Job finished successfully. Cleaning up..."
  rm -rf "$REALM_ROOT/tmp/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
else
  echo "Job failed (exit code $EXIT_CODE). Preserving temporary directories for debugging."
fi

exit $EXIT_CODE
