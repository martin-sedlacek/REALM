#!/bin/bash
#SBATCH --job-name omnigibson-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --exclude=l40s-06
#SBATCH --time 00-04:30:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/server.sh"
source "$SCRIPT_DIR/lib/apptainer.sh"

#--- Argument parsing ----------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config)         POLICY_CONFIG="$2";             shift 2 ;;
    --checkpoint_path)       CHECKPOINT_PATH="$2";           shift 2 ;;
    --policy_run_dir)        POLICY_RUN_DIR="$2";            shift 2 ;;
    --base_port|--base-port) BASE_PORT="$2";                 shift 2 ;;
    --max_steps)             MAX_STEPS="$2";                 shift 2 ;;
    --horizon)               HORIZON="$2";                   shift 2 ;;
    --repeats)               REPEATS="$2";                   shift 2 ;;
    --experiment_name)       EXPERIMENT_NAME="$2";           shift 2 ;;
    --task_id)               TASK_ID="$2";                   shift 2 ;;
    --task_cfg_path)         TASK_CFG_PATH="$2";             shift 2 ;;
    --perturbation_id)       PERTURBATION_ID="$2";           shift 2 ;;
    --run_id)                RUN_ID="$2";                    shift 2 ;;
    --model_type)            MODEL_TYPE="$2";                shift 2 ;;
    --host)                  HOST="$2";                      shift 2 ;;
    --spp)                   SPP="$2";                       shift 2 ;;
    --debug)                 DEBUG=true;                     shift 1 ;;
    --rendering_mode)        RENDERING_MODE="$2";            shift 2 ;;
    --multi-view)            MULTI_VIEW_FLAG="--multi-view"; shift 1 ;;
    --resume)                RESUME_FLAG="--resume";         shift 1 ;;
    --no_render)             NO_RENDER_FLAG="--no_render";   shift 1 ;;
    --no_record)             NO_RECORD_FLAG="--no_record";   shift 1 ;;
    --robot)                 ROBOT_FLAG="--robot $2";        shift 2 ;;
    --og_lite)               OG_LITE=true;                   shift 1 ;;
    *) shift ;;
  esac
done

#--- Setup ---------------------------------------------------------------------

# TODO: try commenting these out to see if it runs faster on clara??
setup_hf_cache
compute_og_lite_bind

port=$((BASE_PORT + PERTURBATION_ID + 100 * TASK_ID))

[ "$DEBUG" = "false" ] && start_policy_server

cd "$REALM_ROOT" || exit
setup_job_dirs
compute_model_name

[ -n "$TASK_CFG_PATH" ] && TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH" || TASK_CFG_ARG=""

#--- Evaluation ----------------------------------------------------------------

apptainer_eval "python examples/02_evaluate.py \
  --perturbation_id $PERTURBATION_ID \
  --task_id $TASK_ID \
  $TASK_CFG_ARG \
  --repeats $REPEATS \
  --max_steps $MAX_STEPS \
  --horizon $HORIZON \
  --model_name $MODEL_NAME \
  --model_type $MODEL_TYPE \
  --port $port \
  --host $HOST \
  --spp $SPP \
  --run_id $RUN_ID \
  --experiment_name $EXPERIMENT_NAME \
  --rendering_mode $RENDERING_MODE \
  $MULTI_VIEW_FLAG \
  $RESUME_FLAG \
  $NO_RENDER_FLAG \
  $NO_RECORD_FLAG \
  $ROBOT_FLAG"

EXIT_CODE=$?
cleanup_job_dirs $EXIT_CODE
exit $EXIT_CODE
