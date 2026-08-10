#!/bin/bash
#SBATCH --job-name realm-eval-sequential
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 01-00:00:00

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts/clara"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/server.sh"
source "$SCRIPT_DIR/lib/apptainer.sh"

#--- Argument parsing ----------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config)         POLICY_CONFIG="$2";                      shift 2 ;;
    --checkpoint_path)       CHECKPOINT_PATH="$2";                    shift 2 ;;
    --policy_run_dir)        POLICY_RUN_DIR="$2";                     shift 2 ;;
    --base_port|--base-port) BASE_PORT="$2";                          shift 2 ;;
    --max_steps)             MAX_STEPS="$2";                          shift 2 ;;
    --horizon)               HORIZON="$2";                            shift 2 ;;
    --repeats)               REPEATS="$2";                            shift 2 ;;
    --experiment_name)       EXPERIMENT_NAME="$2";                    shift 2 ;;
    --task_ids)              T_RAW="$2"; TASK_IDS=($(expand_ids "$2")); shift 2 ;;
    --task_cfg_path)         TASK_CFG_PATH="$2";                      shift 2 ;;
    --perturbation_ids)      P_RAW="$2"; PERT_IDS=($(expand_ids "$2")); shift 2 ;;
    --run_id)                RUN_ID="$2";                             shift 2 ;;
    --model_type)            MODEL_TYPE="$2";                         shift 2 ;;
    --host)                  HOST="$2";                               shift 2 ;;
    --spp)                   SPP="$2";                                shift 2 ;;
    --debug)                 DEBUG=true;                              shift 1 ;;
    --rendering_mode)        RENDERING_MODE="$2";                     shift 2 ;;
    --multi-view)            MULTI_VIEW_FLAG="--multi-view";          shift 1 ;;
    --resume)                RESUME=true; RESUME_FLAG="--resume";     shift 1 ;;
    --no_render)             NO_RENDER_FLAG="--no_render";            shift 1 ;;
    --no_record)             NO_RECORD_FLAG="--no_record";            shift 1 ;;
    --robot)                 ROBOT_FLAG="--robot $2";                 shift 2 ;;
    --og_lite)               OG_LITE=true;                            shift 1 ;;
    *) shift ;;
  esac
done

[ ${#TASK_IDS[@]} -eq 0 ] && { T_RAW="0-9";  TASK_IDS=($(expand_ids "$T_RAW")); }
[ ${#PERT_IDS[@]} -eq 0 ] && { P_RAW="0-15"; PERT_IDS=($(expand_ids "$P_RAW")); }
[ -z "$EXPERIMENT_NAME" ] && \
  EXPERIMENT_NAME="t${T_RAW//,/_}_p${P_RAW//,/_}_s${MAX_STEPS}_h${HORIZON}_r${REPEATS}"

#--- Setup ---------------------------------------------------------------------

extract_task_pert_names

# TODO: try commenting these out to see if it runs faster on clara??
setup_hf_cache
compute_og_lite_bind
compute_model_name

port=$BASE_PORT
VIDEO_DIR="logs/$EXPERIMENT_NAME/$MODEL_NAME/$RUN_ID/videos"

[ -n "$TASK_CFG_PATH" ] && TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH" || TASK_CFG_ARG=""

[ "$DEBUG" = "false" ] && start_policy_server

cd "$REALM_ROOT" || exit
setup_job_dirs

#--- Evaluation loop -----------------------------------------------------------

for i in "${TASK_IDS[@]}"; do
  for j in "${PERT_IDS[@]}"; do
    if [ "$RESUME" = "true" ]; then
      TASK_NAME=${ALL_TASKS[$i]}
      PERT_NAME=${ALL_PERTS[$j]}
      COUNT=$(ls "$VIDEO_DIR/${TASK_NAME}_${PERT_NAME}_"*.mp4 2>/dev/null | wc -l)
      if [ "$COUNT" -ge "$REPEATS" ]; then
        echo "Skipping Task $i ($TASK_NAME) Pert $j ($PERT_NAME): Found $COUNT/$REPEATS videos."
        continue
      fi
    fi

    echo "Starting evaluation for Task $i, Perturbation $j..."

    apptainer_eval "python examples/02_evaluate.py \
      --perturbation_id $j \
      --task_id $i \
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
  done
done

EXIT_CODE=$?

[ "$DEBUG" = "false" ] && stop_policy_server
cleanup_job_dirs $EXIT_CODE "Sequential evaluation"
exit $EXIT_CODE
