#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

unset EXPERIMENT_NAME T_RAW P_RAW TASK_IDS PERT_IDS
mkdir -p "$REALM_ROOT/tmp"

#--- Argument parsing ----------------------------------------------------------

OG_LITE_FLAG=""

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
    --model_type)            MODEL_TYPE="$2";                         shift 2 ;;
    --host)                  HOST="$2";                               shift 2 ;;
    --spp)                   SPP="$2";                                shift 2 ;;
    --debug)                 DEBUG=true;                              shift 1 ;;
    --multi-view)            MULTI_VIEW_FLAG="--multi-view";          shift 1 ;;
    --no_render)             NO_RENDER_FLAG="--no_render";            shift 1 ;;
    --no_record)             NO_RECORD_FLAG="--no_record";            shift 1 ;;
    --run-id)                RUN_ID="$2";                             shift 2 ;;
    --resume)                RESUME=true; RESUME_FLAG="--resume";     shift 1 ;;
    --rendering_mode)        RENDERING_MODE="$2";                     shift 2 ;;
    --robot)                 ROBOT_FLAG="--robot $2";                 shift 2 ;;
    --og_lite)               OG_LITE_FLAG="--og_lite";                shift 1 ;;
    *) shift ;;
  esac
done

[ ${#TASK_IDS[@]} -eq 0 ] && { T_RAW="0-9";  TASK_IDS=($(expand_ids "$T_RAW")); }
[ ${#PERT_IDS[@]} -eq 0 ] && { P_RAW="0-15"; PERT_IDS=($(expand_ids "$P_RAW")); }
[ -z "$EXPERIMENT_NAME" ] && \
  EXPERIMENT_NAME="t${T_RAW//,/_}_p${P_RAW//,/_}_s${MAX_STEPS}_h${HORIZON}_r${REPEATS}"

#--- Metadata ------------------------------------------------------------------

METADATA_DIR="logs/$EXPERIMENT_NAME"
mkdir -p "$METADATA_DIR"
{
  echo "{"
  echo "  \"max_steps\": $MAX_STEPS,"
  echo "  \"horizon\": $HORIZON,"
  echo "  \"repeats\": $REPEATS,"
  echo "  \"task_ids\": [${T_RAW}]",
  echo "  \"perturbation_ids\": [${P_RAW}]"
  echo "}"
} > "$METADATA_DIR/metadata.json"

#--- Setup ---------------------------------------------------------------------

extract_task_pert_names
compute_model_name

VIDEO_DIR="logs/$EXPERIMENT_NAME/$MODEL_NAME/$RUN_ID/videos"

[ "$DEBUG" = "true" ] && DEBUG_FLAG="--debug" || DEBUG_FLAG=""
[ -n "$TASK_CFG_PATH" ] && TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH" || TASK_CFG_ARG=""

#--- Submit sbatch jobs --------------------------------------------------------

for i in "${TASK_IDS[@]}"; do
  for j in "${PERT_IDS[@]}"; do
    TASK_PORT=$((BASE_PORT + 100 * i + j))

    sbatch --job-name="realm-${EXPERIMENT_NAME}-t${i}-p${j}" \
      "$SCRIPT_DIR/run_eval_sequential.sh" \
      --task_ids "$i" \
      --perturbation_ids "$j" \
      --repeats "$REPEATS" \
      --max_steps "$MAX_STEPS" \
      --horizon "$HORIZON" \
      --policy_config "$POLICY_CONFIG" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --policy_run_dir "$POLICY_RUN_DIR" \
      --base_port "$TASK_PORT" \
      --experiment_name "$EXPERIMENT_NAME" \
      --run_id "$RUN_ID" \
      --model_type "$MODEL_TYPE" \
      --rendering_mode "$RENDERING_MODE" \
      --host "$HOST" \
      --spp "$SPP" \
      $TASK_CFG_ARG \
      $DEBUG_FLAG \
      $MULTI_VIEW_FLAG \
      $RESUME_FLAG \
      $NO_RENDER_FLAG \
      $NO_RECORD_FLAG \
      $ROBOT_FLAG \
      $OG_LITE_FLAG
  done
done
