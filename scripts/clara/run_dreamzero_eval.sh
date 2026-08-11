#!/bin/bash
#SBATCH --job-name realm-dreamzero
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 01-00:00:00
#
# All-in-one DreamZero evaluation script.
#
# This script is the single entry point for running DreamZero evaluations on
# the CLARA cluster. It:
#   1. Allocates 2x H200 GPUs and launches the DreamZero inference server
#   2. Waits until the server is loaded and listening
#   3. Captures the server node's IP automatically
#   4. Runs the sequential evaluation loop (same as run_eval_sequential.sh)
#   5. Cleans up the server on exit (success or failure)
#
# Usage:
#   sbatch scripts/clara/run_dreamzero_eval.sh \
#       --task_ids 0 --perturbation_ids 0,2,13,15 \
#       --experiment_name my_exp --repeats 10
#
#   # With custom checkpoint and port:
#   sbatch scripts/clara/run_dreamzero_eval.sh \
#       --task_ids 4 --perturbation_ids 0 \
#       --dz_checkpoint checkpoints/dreamzero_droid_3epoch_h200 \
#       --base_port 12345
#
# DreamZero-specific defaults applied automatically:
#   --model_type dreamzero  (forced)
#   --multi-view            (forced — DreamZero requires second camera)
#   --rendering_mode pt     (default, overridable)
#   --horizon 24            (default, overridable)
#   --spp 16                (default, overridable)

#--- Source helpers -------------------------------------------------------------

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts/clara"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/apptainer.sh"

#--- DreamZero defaults (override common.sh defaults where needed) -------------

MODEL_TYPE="dreamzero"
MULTI_VIEW_FLAG="--multi-view"
RENDERING_MODE="pt"
HORIZON=24
SPP=16

DZ_CHECKPOINT="checkpoints/DreamZero-DROID"
DZ_CONDA_ENV="dreamzero"
DZ_TIME="30:00:00"
DZ_DREAMZERO_DIR=""    # auto-detected below
DZ_SERVER_TIMEOUT=900  # seconds to wait for server to load (15 min)

#--- Argument parsing ----------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
  case $1 in
    # Eval args (same as run_eval_sequential.sh)
    --base_port|--base-port) BASE_PORT="$2";                          shift 2 ;;
    --max_steps)             MAX_STEPS="$2";                          shift 2 ;;
    --horizon)               HORIZON="$2";                            shift 2 ;;
    --repeats)               REPEATS="$2";                            shift 2 ;;
    --experiment_name)       EXPERIMENT_NAME="$2";                    shift 2 ;;
    --task_ids)              T_RAW="$2"; TASK_IDS=($(expand_ids "$2")); shift 2 ;;
    --task_cfg_path)         TASK_CFG_PATH="$2";                      shift 2 ;;
    --perturbation_ids)      P_RAW="$2"; PERT_IDS=($(expand_ids "$2")); shift 2 ;;
    --run_id)                RUN_ID="$2";                             shift 2 ;;
    --spp)                   SPP="$2";                                shift 2 ;;
    --rendering_mode)        RENDERING_MODE="$2";                     shift 2 ;;
    --resume)                RESUME=true; RESUME_FLAG="--resume";     shift 1 ;;
    --no_render)             NO_RENDER_FLAG="--no_render";            shift 1 ;;
    --no_record)             NO_RECORD_FLAG="--no_record";            shift 1 ;;
    --robot)                 ROBOT_FLAG="--robot $2";                 shift 2 ;;
    --og_lite)               OG_LITE=true;                            shift 1 ;;

    # DreamZero server args
    --dz_checkpoint)         DZ_CHECKPOINT="$2";                      shift 2 ;;
    --dz_conda_env)          DZ_CONDA_ENV="$2";                       shift 2 ;;
    --dz_time)               DZ_TIME="$2";                            shift 2 ;;
    --dz_dir)                DZ_DREAMZERO_DIR="$2";                   shift 2 ;;
    --dz_server_timeout)     DZ_SERVER_TIMEOUT="$2";                  shift 2 ;;
    *) shift ;;
  esac
done

[ ${#TASK_IDS[@]} -eq 0 ] && { T_RAW="0-9";  TASK_IDS=($(expand_ids "$T_RAW")); }
[ ${#PERT_IDS[@]} -eq 0 ] && { P_RAW="0-15"; PERT_IDS=($(expand_ids "$P_RAW")); }
[ -z "$EXPERIMENT_NAME" ] && \
  EXPERIMENT_NAME="dz_t${T_RAW//,/_}_p${P_RAW//,/_}_s${MAX_STEPS}_h${HORIZON}_r${REPEATS}"

#--- Auto-detect dreamzero repo if not set -------------------------------------

if [[ -z "$DZ_DREAMZERO_DIR" ]]; then
  for candidate in \
      "$REALM_ROOT/../dreamzero" \
      "$HOME/projects/dreamzero" \
      "/mnt/home_lustre/$USER/projects/dreamzero"; do
    if [[ -d "$candidate" ]]; then
      DZ_DREAMZERO_DIR="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi
if [[ -z "$DZ_DREAMZERO_DIR" || ! -d "$DZ_DREAMZERO_DIR" ]]; then
  echo "ERROR: dreamzero repo not found. Pass --dz_dir /path/to/dreamzero" >&2
  exit 1
fi

#--- Logging helpers -----------------------------------------------------------

DZ_LOG_DIR="$REALM_ROOT/tmp/dreamzero_logs"
mkdir -p "$DZ_LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
  echo "[$(ts)] [dreamzero-eval] $*"
}

log_section() {
  echo ""
  echo "========================================================================"
  echo " $(ts) | $*"
  echo "========================================================================"
  echo ""
}

#--- Setup ---------------------------------------------------------------------

extract_task_pert_names
setup_hf_cache
compute_og_lite_bind
compute_model_name

port=$BASE_PORT

[ -n "$TASK_CFG_PATH" ] && TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH" || TASK_CFG_ARG=""

cd "$REALM_ROOT" || exit
setup_job_dirs

VIDEO_DIR="logs/$EXPERIMENT_NAME/$MODEL_NAME/$RUN_ID/videos"

#===============================================================================
# PHASE 1: Launch the DreamZero inference server on 2x H200
#===============================================================================

log_section "PHASE 1: Launching DreamZero server"

DZ_SERVER_LOG="$DZ_LOG_DIR/server_eval${SLURM_JOB_ID}.log"
DZ_SERVER_JOB_ID=""

log "Configuration:"
log "  checkpoint    : $DZ_CHECKPOINT"
log "  conda env     : $DZ_CONDA_ENV"
log "  dreamzero dir : $DZ_DREAMZERO_DIR"
log "  port          : $port"
log "  server time   : $DZ_TIME"
log "  server log    : $DZ_SERVER_LOG"
log "  load timeout  : ${DZ_SERVER_TIMEOUT}s"

# Cleanup function: always kill the server on exit.
cleanup_server() {
  local exit_code=$?
  if [[ -n "$DZ_SERVER_JOB_ID" ]]; then
    log "Cleaning up DreamZero server (job $DZ_SERVER_JOB_ID)..."
    scancel "$DZ_SERVER_JOB_ID" 2>/dev/null && \
      log "Server job $DZ_SERVER_JOB_ID cancelled." || \
      log "Server job $DZ_SERVER_JOB_ID already finished."
  fi
  cleanup_job_dirs $exit_code "DreamZero evaluation"
}
trap cleanup_server EXIT

# Submit the server as an independent sbatch job.
# (Using sbatch instead of srun because SLURM does not permit nested step
# allocations across different gres specs — our parent job is on l40s and
# we need 2x h200 for the server.)
log "Requesting 2x H200 allocation for DreamZero server..."

DZ_SERVER_SUBMIT_SCRIPT="$REALM_ROOT/tmp/dreamzero_logs/server_eval${SLURM_JOB_ID}.sbatch"
mkdir -p "$(dirname "$DZ_SERVER_SUBMIT_SCRIPT")"
cat > "$DZ_SERVER_SUBMIT_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --partition=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=250G
#SBATCH --gpu-bind=closest
#SBATCH --time=$DZ_TIME
#SBATCH --job-name=dz-srv-${SLURM_JOB_ID}
#SBATCH --output=$DZ_SERVER_LOG

bash "$REALM_ROOT/scripts/run_dreamzero_server.sh" \\
    --port "$port" \\
    --checkpoint "$DZ_CHECKPOINT" \\
    --dreamzero-dir "$DZ_DREAMZERO_DIR" \\
    --conda-env "$DZ_CONDA_ENV"
EOF

DZ_SERVER_JOB_ID=$(sbatch --parsable "$DZ_SERVER_SUBMIT_SCRIPT")

if [[ -z "$DZ_SERVER_JOB_ID" ]]; then
  log "ERROR: failed to submit DZ server sbatch."
  exit 1
fi

# We don't need DZ_SERVER_PID for an sbatch job — use squeue to track liveness.
DZ_SERVER_PID=""
log "Server SLURM job ID: $DZ_SERVER_JOB_ID"

#===============================================================================
# PHASE 2: Wait for server to be ready
#===============================================================================

log_section "PHASE 2: Waiting for DreamZero server to load"

elapsed=0
poll_interval=10
last_status=""

while [[ $elapsed -lt $DZ_SERVER_TIMEOUT ]]; do
  # Check if the server SLURM job is still alive (queued or running).
  if ! squeue -j "$DZ_SERVER_JOB_ID" -h -o "%i" 2>/dev/null | grep -qx "$DZ_SERVER_JOB_ID"; then
    sstate=$(sacct -j "$DZ_SERVER_JOB_ID" -n -o State --parsable2 2>/dev/null | head -n1 | awk '{print $1}')
    log "ERROR: DZ server SLURM job $DZ_SERVER_JOB_ID is no longer queued/running (state=$sstate)."
    log "Server log tail:"
    tail -20 "$DZ_SERVER_LOG" 2>/dev/null | while IFS= read -r line; do log "  | $line"; done
    exit 1
  fi

  # Check log for progress milestones.
  if [[ -f "$DZ_SERVER_LOG" ]]; then
    if grep -q "server listening" "$DZ_SERVER_LOG" 2>/dev/null; then
      log "Server is listening! Loading complete."
      break
    fi

    # Report loading progress.
    new_status=""
    if grep -q "Loading shard" "$DZ_SERVER_LOG" 2>/dev/null; then
      loaded=$(grep -c "Loading shard" "$DZ_SERVER_LOG" 2>/dev/null)
      new_status="loading_shards ($loaded shards loaded)"
    elif grep -q "loading model" "$DZ_SERVER_LOG" 2>/dev/null; then
      new_status="building_model"
    elif grep -q "Worker loop started" "$DZ_SERVER_LOG" 2>/dev/null; then
      new_status="worker_ready"
    elif grep -q "Loading model" "$DZ_SERVER_LOG" 2>/dev/null; then
      new_status="starting_load"
    elif grep -q "CLIENT ENDPOINT" "$DZ_SERVER_LOG" 2>/dev/null; then
      new_status="node_allocated"
    fi

    if [[ -n "$new_status" && "$new_status" != "$last_status" ]]; then
      log "Server status: $new_status"
      last_status="$new_status"
    fi
  else
    if [[ $((elapsed % 30)) -eq 0 ]]; then
      log "Waiting for server log to appear (SLURM allocation pending)... (${elapsed}s)"
    fi
  fi

  sleep "$poll_interval"
  elapsed=$((elapsed + poll_interval))
done

# Verify server is actually listening.
if ! grep -q "server listening" "$DZ_SERVER_LOG" 2>/dev/null; then
  log "ERROR: Server did not become ready within ${DZ_SERVER_TIMEOUT}s."
  log "Server log tail:"
  tail -30 "$DZ_SERVER_LOG" 2>/dev/null | while IFS= read -r line; do log "  | $line"; done
  exit 1
fi

# Extract the server's IP from the log.
DZ_HOST=$(grep "node ip" "$DZ_SERVER_LOG" | awk '{print $NF}' | head -n1)
if [[ -z "$DZ_HOST" ]]; then
  log "ERROR: Could not extract server IP from log."
  exit 1
fi

HOST="$DZ_HOST"

log_section "PHASE 2 COMPLETE — Server ready"
log "  Server endpoint : ${HOST}:${port}"
log "  Server job ID   : ${DZ_SERVER_JOB_ID}"
log "  Server log      : ${DZ_SERVER_LOG}"

#===============================================================================
# PHASE 3: Run the sequential evaluation loop
#===============================================================================

log_section "PHASE 3: Running evaluation"
log "  tasks           : ${TASK_IDS[*]}"
log "  perturbations   : ${PERT_IDS[*]}"
log "  repeats         : $REPEATS"
log "  max_steps       : $MAX_STEPS"
log "  horizon         : $HORIZON"
log "  spp             : $SPP"
log "  rendering_mode  : $RENDERING_MODE"
log "  experiment_name : $EXPERIMENT_NAME"
log "  run_id          : $RUN_ID"
log "  model_name      : $MODEL_NAME"
log "  host            : $HOST"
log "  port            : $port"
log "  video_dir       : $VIDEO_DIR"
log ""

for i in "${TASK_IDS[@]}"; do
  for j in "${PERT_IDS[@]}"; do
    TASK_NAME=${ALL_TASKS[$i]}
    PERT_NAME=${ALL_PERTS[$j]}

    if [ "$RESUME" = "true" ]; then
      COUNT=$(ls "$VIDEO_DIR/${TASK_NAME}_${PERT_NAME}_"*.mp4 2>/dev/null | wc -l)
      if [ "$COUNT" -ge "$REPEATS" ]; then
        log "Skipping Task $i ($TASK_NAME) Pert $j ($PERT_NAME): Found $COUNT/$REPEATS videos."
        continue
      fi
    fi

    log "Starting Task $i ($TASK_NAME), Perturbation $j ($PERT_NAME)..."

    # Verify the DZ server SLURM job is still running before each combo.
    if ! squeue -j "$DZ_SERVER_JOB_ID" -h -o "%T" 2>/dev/null | grep -qx "RUNNING"; then
      log "ERROR: DreamZero server SLURM job $DZ_SERVER_JOB_ID is no longer RUNNING!"
      log "Server log tail:"
      tail -20 "$DZ_SERVER_LOG" 2>/dev/null | while IFS= read -r line; do log "  | $line"; done
      exit 1
    fi

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

    EVAL_EXIT=$?
    if [ $EVAL_EXIT -ne 0 ]; then
      log "WARNING: Eval returned exit code $EVAL_EXIT for Task $i Pert $j"
    else
      log "Completed Task $i ($TASK_NAME), Perturbation $j ($PERT_NAME)"
    fi
  done
done

EXIT_CODE=$?

log_section "PHASE 3 COMPLETE — Evaluation finished (exit code $EXIT_CODE)"
log "Results: logs/$EXPERIMENT_NAME/$MODEL_NAME/$RUN_ID/"

# Cleanup handled by the EXIT trap (cleanup_server).
exit $EXIT_CODE
