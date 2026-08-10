#!/usr/bin/env bash
# Sequential driver for the full DreamZero task sweep.
# - Submits one `run_eval_sequential.sh` job per task (0..9).
# - Runs each perturbation set {0,2,13,15} inside the job.
# - Serialises jobs (one at a time) so they share a single DZ server.
# - Retries a task up to $MAX_RETRIES times on failure or missing artifacts.
# - Logs every action to $SWEEP_LOG for post-hoc review.
#
# Usage:
#   bash scripts/dreamzero_sweep_driver.sh --host <IP> --port <PORT> \
#        [--experiment_name NAME] [--max_retries N] [--tasks "0 1 2"]

set -eo pipefail

HOST=""
PORT=""
EXPERIMENT_NAME="dreamzero_full_sweep_$(date +%Y%m%d_%H%M%S)"
PERT_IDS="0,2,13,15"
REPEATS=10
MAX_STEPS=800
HORIZON=24
SPP=16
RENDERING_MODE="pt"
TASKS="0 1 2 3 4 5 6 7 8 9"
MAX_RETRIES=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)             HOST="$2"; shift 2 ;;
    --port)             PORT="$2"; shift 2 ;;
    --experiment_name)  EXPERIMENT_NAME="$2"; shift 2 ;;
    --max_retries)      MAX_RETRIES="$2"; shift 2 ;;
    --tasks)            TASKS="$2"; shift 2 ;;
    --pert_ids)         PERT_IDS="$2"; shift 2 ;;
    --repeats)          REPEATS="$2"; shift 2 ;;
    --max_steps)        MAX_STEPS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$HOST" || -z "$PORT" ]] && { echo "ERROR: --host and --port are required" >&2; exit 2; }

REALM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REALM_ROOT"

mkdir -p tmp/sweep_logs
SWEEP_LOG="$REALM_ROOT/tmp/sweep_logs/${EXPERIMENT_NAME}.log"
SWEEP_STATE="$REALM_ROOT/tmp/sweep_logs/${EXPERIMENT_NAME}.state"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$SWEEP_LOG"
}

# Wait until the given SLURM job leaves the queue. Returns nothing.
wait_for_job() {
  local jid="$1"
  while squeue -j "$jid" -h -o "%i" 2>/dev/null | grep -qx "$jid"; do
    sleep 60
  done
}

# Return the job's final sacct state ("COMPLETED", "FAILED", "CANCELLED", ...).
job_state() {
  local jid="$1"
  sacct -j "$jid" -n -o State --parsable2 2>/dev/null | head -n1 | awk '{print $1}'
}

# Check that the per-task artifacts exist and look non-trivial.
# Args: task_id run_dir
# Returns 0 on success, 1 on failure.
verify_artifacts() {
  local tid="$1"
  local run_dir="$2"
  # The task-name-to-id mapping mirrors eval.py:SUPPORTED_TASKS.
  local TASK_NAMES=(put_green_block_into_bowl put_banana_into_box rotate_marker \
                    rotate_mug pick_spoon pick_water_bottle stack_cubes push_switch \
                    open_drawer close_drawer)
  local tname="${TASK_NAMES[$tid]}"
  local reports_dir="$run_dir/reports"
  local videos_dir="$run_dir/videos"

  if [[ ! -d "$reports_dir" ]]; then
    log "  VERIFY FAIL: reports dir missing: $reports_dir"
    return 1
  fi
  local csv_count
  csv_count=$(ls "$reports_dir/${tname}_"*.csv 2>/dev/null | wc -l)
  if [[ "$csv_count" -lt 1 ]]; then
    log "  VERIFY FAIL: no CSVs for ${tname} in $reports_dir"
    return 1
  fi
  if [[ ! -f "$videos_dir/${tname}.parquet" ]]; then
    log "  VERIFY FAIL: video parquet missing: $videos_dir/${tname}.parquet"
    return 1
  fi
  log "  VERIFY OK: $csv_count CSV(s) + video parquet for ${tname}"
  return 0
}

log "========================================"
log "DreamZero sweep driver starting"
log "  host=$HOST  port=$PORT"
log "  experiment_name=$EXPERIMENT_NAME"
log "  tasks=$TASKS"
log "  perturbations=$PERT_IDS"
log "  repeats=$REPEATS  max_steps=$MAX_STEPS  horizon=$HORIZON  spp=$SPP"
log "  max_retries=$MAX_RETRIES"
log "========================================"

for TID in $TASKS; do
  success=0
  run_dir=""
  for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
    log "[task=$TID attempt=$attempt] submitting sbatch..."
    JOB_ID=$(sbatch --parsable scripts/clara/run_eval_sequential.sh \
      --model_type dreamzero \
      --host "$HOST" --base_port "$PORT" \
      --task_ids "$TID" \
      --perturbation_ids "$PERT_IDS" \
      --experiment_name "$EXPERIMENT_NAME" \
      --multi-view --rendering_mode "$RENDERING_MODE" \
      --max_steps "$MAX_STEPS" --repeats "$REPEATS" \
      --horizon "$HORIZON" --spp "$SPP")
    log "[task=$TID attempt=$attempt] submitted JOB_ID=$JOB_ID"
    echo "task=$TID attempt=$attempt job=$JOB_ID started=$(date '+%Y-%m-%d_%H:%M:%S')" >> "$SWEEP_STATE"

    wait_for_job "$JOB_ID"
    state=$(job_state "$JOB_ID")
    log "[task=$TID attempt=$attempt] job $JOB_ID final state: $state"
    echo "task=$TID attempt=$attempt job=$JOB_ID ended=$(date '+%Y-%m-%d_%H:%M:%S') state=$state" >> "$SWEEP_STATE"

    # Locate the run directory — pick the newest matching run_id under the experiment.
    run_dir=$(ls -td "$REALM_ROOT/logs/$EXPERIMENT_NAME/dreamzero/"*/ 2>/dev/null | head -n1 | sed 's:/*$::')
    if [[ -z "$run_dir" ]]; then
      log "  no run dir found under logs/$EXPERIMENT_NAME/dreamzero/"
      continue
    fi
    log "  run_dir=$run_dir"

    if [[ "$state" == "COMPLETED" ]] && verify_artifacts "$TID" "$run_dir"; then
      log "[task=$TID] SUCCESS on attempt $attempt"
      success=1
      break
    else
      log "[task=$TID] attempt $attempt FAILED (state=$state, verify=$?)"
    fi
  done

  if [[ "$success" -ne 1 ]]; then
    log "!!!! [task=$TID] GAVE UP after $((MAX_RETRIES + 1)) attempts !!!!"
  fi
done

log "========================================"
log "Sweep driver finished."
log "State file: $SWEEP_STATE"
log "========================================"
