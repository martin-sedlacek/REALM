#!/usr/bin/env bash
# Wait for each DreamZero server to come up, then submit its eval job.
# Each task gets its own server → fully parallel execution.
# Includes retry logic (up to MAX_RETRIES per task).
set -eo pipefail

cd /mnt/home_lustre/sedlam56/projects/REALM

EXPERIMENT_NAME="dreamzero_full_sweep"
PERT_IDS="0,2,13,15"
REPEATS=10
MAX_STEPS=800
HORIZON=24
SPP=16
RENDERING_MODE="pt"
MAX_RETRIES=2

LOG_DIR="tmp/sweep_logs"
mkdir -p "$LOG_DIR"
PLOG="$LOG_DIR/parallel_sweep.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PLOG"; }

# task → server mapping (task:job_id:ip:port)
declare -A SERVERS
SERVERS[1]="60613:192.168.0.6:11112"
SERVERS[2]="60614:192.168.0.6:11113"
SERVERS[3]="60615:192.168.0.6:11114"
SERVERS[4]="60617:192.168.0.3:11115"
SERVERS[5]="60616:192.168.0.3:11116"
SERVERS[6]="60619:192.168.0.4:11117"
SERVERS[7]="60620:192.168.0.4:11118"
SERVERS[8]="60618:192.168.0.4:11119"
# task 9 will be added once job 60621 starts

wait_for_server() {
  local jid="$1"
  local logfile="tmp/dreamzero_logs/server_${jid}.log"
  log "  waiting for server job $jid..."
  while ! grep -q "server listening" "$logfile" 2>/dev/null; do
    # Check job is still alive
    if ! squeue -j "$jid" -h -o "%i" 2>/dev/null | grep -qx "$jid"; then
      log "  ERROR: server job $jid is no longer running!"
      return 1
    fi
    sleep 10
  done
  log "  server job $jid is ready"
  return 0
}

wait_for_eval_job() {
  local jid="$1"
  while squeue -j "$jid" -h -o "%i" 2>/dev/null | grep -qx "$jid"; do
    sleep 60
  done
}

job_state() {
  sacct -j "$1" -n -o State --parsable2 2>/dev/null | head -n1 | awk '{print $1}'
}

run_task() {
  local tid="$1"
  local host="$2"
  local port="$3"

  for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
    log "[task=$tid attempt=$attempt] submitting eval..."
    local EVAL_JID
    EVAL_JID=$(sbatch --parsable scripts/clara/run_eval_sequential.sh \
      --model_type dreamzero \
      --host "$host" --base_port "$port" \
      --task_ids "$tid" \
      --perturbation_ids "$PERT_IDS" \
      --experiment_name "$EXPERIMENT_NAME" \
      --multi-view --rendering_mode "$RENDERING_MODE" \
      --max_steps "$MAX_STEPS" --repeats "$REPEATS" \
      --horizon "$HORIZON" --spp "$SPP")
    log "[task=$tid attempt=$attempt] submitted JOB_ID=$EVAL_JID"

    wait_for_eval_job "$EVAL_JID"
    local state
    state=$(job_state "$EVAL_JID")
    log "[task=$tid attempt=$attempt] final state: $state"

    if [[ "$state" == "COMPLETED" ]]; then
      log "[task=$tid] SUCCESS"
      return 0
    fi
    log "[task=$tid attempt=$attempt] FAILED — retrying..."
  done
  log "!!!! [task=$tid] GAVE UP after $((MAX_RETRIES + 1)) attempts !!!!"
  return 1
}

log "========================================"
log "Parallel sweep launcher starting"
log "  tasks: 1-8 (9 pending)"
log "  experiment: $EXPERIMENT_NAME"
log "========================================"

# Launch each task in a background subshell
for tid in "${!SERVERS[@]}"; do
  IFS=: read -r sjid host port <<< "${SERVERS[$tid]}"
  (
    if wait_for_server "$sjid"; then
      run_task "$tid" "$host" "$port"
    else
      log "[task=$tid] server $sjid failed to start, skipping"
    fi
  ) &
  log "Spawned background worker for task $tid (server $sjid, ${host}:${port})"
done

# Handle task 9 separately — wait for job 60621 to get a node
(
  log "[task=9] waiting for server job 60621 to start..."
  while true; do
    if squeue -j 60621 -h -o "%i" 2>/dev/null | grep -qx 60621; then
      logfile="tmp/dreamzero_logs/server_60621.log"
      if [[ -f "$logfile" ]]; then
        ip=$(grep "node ip" "$logfile" 2>/dev/null | awk '{print $NF}')
        port=$(grep " port " "$logfile" 2>/dev/null | awk '{print $NF}')
        if [[ -n "$ip" && -n "$port" ]]; then
          log "[task=9] server 60621 allocated: ${ip}:${port}"
          if wait_for_server 60621; then
            run_task 9 "$ip" "$port"
          else
            log "[task=9] server 60621 failed to start, skipping"
          fi
          break
        fi
      fi
    else
      # Job no longer in queue — check if it ever ran
      state=$(job_state 60621)
      if [[ "$state" == "CANCELLED" || "$state" == "FAILED" ]]; then
        log "[task=9] server job 60621 $state before starting. Skipping task 9."
        break
      fi
    fi
    sleep 15
  done
) &
log "Spawned background worker for task 9 (server 60621, pending)"

log "All workers launched. Waiting for completion..."
wait
log "========================================"
log "Parallel sweep launcher finished."
log "========================================"
