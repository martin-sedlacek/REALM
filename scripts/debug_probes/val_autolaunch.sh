#!/bin/bash
# Wait for the first of the reserved allocations to start, then fire the validation chain on it.
# Isaac's teardown hang makes an interactive tail unreliable, so this is detached and logs to file.
#   usage: val_autolaunch.sh "<jobid> [jobid...]" [phase ...]
set -uo pipefail
IDS=${1:?usage: val_autolaunch.sh "<jobids>" [phases]}
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG=/mnt/home_lustre/sedlam56/projects/REALM/logs/gripper_squeeze/val_autolaunch.log
echo "=== $(date -Is) waiting for one of: $IDS ===" >> "$LOG"
while true; do
  for j in $IDS; do
    st=$(squeue -j "$j" -h -o %T 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      echo "=== $(date -Is) job $j RUNNING on $(squeue -j "$j" -h -o %R) -- launching chain ===" >> "$LOG"
      exec "$HERE/scripts/debug_probes/val_run_chain.sh" "$j" "$@" >> "$LOG" 2>&1
    fi
  done
  sleep 30
done
