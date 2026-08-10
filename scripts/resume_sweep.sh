#!/usr/bin/env bash
# Wait for job 60610 (task 0) to finish, then drive tasks 1-9.
set -eo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM

echo "[$(date)] Waiting for job 60610 (task 0) to finish..."
while squeue -j 60610 -h -o "%i" 2>/dev/null | grep -qx 60610; do
  sleep 60
done
echo "[$(date)] Job 60610 finished. Launching driver for tasks 1-9."

exec bash scripts/dreamzero_sweep_driver.sh \
  --host 192.168.0.3 --port 11111 \
  --experiment_name dreamzero_full_sweep \
  --tasks "1 2 3 4 5 6 7 8 9" \
  --max_retries 2
