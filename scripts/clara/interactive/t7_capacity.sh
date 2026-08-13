#!/bin/bash
# How many REALM envs fit on one GPU. See t7_env_capacity.py for what is and is not measured.
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
echo "### capacity probe: max_envs=${MAX_ENVS:-12} reserve=${RESERVE:-3000} robot=${ROBOT:-DROID_robolab}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODE=oglite ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t7_env_capacity.py \
    --max_envs "${MAX_ENVS:-12}" --reserve "${RESERVE:-3000}" --robot "${ROBOT:-DROID_robolab}"
echo "### capacity probe exit: $?"
