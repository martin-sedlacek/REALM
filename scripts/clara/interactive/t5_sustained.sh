#!/bin/bash
# Sustained stepping at a given num_envs, with GPU memory around play(). Needs MODE=oglite.
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
echo "### sustained: num_envs=${NUM_ENVS:-4} steps=${STEPS:-200} robot=${ROBOT:-DROID_robolab}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODE=oglite ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t5_vec_sustained.py \
    --num_envs "${NUM_ENVS:-4}" --steps "${STEPS:-200}" --robot "${ROBOT:-DROID_robolab}" \
    --check_every "${CHECK_EVERY:-50}"
echo "### sustained exit: $?"
