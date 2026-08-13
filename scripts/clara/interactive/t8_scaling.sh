#!/bin/bash
# Profiled vector-env scaling in the max-juice configuration (ROD on, OG-lite fold + gate on).
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
N=${NUM_ENVS:-16}
echo "### max-juice scaling: num_envs=$N steps=${STEPS:-96} horizon=${HORIZON:-8} robot=${ROBOT:-DROID_robolab_v2}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODE=oglite ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t8_vec_scaling.py \
    --num_envs "$N" --steps "${STEPS:-96}" --horizon "${HORIZON:-8}" \
    --robot "${ROBOT:-DROID_robolab_v2}" --out "/app/tmp/interactive/prof/scaling_n${N}.json" \
    ${PRE_RENDER_MODE:+--pre_render_mode "$PRE_RENDER_MODE"}
echo "### scaling exit: $?"
