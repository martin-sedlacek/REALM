#!/bin/bash
# Thread 1: dump per-member scene state to find why apply_scene_fixes_from_cfg only takes in scene 0.
#
#   NUM_ENVS=2 ./scripts/clara/interactive/t1_probe.sh     # cross-scene naming + fix outcome (the key run)
#   NUM_ENVS=1 ./scripts/clara/interactive/t1_probe.sh     # discriminator: does the fix work without batching?
#
# Runs in MODE=stock, which is where the vector-env work was done. The bug is REALM-side and has
# nothing to do with the OG-lite fork.
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391

NUM_ENVS=${NUM_ENVS:-2}
TASK_ID=${TASK_ID:-0}
WARMUP_FLAG=""
[ "${WARMUP:-0}" = "1" ] && WARMUP_FLAG="--warmup"
FRAMES_FLAG=""
[ -n "${FRAMES_DIR:-}" ] && FRAMES_FLAG="--frames_dir $FRAMES_DIR"

echo "### t1 probe: num_envs=$NUM_ENVS task_id=$TASK_ID warmup=${WARMUP:-0}"

MODE=${MODE:-stock} ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t1_scene_probe.py \
    --num_envs "$NUM_ENVS" --task_id "$TASK_ID" $WARMUP_FLAG $FRAMES_FLAG
echo "### probe exit: $?"
