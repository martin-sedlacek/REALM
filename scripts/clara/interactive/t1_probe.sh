#!/bin/bash
# Thread 1: dump per-member scene state to find why apply_scene_fixes_from_cfg only takes in scene 0.
#
#   NUM_ENVS=2 ./scripts/clara/interactive/t1_probe.sh     # cross-scene naming + fix outcome (the key run)
#   NUM_ENVS=1 ./scripts/clara/interactive/t1_probe.sh     # discriminator: does the fix work without batching?
#
# Runs in MODE=stock, which is where the vector-env work was done. The bug is REALM-side and has
# nothing to do with the OG-lite fork.
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1

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
