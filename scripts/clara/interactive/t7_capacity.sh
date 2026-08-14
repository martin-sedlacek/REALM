#!/bin/bash
# How many REALM envs fit on one GPU. See t7_env_capacity.py for what is and is not measured.
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1
echo "### capacity probe: max_envs=${MAX_ENVS:-12} reserve=${RESERVE:-3000} robot=${ROBOT:-DROID_robolab_v2}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODE=oglite ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t7_env_capacity.py \
    --max_envs "${MAX_ENVS:-12}" --reserve "${RESERVE:-3000}" --robot "${ROBOT:-DROID_robolab_v2}"
echo "### capacity probe exit: $?"
