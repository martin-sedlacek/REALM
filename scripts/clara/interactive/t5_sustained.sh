#!/bin/bash
# Sustained stepping at a given num_envs, with GPU memory around play(). Needs MODE=oglite.
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1
echo "### sustained: num_envs=${NUM_ENVS:-4} steps=${STEPS:-200} robot=${ROBOT:-DROID_robolab_v2}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODE=oglite ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/t5_vec_sustained.py \
    --num_envs "${NUM_ENVS:-4}" --steps "${STEPS:-200}" --robot "${ROBOT:-DROID_robolab_v2}" \
    --check_every "${CHECK_EVERY:-50}"
echo "### sustained exit: $?"
