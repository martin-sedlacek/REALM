#!/bin/bash
# Exact reproduction of the run that produced docs/vector_env/frames/, so the new montages are
# directly comparable to the committed ones.
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1
NUM_ENVS=${NUM_ENVS:-4}
OUT=${OUT:-/logs/vector_first_frames_190155}
echo "### t1 frames: num_envs=$NUM_ENVS out=$OUT"
MODE=${MODE:-stock} ./scripts/clara/interactive/rr \
  python -u examples/03_vector_first_frames.py \
    --num_envs "$NUM_ENVS" --task_id 0 --out_dir "$OUT"
echo "### frames exit: $?"
