#!/bin/bash
# End-to-end check that the shipped defaults work with NO env vars set at all.
#
# As of 2026-08-13 realm/sim_config.py turns the OG-lite incremental contact cache ON by default and
# pins the proximity gate ON, so this is the configuration every eval now gets for free. Verify it
# the same way as any other run: all four artifacts, populated rows, no assert, no segfault.
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1
RUN_ID=${RUN_ID:-defaults}
echo "### default-config check: no REALM_* env vars set"
MODE=oglite ./scripts/clara/interactive/rr python -u scripts/clara/interactive/show_macros.py
MODE=oglite ./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 --repeats 1 --max_steps "${MAX_STEPS:-20}" \
    --model_name debug --model_type debug --port 8000 \
    --experiment_name oglite_defaults --run_id "$RUN_ID" --log_dir /logs
echo "### eval exit: $?"
