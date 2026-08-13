#!/bin/bash
# End-to-end check that the shipped defaults work with NO env vars set at all.
#
# As of 2026-08-13 realm/sim_config.py turns the OG-lite incremental contact cache ON by default and
# pins the proximity gate ON, so this is the configuration every eval now gets for free. Verify it
# the same way as any other run: all four artifacts, populated rows, no assert, no segfault.
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
RUN_ID=${RUN_ID:-defaults}
echo "### default-config check: no REALM_* env vars set"
MODE=oglite ./scripts/clara/interactive/rr python -u scripts/clara/interactive/show_macros.py
MODE=oglite ./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 --repeats 1 --max_steps "${MAX_STEPS:-20}" \
    --model_name debug --model_type debug --port 8000 \
    --experiment_name oglite_defaults --run_id "$RUN_ID" --log_dir /logs
echo "### eval exit: $?"
