#!/bin/bash
# Thread 2: exercise OG-lite's incremental contact cache for the first time.
#
# gm.INCREMENTAL_CONTACT_CACHE has been off everywhere, so the incremental fold in
# RigidContactAPI.update_contact_cache has never run against REALM. This is the correctness
# gate before any timing work: does a rollout complete with the fold on?
#
# Pass requires ALL of:
#   - exit 0
#   - no "row mismatch" / Traceback / Segmentation fault in the log
#   - all four artifacts written, each with a populated data row
# Exit 0 alone is NOT sufficient: the OG-lite failure mode asserts inside an Isaac callback and
# then segfaults, which can still leave a 0 on some paths.
#
#   INC=1 ./scripts/clara/interactive/t2_inc_on.sh   # fold on
#   INC=0 ./scripts/clara/interactive/t2_inc_on.sh   # control, fold off
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391

INC=${INC:-1}
RUN_ID=${RUN_ID:-inc_on}
MAX_STEPS=${MAX_STEPS:-2}

echo "### t2: INCREMENTAL_CONTACT_CACHE=$INC run_id=$RUN_ID max_steps=$MAX_STEPS"

# Print what the macros actually resolve to before booting a simulator, so a null test
# (flag silently not applied) cannot masquerade as a pass.
MODE=oglite REALM_INCREMENTAL_CONTACT_CACHE=$INC ./scripts/clara/interactive/rr \
  python -u scripts/clara/interactive/show_macros.py

MODE=oglite REALM_INCREMENTAL_CONTACT_CACHE=$INC ./scripts/clara/interactive/rr \
  python -u examples/02_evaluate.py \
    --task_id 0 --perturbation_id 0 --repeats 1 --max_steps "$MAX_STEPS" \
    --model_name debug --model_type debug --port 8000 \
    --experiment_name oglite_verify --run_id "$RUN_ID" \
    --log_dir /logs
echo "### eval exit: $?"
