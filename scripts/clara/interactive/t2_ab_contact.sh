#!/bin/bash
# Interleaved A/B of OG-lite's incremental contact cache, under a realistic pi0.5 workload.
#
# Why pi0.5 and not --model_type debug: debug returns a CONSTANT action (realm/inference/client.py:33),
# so the gripper never touches anything. The perf doc measured the contact cache as bimodal --
# ~23-28 ms on most steps but ~300 ms on ~28% of them, the spikes appearing exactly when the gripper
# contacts an object. A debug A/B would only ever measure the cheap mode, i.e. the regime where the
# fold matters least.
#
# Protocol, from docs/perf/og391_step_profile.md's own post-mortem on the fork comparison:
#   - interleave conditions (0,1,0,1,...), never run all of one side then all of the other
#   - n >= 3 per side before believing a single-digit difference; run-to-run variance hit 17%
#   - compare STEPPING time, never wall clock -- startup is ~64% of wall
#   - per-call medians of update_contact_cache are the primary statistic (n ~ 600+ per run),
#     totals are secondary because they inherit between-run drift
#
# Requires the pi0.5 server already listening on $PORT -- start scripts/clara/interactive/pi05_server.sh first.
#
#   N=3 ./scripts/clara/interactive/t2_ab_contact.sh
set -uo pipefail
# Paths from lib/paths.sh, derived from this script's own location -- never from the profile's
# exported $REALM_ROOT, which names the pre-port 1.1.1 checkout. See that file's header.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/paths.sh"
# A failed `source` is not fatal by itself (no set -e), and $REALM_ROOT would then hold the
# value the shell profile exports -- the PRE-PORT 1.1.1 checkout -- so this run would silently
# evaluate the wrong tree. paths.sh sets REALM_PATHS_SH last and does not export it.
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }
cd "$REALM_ROOT" || exit 1

N=${N:-3}
REPEATS=${REPEATS:-2}
MAX_STEPS=${MAX_STEPS:-300}
PORT=${PORT:-8000}
TASK_ID=${TASK_ID:-0}
MODEL_NAME=${MODEL_NAME:-checkpoints_pi05_droid_jointpos}

mkdir -p tmp/interactive/prof

# Fail before booting a simulator if the policy server is not actually up.
python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)
" || { echo "### no policy server on 127.0.0.1:$PORT -- start scripts/clara/interactive/pi05_server.sh" >&2; exit 1; }
echo "### policy server up on :$PORT"

for i in $(seq 1 "$N"); do
  for INC in 0 1; do
    TAG="inc${INC}_r${i}"
    echo ""
    echo "############################################################"
    echo "### A/B run $TAG  (INCREMENTAL_CONTACT_CACHE=$INC)  $(date -Is)"
    echo "############################################################"
    MODE=oglite REALM_INCREMENTAL_CONTACT_CACHE=$INC ./scripts/clara/interactive/rr \
      python -u scripts/clara/interactive/profile_step.py \
        --out "/app/tmp/interactive/prof/${TAG}.json" -- \
        --task_id "$TASK_ID" --perturbation_id 0 \
        --repeats "$REPEATS" --max_steps "$MAX_STEPS" --horizon 8 \
        --model_type openpi --robot DROID \
        --model_name "$MODEL_NAME" \
        --port "$PORT" --host 127.0.0.1 \
        --experiment_name oglite_ab --run_id "$TAG" \
        --log_dir /logs
    echo "### $TAG exit: $?"
  done
done
echo "### A/B complete; analyse with scripts/clara/interactive/analyze_ab.py"
