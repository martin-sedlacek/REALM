#!/bin/bash
# The whole xflat measurement chain, one allocation, four Isaac starts. MODE=stock throughout --
# the image's OWN OmniGibson, with NO loader patch bound over it. That is the point of this route:
# the asset alone has to fix it. Every run is paired with a DROID_robolab_v2 control taken in the
# same session with identical flags, so "the loader is still broken" is measured, not assumed.
#
#   usage: ./scripts/debug_probes/xflat_run_chain.sh <jobid>
#
# Isaac exits 139 at teardown regardless of outcome -- grep the verdict lines, never the exit code.
set -uo pipefail
JOB=${1:?usage: xflat_run_chain.sh <jobid>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L=/mnt/home_lustre/sedlam56/projects/REALM/logs/gripper_squeeze
CURL_FLAGS=(--load tip --states open --tip-fingers both
            --rungs "nf1000a=1000/0.05,nf1000b=1000/0.05"
            --hold-steps 3 --traverse-steps 0 --rest-steps 8 --retract-steps 8
            --tip-gap 0.020 --tip-dz 0.0005 --tip-steps 80 --tip-past 40 --video 0)

run() {  # run <logname> <args...>
  local name=$1; shift
  echo "=== $(date -Is) START $name ===" | tee -a "$L/xflat_chain.log"
  srun --jobid="$JOB" --overlap -n1 "$HERE/scripts/clara/interactive/rr" python -u "$@" \
      > "$L/$name.log" 2>&1
  echo "=== $(date -Is) DONE  $name (srun exit $?) ===" | tee -a "$L/xflat_chain.log"
}

# 1+2. runtime mass properties: the flattened asset, then the ORIGINAL as the loader control.
run xflat_inertia_runtime /app/scripts/debug_probes/inertia_runtime_realm.py \
    --robot DROID_robolab_xflat --out /logs/gripper_squeeze/xflat_inertia_runtime.json
run xflat_inertia_runtime_v2ctl /app/scripts/debug_probes/inertia_runtime_realm.py \
    --robot DROID_robolab_v2 --out /logs/gripper_squeeze/xflat_inertia_runtime_v2ctl.json

# 3+4. the curl at the AUTHORED naturalFrequency, flattened asset then original, same flags.
run xflat_curl /app/scripts/debug_probes/curl_press_direction.py \
    --robot DROID_robolab_xflat --tag xflat_curl "${CURL_FLAGS[@]}"
run xflat_curl_v2ctl /app/scripts/debug_probes/curl_press_direction.py \
    --robot DROID_robolab_v2 --tag xflat_curl_v2ctl "${CURL_FLAGS[@]}"

echo "XFLAT_CHAIN_COMPLETE $(date -Is)" | tee -a "$L/xflat_chain.log"
