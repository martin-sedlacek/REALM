#!/bin/bash
# Run the vectorized perturbation check (t9) for several perturbations at once, fanned across
# however many interactive allocations you hold.
#
#   ALLOCS="190155 190713" ./scripts/clara/interactive/t9_sweep.sh VB-POSE V-VIEW V-SC VSB-NOBJ
#   ALLOCS=190155 NUM_ENVS=2 RESETS=3 ./scripts/clara/interactive/t9_sweep.sh Default S-LANG
#
# Each perturbation gets its OWN env build (~6 min at 2 envs), which is deliberate and not an
# oversight: several perturbations are destructive to the env they run in -- SB-VRB rewrites
# env.task_type and its whole task progression, VSB-NOBJ replaces the main object, V-SC replaces the
# distractors -- so cycling perturbations through one shared build would let an earlier one
# contaminate every later one. The parallelism comes from holding several allocations, not from
# sharing a build.
#
# Logs land on Lustre at $LOGS/t9_sweep_<pert>.log. A summary is printed at the end; the per-run
# verdict line is what matters ("PASSED" / "FAILED -- N problem(s)").
set -uo pipefail

# Derived from this script's own location, NOT from $REALM_ROOT. The shell profile on this machine
# exports REALM_ROOT=/home/sedlam56/projects/REALM -- the PRE-PORT 1.1.1 checkout -- along with
# REALM_SIF pointing at the 1.1.1 image. A `${REALM_ROOT:-<og391 default>}` here therefore resolved
# to the wrong repo entirely and the run died with "rr: No such file or directory". A silent version
# of that mistake would be far worse: an og391 script quietly evaluating against the old stack.
# Deriving the root from the script path makes it impossible to point at the wrong tree.
REALM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LOGS=${LOGS:-/mnt/home_lustre/sedlam56/projects/REALM/logs}
ALLOCS=${ALLOCS:-}
NUM_ENVS=${NUM_ENVS:-2}
RESETS=${RESETS:-3}
STEPS=${STEPS:-15}
ROBOT=${ROBOT:-DROID_robolab_v2}
# Task matters for coverage, not just for variety: a perturbation can take a completely different
# code path depending on the task's main-object TYPE. VB-MOBJ rescales a PrimitiveObject but
# REMOVES AND RE-ADDS a DatasetObject, and only the second path touches the add/remove machinery.
# Task 0 (put_green_block_into_bowl) and 6 (stack_cubes) are PrimitiveObject; 1,2,3,4,5,7 are
# DatasetObject; 8,9 (open/close_drawer) do not currently load at all on this port (cabinet.usd,
# TypeError missing 'preset_name'). So a task-0 pass does NOT imply a perturbation is safe.
TASK_ID=${TASK_ID:-0}
TAG=${TAG:-sweep}

[ -n "$ALLOCS" ] || { echo "ERROR: set ALLOCS to one or more running interactive job IDs" >&2; exit 1; }
[ $# -gt 0 ]     || { echo "ERROR: pass at least one perturbation name" >&2; exit 1; }

read -r -a ALLOC_ARR <<< "$ALLOCS"
n_alloc=${#ALLOC_ARR[@]}
echo "=================================================================="
echo " t9 sweep: ${#} perturbation(s) over $n_alloc alloc(s)"
echo " num_envs=$NUM_ENVS resets=$RESETS steps=$STEPS task=$TASK_ID robot=$ROBOT"
echo "=================================================================="

pids=()
perts=()
i=0
for pert in "$@"; do
  alloc=${ALLOC_ARR[$((i % n_alloc))]}
  log="$LOGS/t9_${TAG}_${pert}.log"
  echo "[sweep] $pert (task $TASK_ID) -> alloc $alloc -> $log"
  (
    cd "$REALM_ROOT" || exit 1
    MODE=oglite srun --jobid="$alloc" --overlap -n1 ./scripts/clara/interactive/rr \
      python -u scripts/clara/interactive/t9_vbpose_nostopplay.py \
        --num_envs "$NUM_ENVS" --resets "$RESETS" --steps "$STEPS" \
        --task_id "$TASK_ID" --robot "$ROBOT" --perturbation "$pert"
  ) > "$log" 2>&1 &
  pids+=($!)
  perts+=("$pert")
  i=$((i + 1))
done

echo "[sweep] ${#pids[@]} run(s) launched; waiting..."
for idx in "${!pids[@]}"; do
  wait "${pids[$idx]}"
done

echo
echo "=================================== SUMMARY ==================================="
rc=0
for pert in "${perts[@]}"; do
  log="$LOGS/t9_${TAG}_${pert}.log"
  # strip ANSI, then take the verdict line the harness prints
  verdict=$(sed 's/\x1b\[[0-9;]*m//g' "$log" 2>/dev/null | grep -E "^(PASSED|FAILED)" | tail -1)
  if [ -z "$verdict" ]; then
    # No verdict at all means it died before finishing -- surface the cause rather than a blank,
    # because a silent run reads exactly like a passing one otherwise.
    #
    # FIRST error, not last: Isaac segfaults during teardown on EVERY run including passing ones
    # (og.shutdown -> SimulationApp.close hard-exits), so the last error is always that segfault and
    # tells you nothing. The first one is the real cause.
    cause=$(sed 's/\x1b\[[0-9;]*m//g' "$log" 2>/dev/null \
            | grep -E "AssertionError|IndexError|RuntimeError|KeyError|TypeError|ValueError" \
            | head -1 | cut -c1-110)
    [ -n "$cause" ] || cause=$(sed 's/\x1b\[[0-9;]*m//g' "$log" 2>/dev/null \
            | grep -E "Segmentation fault|Error:" | head -1 | cut -c1-110)
    verdict="NO VERDICT (died) -- ${cause:-see log}"
    rc=1
  fi
  case "$verdict" in FAILED*) rc=1 ;; esac
  printf '  %-12s %s\n' "$pert" "$verdict"
done
echo "==============================================================================="
exit $rc
