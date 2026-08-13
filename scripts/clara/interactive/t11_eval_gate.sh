#!/bin/bash
# Does the eval wrapper still refuse to call a crashed run a success?
#
# Isaac's SimulationApp.close() hard-exits the process with status 0, so `$?` says nothing about
# whether an eval worked. Job 190683 (2026-08-13) died on `AssertionError: droid_robolab_v2 is not a
# registered robot` after ~6 minutes, wrote zero results, printed "[eval] exited 0", and Slurm logged
# it COMPLETED. sbatch_eval_pi05.sh now gates on check_run.py instead. This script is the proof that
# the gate works, and it costs seconds -- host only, no container, no GPU, no allocation.
#
#     bash scripts/clara/interactive/t11_eval_gate.sh
#
# Two matrices:
#   A. check_run.py directly, against REAL previously-successful result directories (which must keep
#      passing) and every failure shape the gate has to catch.
#   B. the actual tail of sbatch_eval_pi05.sh -- tee, ${PIPESTATUS[0]}, EXIT_CODE marker, gate call,
#      banner, exit code -- with the 40-minute apptainer eval replaced by a stub that replays a real
#      log and exits with a chosen status. The stub exits 0 in the cases that matter, exactly as
#      SimulationApp.close() does after an unhandled exception; if the wrapper reports success there,
#      the bug is back.
#
# Read-only with respect to everything under $REAL and $LOGS. Scratch goes in tmp/.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CHECK="python3 $HERE/check_run.py"
SBATCH=$HERE/sbatch_eval_pi05.sh

LOGS=${LOGS:-/mnt/home_lustre/sedlam56/projects/REALM/logs}
REAL=$LOGS/vec_pi05_verify/checkpoints_pi05_droid_jointpos
GOOD1=$REAL/default_vec4_800              # 25 rollouts, Default, VEC=4
GOOD2=$REAL/vbpose_vec4_800_framefix      # 25 rollouts, VB-POSE, VEC=4
CRASH_LOG=$LOGS/pi05_eval_190683.log      # AssertionError, zero results, reported exit 0
OK_LOG=$LOGS/pi05_eval_190692.log         # the run that produced GOOD2
REPORT_CSV=$GOOD1/reports/put_green_block_into_bowl_Default.csv

for p in "$GOOD1" "$GOOD2" "$CRASH_LOG" "$OK_LOG"; do
  [ -e "$p" ] || { echo "SKIP: fixture missing: $p" >&2; exit 0; }
done

WORK=$ROOT/tmp/evalgate
rm -rf "$WORK"; mkdir -p "$WORK"
PASSED=0; FAILED=0

# ---- fixtures ---------------------------------------------------------------------------------
cp "$OK_LOG"    "$WORK/ok_marked.log";    printf '### EXIT_CODE=0\n' >> "$WORK/ok_marked.log"
cp "$OK_LOG"    "$WORK/ok_unmarked.log"
cp "$CRASH_LOG" "$WORK/crash_marked.log"; printf '### EXIT_CODE=0\n' >> "$WORK/crash_marked.log"

mkfix() {  # mkfix <dir> <lines of the real CSV to keep, incl. header>
  mkdir -p "$1"/{reports,actions,qpos,videos}
  head -"$2" "$REPORT_CSV" > "$1/reports/put_green_block_into_bowl_Default.csv"
  for d in actions qpos videos; do printf 'parquet-stub' > "$1/$d/put_green_block_into_bowl.parquet"; done
}
EXP=$WORK/logs/exp/model
mkfix "$EXP/complete" 26      # header + 25 rollouts
mkfix "$EXP/partial"  13      # header + 12 rollouts: died half way through
mkfix "$EXP/empty"     1      # header only: no rollout ever finished
mkfix "$EXP/stale"    26      # complete, but written long before this run started
find "$EXP/stale" -type f -exec touch -d '2026-08-01 09:00:00' {} +
mkdir -p "$EXP/only_reports/reports"
cp "$REPORT_CSV" "$EXP/only_reports/reports/r.csv"
NOW=$(date +%s); OLD=$((NOW - 86400))

# ---- A: check_run.py directly -----------------------------------------------------------------
case_a() {  # case_a <expected rc> <name> <args...>
  local want=$1 name=$2; shift 2
  local out rc
  out=$($CHECK "$@" 2>&1); rc=$?
  if [ "$rc" -eq "$want" ]; then echo "  [ok]   $name -> rc=$rc"; PASSED=$((PASSED+1))
  else echo "  [FAIL] $name -> rc=$rc (expected $want)"; echo "$out" | sed 's/^/         | /'
       FAILED=$((FAILED+1)); fi
}

echo "===== A. check_run.py -- real successful runs MUST still pass ====="
case_a 0 "GOOD1 + repeats 25"                    "$GOOD1" --repeats 25
case_a 0 "GOOD2 + repeats 25"                    "$GOOD2" --repeats 25
case_a 0 "GOOD2 + clean marked log"              "$GOOD2" "$WORK/ok_marked.log" --repeats 25
case_a 0 "GOOD1 + newer-than a day ago"          "$GOOD1" --repeats 25 --newer-than "$OLD"
case_a 0 "GOOD1, no new flags (legacy calls)"    "$GOOD1"

echo "===== A. check_run.py -- these MUST fail ====="
case_a 1 "results dir never created"             "$REAL/no_such_run_id" --repeats 25
case_a 1 "crashed log, artifacts fine"           "$GOOD1" "$WORK/crash_marked.log" --repeats 25
case_a 1 "clean log but no EXIT_CODE marker"     "$GOOD2" "$WORK/ok_unmarked.log" --repeats 25
case_a 1 "25 rows, 40 requested (truncated)"     "$GOOD1" --repeats 40
case_a 1 "25 rows, 10 requested (not this run)"  "$GOOD1" --repeats 10
case_a 1 "header-only CSV"                       "$EXP/empty" --repeats 25
case_a 1 "12 of 25 rollouts"                     "$EXP/partial" --repeats 25
case_a 1 "CSV complete, other artifacts absent"  "$EXP/only_reports" --repeats 25
case_a 1 "artifacts from an earlier run"         "$EXP/stale" --repeats 25 --newer-than "$NOW"
echo "      (and the same stale dir with the mtime gate OFF, to show why the gate exists)"
case_a 0 "stale dir, no --newer-than"            "$EXP/stale" --repeats 25

# ---- B: the wrapper's own tail, with the eval stubbed -----------------------------------------
python3 - "$SBATCH" "$WORK/tail.sh" <<'PY'
import sys
src = open(sys.argv[1]).read()
start = src.index('RESULTS="$REALM_LOGS')
a = src.index("apptainer run --userns", start)
b = src.index('2>&1 | tee "$EVAL_LOG"', a) + len('2>&1 | tee "$EVAL_LOG"')
tail = src[start:a] + '{ cat "$STUB_LOG"; exit "$STUB_EXIT"; } 2>&1 | tee "$EVAL_LOG"' + src[b:]
open(sys.argv[2], "w").write("#!/bin/bash\nset -uo pipefail\n" + tail)
PY
for marker in 'PIPESTATUS\[0\]' 'check_run.py' 'EVAL FAILED'; do
  grep -q "$marker" "$WORK/tail.sh" || {
    echo "  [FAIL] could not carve the wrapper tail: no $marker -- did sbatch_eval_pi05.sh change shape?"
    FAILED=$((FAILED+1)); }
done

case_b() {  # case_b <expected rc> <run_id> <stub log> <stub exit> <name>
  local want=$1 rid=$2 stub=$3 sexit=$4 name=$5 out rc
  mkdir -p "$ROOT/tmp/gatetest"
  out=$(REALM_LOGS="$WORK/logs" EXPERIMENT=exp MODEL_NAME=model RUN_ID="$rid" JOB=gatetest \
        REPEATS=25 VEC=4 PERT_ID=10 TASK_ID=0 REALM_ROOT="$ROOT" \
        STUB_LOG="$stub" STUB_EXIT="$sexit" bash "$WORK/tail.sh" 2>&1); rc=$?
  if [ "$rc" -eq "$want" ]; then echo "  [ok]   $name -> rc=$rc"; PASSED=$((PASSED+1))
  else echo "  [FAIL] $name -> rc=$rc (expected $want)"; echo "$out" | tail -20 | sed 's/^/         | /'
       FAILED=$((FAILED+1)); fi
  printf '%s' "$out" > "$WORK/b_${rid}_${sexit}.txt"
}

echo "===== B. sbatch_eval_pi05.sh tail, eval stubbed ====="
case_b 0 complete    "$OK_LOG"    0   "complete run, clean log, exit 0 -- the success path"
case_b 1 no_such_run "$CRASH_LOG" 0   "190683 replay: crash, no results, exit 0"
case_b 1 complete    "$CRASH_LOG" 0   "crash, but an earlier run left complete artifacts"
case_b 1 stale       "$OK_LOG"    0   "clean log, complete artifacts, all stale"
case_b 1 partial     "$OK_LOG"    0   "clean log, 12 of 25 rollouts"
case_b 1 complete    "$OK_LOG"    139 "complete artifacts but the eval segfaulted"

echo
echo "===== $PASSED passed, $FAILED failed ====="
exit $((FAILED > 0))
