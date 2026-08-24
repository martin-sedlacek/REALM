#!/bin/bash
# Render the t=0 initial condition of every DROID100 tabletop task, 3 views each, as a Slurm ARRAY.
#
# One array task = one L40S = one Isaac boot, which then walks its own round-robin slice of the 100
# task configs in-process (og.clear() between them, relaunching when that fails -- see the loop
# below). No policy server, no model: the probe holds the reset pose for one step and saves that
# observation. See
# scripts/debug_probes/droid100_first_frames.py for what it records and which defects it flags.
#
# All array tasks write into ONE run directory so the result is a single pull:
#   $REALM_LOGS/droid100_first_frames/<RUN_ID>/{frames/<task>/{cam1,cam2,wrist,panel}.jpg,shardNN.json}
# RUN_ID defaults to $SLURM_ARRAY_JOB_ID, which is the same value in every array task -- do NOT
# default it to a timestamp, each task would compute its own and the run would land in 16 dirs.
#
# Usage:
#   sbatch --array=0-15 scripts/clara/sbatch_droid100_frames.sh
#   SHARDS=8  sbatch --array=0-7  scripts/clara/sbatch_droid100_frames.sh
#   TASKS=001_put_the_marker_in_the_cup sbatch --array=0-0 scripts/clara/sbatch_droid100_frames.sh
#   RUN_ID=retry_of_193xxx sbatch --array=3,7 scripts/clara/sbatch_droid100_frames.sh   # re-run 2 shards
#
# --array=0-15 with SHARDS=16 (the default) is the intended shape: 16 GPUs, ~6-7 tasks each.
#
#SBATCH --job-name realm-droid100-frames
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 03:00:00
#SBATCH --exclude=l40s-06
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/droid100_frames_%A_%a.log

set -uo pipefail

# Locate lib/paths.sh. Under sbatch, Slurm ships this script's TEXT to the node and runs a copy from
# /var/spool/slurmd/job<N>/slurm_script, so BASH_SOURCE points into the spool dir -- `scontrol show
# job` still reports the path sbatch was handed, and $SLURM_SUBMIT_DIR is the last resort. Not
# finding it is FATAL: set -e is off, and carrying on would leave $REALM_ROOT at the value the shell
# profile exports, which is the PRE-PORT 1.1.1 checkout. See scripts/clara/lib/paths.sh.
_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")/lib" 2>/dev/null && pwd)
if [ ! -f "${_lib:-/nonexistent}/paths.sh" ]; then
  _cmd=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
  _lib=$(cd "$(dirname "${_cmd:-/nonexistent}")/lib" 2>/dev/null && pwd)
fi
[ -f "${_lib:-/nonexistent}/paths.sh" ] || _lib=${SLURM_SUBMIT_DIR:-$PWD}/scripts/clara/lib
[ -f "$_lib/paths.sh" ] || { echo "ERROR: cannot locate scripts/clara/lib/paths.sh" >&2; exit 1; }
source "$_lib/paths.sh"
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source $_lib/paths.sh" >&2; exit 1; }

# v3 explicitly: it is the only image with OG-lite baked in (OGLITE_PROVENANCE), so MODE=stock
# already carries the up-axis / z-offset / init-queue / preset_name patches AND honours
# REALM_LIGHT_FIX. paths.sh still prefers v2, which does not -- so name v3 rather than inherit.
export REALM_SIF_OG391=${REALM_SIF_OG391:-$REALM_SHARED/realm_og391_v3.sif}
export MODE=${MODE:-stock}
export REALM_LIGHT_FIX=${REALM_LIGHT_FIX:-1}

SUITE=${SUITE:-DROID100_tabletop}
SHARDS=${SHARDS:-16}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
ROBOT=${ROBOT:-DROID}
RENDERING_MODE=${RENDERING_MODE:-rt}
EXPERIMENT=${EXPERIMENT:-droid100_first_frames}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}
TASKS=${TASKS:-}          # explicit comma-separated task dirs; overrides sharding when set
LIMIT=${LIMIT:-}          # stop after N tasks, for smoke testing one shard

OUT_HOST=$REALM_LOGS/$EXPERIMENT/$RUN_ID
OUT_CONT=/logs/$EXPERIMENT/$RUN_ID

[ -f "$REALM_SIF_OG391" ] || { echo "ERROR: no SIF at $REALM_SIF_OG391" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ] || { echo "ERROR: no dataset at $REALM_DATA" >&2; exit 1; }
[ -d "$REALM_LOGS" ] || { echo "ERROR: no log dir at $REALM_LOGS" >&2; exit 1; }
mkdir -p "$OUT_HOST"

echo "=================================================================="
echo " DROID100 first frames -- shard $SHARD of $SHARDS"
echo " suite=$SUITE robot=$ROBOT rendering=$RENDERING_MODE light_fix=$REALM_LIGHT_FIX mode=$MODE"
echo " sif     = $(readlink -f "$REALM_SIF_OG391")"
echo " out     = $OUT_HOST"
echo " node    = $(hostname)  job=${SLURM_JOB_ID:-?}  array=${SLURM_ARRAY_JOB_ID:-?}"
echo "=================================================================="

cd "$REALM_ROOT" || exit 1

ARGS=(--out "$OUT_CONT" --suite "$SUITE" --shard "$SHARD" --num_shards "$SHARDS"
      --robot "$ROBOT" --rendering_mode "$RENDERING_MODE")
[ -n "$TASKS" ] && ARGS+=(--tasks "$TASKS")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")

# RELAUNCH LOOP. The probe pays the ~3 min Isaac cold start ONCE and then walks its whole slice
# in-process via og.clear(). But og.clear() dies on the replicator's annotator detach with the three
# vision sensors this probe runs (measured, job 195340 -- see the probe's teardown() docstring), and
# when it does the process cannot build another environment. The probe exits having recorded every
# task it got through, and this loop starts a fresh one for whatever has no record yet.
#
# So the cost degrades gracefully instead of failing: one boot for the shard if og.clear() holds, one
# boot per task if it never does, and anything in between.
#
# The loop terminates on PROGRESS, not on an exit code: Isaac exits 0 on unhandled exceptions and
# segfaults at teardown on passing runs, so the code is not a verdict here any more than it is
# anywhere else in this harness. The shard JSON is. An attempt that adds no record is the end --
# that is a task the probe cannot get past (a segfault leaves no record), and retrying it would burn
# one full Isaac boot per attempt to reproduce it.
SHARD_JSON=$OUT_HOST/shard$(printf '%02d' "$SHARD").json
n_records() { python3 -c "
import json,sys
try: print(len(json.load(open(sys.argv[1]))['records']))
except Exception: print(0)" "$SHARD_JSON"; }

ATTEMPTS=${ATTEMPTS:-110}          # runaway guard only; the progress test is what normally stops us
EXIT=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  before=$(n_records)
  echo "[droid100] --- attempt $attempt/$ATTEMPTS (shard $SHARD, $before recorded) ---"
  ./scripts/clara/interactive/rr \
    python -u scripts/debug_probes/droid100_first_frames.py "${ARGS[@]}"
  EXIT=$?
  after=$(n_records)
  echo "[droid100] attempt $attempt exited $EXIT; records $before -> $after"
  [ "$EXIT" -eq 0 ] && break
  if [ "$after" -le "$before" ]; then
    echo "[droid100] attempt $attempt made NO progress -- stopping (see shard JSON for what is missing)"
    break
  fi
done

# The verdict, so the Slurm log answers "did this shard finish, and what is wrong with these tasks"
# without a second command.
echo "[droid100] last attempt exited $EXIT (not authoritative -- read the JSON below)"
python3 - "$SHARD_JSON" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"[droid100] cannot read {sys.argv[1]}: {e}"); sys.exit(0)
recs = d.get("records", [])
print(f"[droid100] shard {d['shard']}: {sum(r['status']=='ok' for r in recs)}/{len(d.get('tasks',[]))} ok"
      + (f"  ABORTED AFTER {d['aborted_after']}" if "aborted_after" in d else ""))
for r in recs:
    if r["status"] != "ok":
        print(f"  FAIL {r['task']}: {r.get('traceback','').strip().splitlines()[-1:]}")
    elif r.get("flags"):
        print(f"  flag {r['task']}: {','.join(r['flags'])}")
PY

exit "$EXIT"
