#!/bin/bash
# Self-contained pi0.5 REALM eval: starts its OWN policy server on this node, runs the eval, tears
# the server down. One server per batch of evals, so several of these can run in parallel on
# different nodes without sharing a port or a GPU.
#
#   VEC=4 PERT_ID=0  MAX_STEPS=800 REPEATS=25 RUN_ID=def_vec4  sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
#   VEC=0 PERT_ID=13 MAX_STEPS=800 REPEATS=25 RUN_ID=vbpose_single sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
#
# VEC=0 runs the single-env path (examples/02_evaluate.py); VEC=N>=1 runs the vectorized path
# (examples/04_vector_evaluate.py) in waves of N.
#
# PERTURBATION SAFETY: perturbations that call og.sim.stop()/play() are NOT safe vectorized, because
# those are global and REALM applies perturbations per member inside reset() -- one member's cycle
# disturbs every other member. Known offenders: VB-POSE (vb_pose.py:25,49), VB-MOBJ, VSB-NOBJ,
# SB-VRB. Run those with VEC=0 until the stop/play is batched the way apply_scene_fixes_from_cfg
# already is. Default is safe vectorized.
#
# WHY THIS SCRIPT DOES NOT TRUST THE EVAL'S EXIT CODE: Isaac's SimulationApp.close() hard-exits the
# process with status 0, so an unhandled Python exception still leaves $? at 0. Job 190683
# (2026-08-13) died on `AssertionError: droid_robolab_v2 is not a registered robot` after ~6 minutes,
# wrote no results at all, printed "[eval] exited 0", and Slurm recorded COMPLETED. Three runs were
# silently "successful" that day, one of which was read as a result. The authoritative signal is
# therefore the ARTIFACTS plus a scan of the run's own log -- see check_run.py, and the
# "eval gate" block at the bottom. Exit code is reported but never decides.
#
#SBATCH --job-name realm-pi05-eval
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 04:00:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/pi05_eval_%j.log

set -uo pipefail

# Overridable so a worktree can eval its own checkout: this path is what gets bound as /app AND
# where the gate's check_run.py is read from, so the two must not be allowed to drift apart.
REALM_ROOT=${REALM_ROOT:-/mnt/home_lustre/sedlam56/projects/REALM_og391}
OGLITE_ROOT=/mnt/home_lustre/sedlam56/projects/OG-lite_og391
OPENPI_ROOT=/mnt/home_lustre/sedlam56/projects/openpi
REALM_DATA=$REALM_ROOT/data/datasets
REALM_LOGS=/mnt/home_lustre/sedlam56/projects/REALM/logs
REALM_SIF=$REALM_ROOT/realm_og391.sif
APPDATA=$REALM_ROOT/data/cache
CKPT=${CKPT:-/home/sedlam56/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos}

VEC=${VEC:-4}
PERT_ID=${PERT_ID:-0}
TASK_ID=${TASK_ID:-0}
REPEATS=${REPEATS:-25}
MAX_STEPS=${MAX_STEPS:-800}
HORIZON=${HORIZON:-8}
ROBOT=${ROBOT:-DROID_robolab_v2}
MODEL_NAME=${MODEL_NAME:-checkpoints_pi05_droid_jointpos}
EXPERIMENT=${EXPERIMENT:-vec_pi05_verify}
JOB=${SLURM_JOB_ID:-local}
RUN_ID=${RUN_ID:-run_$JOB}
# Own port per job so two of these never collide even if Slurm co-locates them.
PORT=${PORT:-$((8000 + (JOB % 1000)))}
SERVER_WAIT=${SERVER_WAIT:-300}

[ -f "$REALM_SIF" ]                     || { echo "ERROR: no SIF" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ] || { echo "ERROR: no dataset" >&2; exit 1; }
[ -d "$CKPT/params" ]                   || { echo "ERROR: no params/ under $CKPT" >&2; exit 1; }
[ -d "$OGLITE_ROOT/omnigibson" ]        || { echo "ERROR: no OG-lite" >&2; exit 1; }
mkdir -p "$REALM_ROOT/tmp/$JOB" "$APPDATA/appdata" "$REALM_LOGS/$EXPERIMENT"

echo "=================================================================="
echo " pi0.5 eval  vec=$VEC  pert_id=$PERT_ID  task=$TASK_ID"
echo " repeats=$REPEATS  max_steps=$MAX_STEPS  horizon=$HORIZON  robot=$ROBOT"
echo " run_id=$RUN_ID  port=$PORT  node=$(hostname)"
echo "=================================================================="

#--- own policy server -----------------------------------------------------------------------------
SERVER_LOG="$REALM_LOGS/pi05_server_${JOB}.log"
(
  cd "$OPENPI_ROOT" || exit 1
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  export HF_HUB_OFFLINE=1
  exec uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
      --policy.config=pi05_full_droid_finetune --policy.dir="$CKPT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval] server pid=$SERVER_PID port=$PORT log=$SERVER_LOG"
trap 'kill $SERVER_PID 2>/dev/null' EXIT

SERVER_UP=0
for i in $(seq 1 "$SERVER_WAIT"); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[eval] server died:" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
  python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null && { echo "[eval] server up after ${i}s"; SERVER_UP=1; break; }
  sleep 1
done
# Falling out of the loop without a connection used to run the eval anyway against a dead port. The
# eval then fails ~6 minutes later inside the container, which the gate at the bottom now catches --
# but there is no reason to spend those minutes, or to report the failure as an eval bug.
[ "$SERVER_UP" -eq 1 ] || {
  echo "[eval] FATAL: policy server did not accept connections on port $PORT within ${SERVER_WAIT}s" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
}

#--- eval, inside the container with the OG-lite bind ----------------------------------------------
if [ "$VEC" -ge 1 ]; then
  ENTRY=(python -u /app/examples/04_vector_evaluate.py --num_envs "$VEC")
else
  ENTRY=(python -u /app/examples/02_evaluate.py)
fi

RESULTS="$REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
EVAL_LOG="$REALM_LOGS/pi05_evalout_${JOB}_${RUN_ID}.log"
# Taken BEFORE the eval so the gate can tell this run's artifacts from ones an earlier run with the
# same RUN_ID left in $RESULTS. Without it, re-running a RUN_ID that previously succeeded makes any
# crash look like a pass, because the directory is already full of a valid-looking eval.
START_EPOCH=$(date +%s)

apptainer run --userns --nv --writable-tmpfs --pwd /app \
  --bind "$REALM_ROOT":/app \
  --bind "$REALM_DATA":/data \
  --bind "$APPDATA":/cache \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT/tmp/$JOB":/tmp \
  --bind "$OGLITE_ROOT/omnigibson":/behavior-src/OmniGibson/omnigibson \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONUNBUFFERED=1 \
  "$REALM_SIF" \
  "${ENTRY[@]}" \
    --task_id "$TASK_ID" --perturbation_id "$PERT_ID" \
    --repeats "$REPEATS" --max_steps "$MAX_STEPS" --horizon "$HORIZON" \
    --model_type openpi --model_name "$MODEL_NAME" \
    --port "$PORT" --host 127.0.0.1 \
    --experiment_name "$EXPERIMENT" --run_id "$RUN_ID" --log_dir /logs \
    --robot "$ROBOT" --rendering_mode rt \
  2>&1 | tee "$EVAL_LOG"
# ${PIPESTATUS[0]}, not $?: $? is the exit of `tee`. The eval's own output is teed to its own file so
# the gate below has an exact, self-contained log to scan even when this runs outside Slurm (JOB=local,
# no %j log) -- and so the scan cannot be confused by anything this wrapper printed.
EXIT=${PIPESTATUS[0]}

echo "[eval] exited $EXIT  (NOT a success signal -- see the gate below)"
printf '### EXIT_CODE=%s\n' "$EXIT" >> "$EVAL_LOG"
echo "[eval] artifacts: $RESULTS"
ls "$RESULTS" 2>/dev/null

#--- eval gate: did this run actually produce a complete set of results? ----------------------------
# The only evidence that counts. check_run.py requires all four artifacts, exactly $REPEATS rollout
# rows in reports/*.csv (so a run that died half way through is not reported as complete), artifact
# mtimes newer than this job's start (so a previous run's leftovers under the same RUN_ID cannot
# stand in for this one), and a log free of Traceback / AssertionError / Segmentation fault / CUDA
# OOM / 'row mismatch'.
python3 "$REALM_ROOT/scripts/clara/interactive/check_run.py" \
    "$RESULTS" "$EVAL_LOG" --repeats "$REPEATS" --newer-than "$START_EPOCH"
GATE=$?

if [ "$GATE" -ne 0 ]; then
  echo "==================================================================" >&2
  echo " EVAL FAILED -- $RUN_ID did NOT produce a complete, verified set of results" >&2
  echo "   job=$JOB  run_id=$RUN_ID  vec=$VEC  pert_id=$PERT_ID  task=$TASK_ID  repeats=$REPEATS" >&2
  echo "   reported exit code was $EXIT, which proves nothing: SimulationApp.close() exits 0" >&2
  echo "   even after an unhandled exception. The gate above is the real verdict." >&2
  echo "   eval log:  $EVAL_LOG" >&2
  echo "   artifacts: $RESULTS" >&2
  echo "   DO NOT report numbers from this run." >&2
  echo "   Keeping $REALM_ROOT/tmp/$JOB for debugging." >&2
  echo "==================================================================" >&2
  exit 1
fi

echo "=================================================================="
echo " EVAL OK -- $RUN_ID: $REPEATS rollouts, all artifacts present, clean log"
echo " artifacts: $RESULTS"
echo "=================================================================="
rm -rf "$REALM_ROOT/tmp/$JOB"
exit 0
