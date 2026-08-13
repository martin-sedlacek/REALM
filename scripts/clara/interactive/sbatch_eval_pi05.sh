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

REALM_ROOT=/mnt/home_lustre/sedlam56/projects/REALM_og391
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

for i in $(seq 1 "$SERVER_WAIT"); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[eval] server died:" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
  python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null && { echo "[eval] server up after ${i}s"; break; }
  sleep 1
done

#--- eval, inside the container with the OG-lite bind ----------------------------------------------
if [ "$VEC" -ge 1 ]; then
  ENTRY=(python -u /app/examples/04_vector_evaluate.py --num_envs "$VEC")
else
  ENTRY=(python -u /app/examples/02_evaluate.py)
fi

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
    --robot "$ROBOT" --rendering_mode rt
EXIT=$?

echo "[eval] exited $EXIT"
RESULTS="$REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
echo "[eval] artifacts: $RESULTS"
ls "$RESULTS" 2>/dev/null
[ "$EXIT" -eq 0 ] && rm -rf "$REALM_ROOT/tmp/$JOB"
exit $EXIT
