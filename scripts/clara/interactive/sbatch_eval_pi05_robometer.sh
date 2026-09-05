#!/bin/bash
# pi0.5 REALM eval scored by a Robometer reward model, both servers started on this node.
#
#   TASK_ID=1 REPEATS=8 VEC=4 MAX_STEPS=300 RUN_ID=rbm_t1 \
#     sbatch scripts/clara/interactive/sbatch_eval_pi05_robometer.sh
#
# This is sbatch_eval_pi05.sh plus a second server and the --robometer flags. Everything about the
# policy server, the OG-lite bind, the light fix and the check_run.py gate is unchanged and the
# comments there still apply -- read that script first; only the Robometer-specific parts are
# commented here.
#
# TWO GPUs, deliberately. GPU 0 carries the pi0.5 server AND Isaac; Robometer-4B (~10-12 GB) gets
# GPU 1 to itself. They could be co-located on one 48 GB L40S, but then a Robometer OOM would look
# like a simulator failure, and the point of this run is to characterise the scorer.
#
# THE CLIENT IS BOUND IN, NOT BAKED IN. .docker/realm.def on this branch installs
# packages/robometer-client into the image, but the image we have (realm_og391_v3.sif, 2026-08-20)
# predates the branch. The client is pure Python over numpy + requests, both already in the image
# (verified: PIL/requests/numpy present, robometer_client MISSING), so binding its src/ onto
# PYTHONPATH is equivalent to the install for running purposes and costs no rebuild. A rebuilt image
# makes the bind a harmless no-op -- the installed copy wins only if this bind is dropped, so keep
# them at the same revision or drop the bind.
#
#SBATCH --job-name realm-pi05-rbm
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 200G
#SBATCH --time 04:00:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/pi05_rbm_%j.log

set -uo pipefail

_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)
if [ ! -f "${_lib:-/nonexistent}/paths.sh" ]; then
  _cmd=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
  _lib=$(cd "$(dirname "${_cmd:-/nonexistent}")/../lib" 2>/dev/null && pwd)
fi
[ -f "${_lib:-/nonexistent}/paths.sh" ] || _lib=${SLURM_SUBMIT_DIR:-$PWD}/scripts/clara/lib
[ -f "$_lib/paths.sh" ] || { echo "ERROR: cannot locate scripts/clara/lib/paths.sh" >&2; exit 1; }
source "$_lib/paths.sh"
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source $_lib/paths.sh" >&2; exit 1; }

# paths.sh only knows v1/v2; v3 is the image with OG-lite baked in and is what main was smoke-tested
# on (job 204578). Pin it rather than inherit whatever the candidate list picks.
REALM_SIF=${REALM_SIF_OG391:-/mnt/home_lustre/sedlam56/projects/REALM/realm_og391_v3.sif}
OGLITE_BIND=${OGLITE_BIND:-0}

OPENPI_ROOT=${OPENPI_ROOT:-/mnt/home_lustre/sedlam56/projects/openpi}
CKPT=${CKPT:-/home/sedlam56/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_full_droid_finetune}
REALM_LIGHT_FIX=${REALM_LIGHT_FIX:-1}

VEC=${VEC:-4}
PERT_ID=${PERT_ID:-0}
TASK_ID=${TASK_ID:-1}
REPEATS=${REPEATS:-8}
MAX_STEPS=${MAX_STEPS:-300}
HORIZON=${HORIZON:-8}
ROBOT=${ROBOT:-DROID_mounted}
MODEL_NAME=${MODEL_NAME:-checkpoints_pi05_droid_jointpos}
# Robometer rows carry a different quantity in task_progression and must never land in a directory
# with rubric rows -- wiki/Robometer.md is explicit about it, and realm/eval.py refuses --resume
# across the two. Its own experiment name, always.
EXPERIMENT=${EXPERIMENT:-robometer_reliability}
JOB=${SLURM_JOB_ID:-local}
RUN_ID=${RUN_ID:-run_$JOB}
PORT=${PORT:-$((8000 + (JOB % 1000)))}
SERVER_WAIT=${SERVER_WAIT:-300}

#--- Robometer knobs -------------------------------------------------------------------------------
# JOB-DERIVED, like the policy port. It was a flat 8010, and that is a real bug that cost jobs
# 204597/204598: Slurm co-located all four task runs on l40s-01, only the first bound 8010, and the
# other three logged "[Errno 98] Address already in use", then had their `wait_for_port` preflight
# SATISFIED BY THE NEIGHBOUR'S SERVER. They scored fine against it until that job finished and its
# trap killed the server, at which point the survivors died mid-run with RemoteDisconnected. The
# 8000-8999 band belongs to the policy port ($((8000 + JOB % 1000))), so this takes 9000-9999.
RBM_PORT=${RBM_PORT:-$((9000 + (JOB % 1000)))}
RBM_MODEL=${RBM_MODEL:-robometer/Robometer-4B}
RBM_THRESHOLD=${RBM_THRESHOLD:-0.9}
RBM_FRAME_SIZE=${RBM_FRAME_SIZE:-256}
# Checkpoint is pre-fetched on the login node (compute-node egress is not assumed). Offline so a
# missing file fails loudly here instead of hanging on a network timeout during model load.
export HF_HOME=${HF_HOME:-/mnt/home_lustre/sedlam56/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Model load is minutes, not seconds: 4B weights off Lustre plus a transformers build.
RBM_WAIT=${RBM_WAIT:-1800}
RBM_CLIENT_SRC=${RBM_CLIENT_SRC:-$REALM_ROOT/packages/robometer-client/src}

# --multi-view adds a SECOND EXTERIOR camera, auto-placed at the other named viewpoint
# (env_config.py: ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"), so it is a
# genuinely different angle -- the point being occlusion coverage.
#
# It does NOT change what pi0.5 sees on these tasks: the policy's use_base_im_second comes from
# rollout.wants_base_im_second(), which is `task_type in DRAWER_TASK_TYPES`, and pick/put/rotate/push
# are none of them. progress_scorer.exterior_frame() applies the same rule. So for non-drawer tasks
# this only adds a camera to the observation dict and to the recorded video; the rollouts stay
# comparable (modulo pi0.5's own non-determinism).
#
# COST: VideoRecorder switches to a 2x2 tiling (base | base_second over wrist | black) and the whole
# frame is capped at VIDEO_TARGET_HEIGHT=480, so each tile is 240 tall instead of 480. Both end at
# 256x144 after the scorer's longest-side-256 downscale, but the multi-view source carries less
# detail -- compare cameras WITHIN a multi-view run, not against the 2-tile runs.
MULTIVIEW=${MULTIVIEW:-0}
if [ "$MULTIVIEW" = 1 ]; then MULTIVIEW_ARG=(--multi-view); else MULTIVIEW_ARG=(); fi

[ -f "$REALM_SIF" ]                     || { echo "ERROR: no SIF at $REALM_SIF" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ] || { echo "ERROR: no dataset" >&2; exit 1; }
[ -d "$CKPT/params" ]                   || { echo "ERROR: no params/ under $CKPT" >&2; exit 1; }
[ -f "$OPENPI_ROOT/scripts/serve_policy.py" ] || { echo "ERROR: no serve_policy.py" >&2; exit 1; }
grep -q "name=\"$POLICY_CONFIG\"" "$OPENPI_ROOT/src/openpi/training/config.py" \
  || { echo "ERROR: POLICY_CONFIG=$POLICY_CONFIG not defined in $OPENPI_ROOT" >&2; exit 1; }
[ -d "$RBM_CLIENT_SRC/robometer_client" ] || { echo "ERROR: no robometer_client at $RBM_CLIENT_SRC" >&2; exit 1; }
# The Robometer-4B snapshot is WEIGHTS ONLY -- no tokenizer.json, no preprocessor_config.json, no
# chat template. The loader takes those from the base model named in the checkpoint's own
# config.yaml (base_model_id: Qwen/Qwen3-VL-4B-Instruct), so BOTH repos must be in the cache before
# HF_HUB_OFFLINE=1 can work. Pre-fetching only the Robometer repo is what killed job 204580: it got
# through tag resolution and died in transformers' config load with "We couldn't connect to
# huggingface.co ... and couldn't find them in the cached files". Fail here instead.
_rbm_base_cache="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct"
[ -d "$_rbm_base_cache" ] || { echo "ERROR: base model not cached at $_rbm_base_cache" >&2
  echo "       Run on a host with egress: HF_HOME=$HF_HOME python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct')\"" >&2
  exit 1; }
[ -d "$HF_HOME/hub/models--robometer--Robometer-4B" ] || { echo "ERROR: Robometer-4B not cached under $HF_HOME" >&2; exit 1; }
[ -x "$REALM_ROOT/packages/robometer/.venv/bin/python" ] \
  || { echo "ERROR: robometer env not built -- run 'uv sync --extra robometer' in packages/robometer" >&2; exit 1; }

if [ "$OGLITE_BIND" = 0 ]; then
  apptainer exec --userns "$REALM_SIF" test -f /behavior-src/OmniGibson/OGLITE_PROVENANCE 2>/dev/null \
    || { echo "ERROR: OGLITE_BIND=0 but $(basename "$REALM_SIF") has no baked-in OG-lite" >&2; exit 1; }
  OGLITE_BIND_ARGS=()
  echo "OG-lite: BAKED INTO THE IMAGE ($(basename "$REALM_SIF"))"
else
  [ -d "$REALM_OGLITE_ROOT/omnigibson" ] || { echo "ERROR: no OG-lite" >&2; exit 1; }
  OGLITE_BIND_ARGS=(--bind "$REALM_OGLITE_ROOT/omnigibson":/behavior-src/OmniGibson/omnigibson)
  echo "OG-lite: BOUND from $REALM_OGLITE_ROOT"
fi
python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$RBM_PORT))==0 else 1)" 2>/dev/null \
  && { echo "ERROR: something is already listening on 127.0.0.1:$RBM_PORT on $(hostname)." >&2
       echo "       Refusing to start: the preflight below would be satisfied by THAT server, this" >&2
       echo "       job would score against a process it does not own, and would die when that" >&2
       echo "       process exits. Set RBM_PORT explicitly." >&2
       exit 1; }

mkdir -p "$REALM_ROOT/tmp/$JOB" "$REALM_APPDATA/appdata" "$REALM_LOGS/$EXPERIMENT"

echo "=================================================================="
echo " pi0.5 + Robometer  vec=$VEC  task=$TASK_ID  pert=$PERT_ID"
echo " repeats=$REPEATS  max_steps=$MAX_STEPS  horizon=$HORIZON  robot=$ROBOT"
echo " policy=$POLICY_CONFIG  ckpt=$CKPT"
echo " robometer=$RBM_MODEL  port=$RBM_PORT  threshold=$RBM_THRESHOLD  frame=$RBM_FRAME_SIZE"
echo " realm_root=$REALM_ROOT"
echo " sif=$(basename "$REALM_SIF")  light_fix=$REALM_LIGHT_FIX"
echo " run_id=$RUN_ID  policy_port=$PORT  node=$(hostname)"
echo " gpus: 0 = pi0.5 + Isaac, 1 = robometer   multi_view=$MULTIVIEW"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

#--- policy server, GPU 0 --------------------------------------------------------------------------
SERVER_LOG="$REALM_LOGS/pi05_server_${JOB}.log"
(
  cd "$OPENPI_ROOT" || exit 1
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  export HF_HUB_OFFLINE=1
  exec uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
      --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval] policy server pid=$SERVER_PID port=$PORT log=$SERVER_LOG"

#--- robometer server, GPU 1 -----------------------------------------------------------------------
RBM_LOG="$REALM_LOGS/robometer_server_${JOB}.log"
(
  export CUDA_VISIBLE_DEVICES=1
  export ROBOMETER_MODEL="$RBM_MODEL" ROBOMETER_PORT="$RBM_PORT" ROBOMETER_HOST=0.0.0.0
  export ROBOMETER_NUM_GPUS=1
  exec "$REALM_ROOT/scripts/run_robometer_server.sh"
) >"$RBM_LOG" 2>&1 &
RBM_PID=$!
echo "[eval] robometer server pid=$RBM_PID port=$RBM_PORT log=$RBM_LOG"

trap 'kill $SERVER_PID $RBM_PID 2>/dev/null' EXIT

wait_for_port() {  # name pid port seconds logfile
  local name=$1 pid=$2 port=$3 secs=$4 log=$5 i
  for i in $(seq 1 "$secs"); do
    kill -0 "$pid" 2>/dev/null || { echo "[eval] $name server DIED:" >&2; tail -40 "$log" >&2; return 1; }
    python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null \
      && { echo "[eval] $name server up after ${i}s"; return 0; }
    sleep 1
  done
  echo "[eval] FATAL: $name server did not accept connections on $port within ${secs}s" >&2
  tail -40 "$log" >&2
  return 1
}

wait_for_port pi0.5 "$SERVER_PID" "$PORT" "$SERVER_WAIT" "$SERVER_LOG" || exit 1
wait_for_port robometer "$RBM_PID" "$RBM_PORT" "$RBM_WAIT" "$RBM_LOG" || exit 1

# An open port is not a loaded model. /health is what the client's own preflight calls, and one real
# scoring call is the only thing that proves the pinned revision still speaks the client's wire
# format -- wiki/Robometer.md says exactly this, and it costs seconds here versus an Isaac start.
echo "[eval] robometer /health: $(curl -s --max-time 30 "http://127.0.0.1:$RBM_PORT/health")"
PYTHONPATH="$RBM_CLIENT_SRC" "$REALM_ROOT/.venv/bin/python" - <<PY || { echo "[eval] FATAL: robometer live probe failed" >&2; exit 1; }
import numpy as np, sys
from robometer_client import RobometerClient
c = RobometerClient(host="127.0.0.1", port=$RBM_PORT)
c.wait_until_healthy(timeout_s=120)
r = c.progress(np.zeros((8, 180, 320, 3), dtype=np.uint8), "put the banana in the box")
print("[eval] live probe: reward=%.4f trace_len=%d success_prob=%s" % (r.reward, len(r.progress), r.success_prob))
PY

#--- eval ------------------------------------------------------------------------------------------
if [ "$VEC" -ge 1 ]; then
  ENTRY=(python -u /app/examples/04_vector_evaluate.py --num_envs "$VEC")
else
  ENTRY=(python -u /app/examples/02_evaluate.py)
fi

RESULTS="$REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
EVAL_LOG="$REALM_LOGS/pi05_rbm_evalout_${JOB}_${RUN_ID}.log"
START_EPOCH=$(date +%s)

apptainer run --userns --nv --writable-tmpfs --pwd /app \
  --bind "$REALM_ROOT":/app \
  --bind "$REALM_DATA":/data \
  --bind "$REALM_APPDATA":/cache \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT/tmp/$JOB":/tmp \
  --bind "$RBM_CLIENT_SRC":/robometer-client \
  "${OGLITE_BIND_ARGS[@]}" \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/app:/robometer-client \
  --env REALM_LIGHT_FIX="$REALM_LIGHT_FIX" \
  "$REALM_SIF" \
  "${ENTRY[@]}" \
    --task_id "$TASK_ID" --perturbation_id "$PERT_ID" \
    --repeats "$REPEATS" --max_steps "$MAX_STEPS" --horizon "$HORIZON" \
    --model_type openpi --model_name "$MODEL_NAME" \
    --port "$PORT" --host 127.0.0.1 \
    --experiment_name "$EXPERIMENT" --run_id "$RUN_ID" --log_dir /logs \
    --robot "$ROBOT" --rendering_mode rt \
    "${MULTIVIEW_ARG[@]}" \
    --robometer --robometer_host 127.0.0.1 --robometer_port "$RBM_PORT" \
    --robometer_success_threshold "$RBM_THRESHOLD" \
    --robometer_frame_size "$RBM_FRAME_SIZE" \
  2>&1 | tee "$EVAL_LOG"
EXIT=${PIPESTATUS[0]}

echo "[eval] exited $EXIT  (NOT a success signal -- see the gate below)"
printf '### EXIT_CODE=%s\n' "$EXIT" >> "$EVAL_LOG"
echo "[eval] artifacts: $RESULTS"
ls "$RESULTS" 2>/dev/null

python3 "$REALM_ROOT/scripts/clara/interactive/check_run.py" \
    "$RESULTS" "$EVAL_LOG" --repeats "$REPEATS" --newer-than "$START_EPOCH"
GATE=$?

if [ "$GATE" -ne 0 ]; then
  echo "==================================================================" >&2
  echo " EVAL FAILED -- $RUN_ID produced no complete, verified result set" >&2
  echo "   job=$JOB  run_id=$RUN_ID  task=$TASK_ID  repeats=$REPEATS" >&2
  echo "   eval log:  $EVAL_LOG" >&2
  echo "   robometer: $RBM_LOG" >&2
  echo "   DO NOT report numbers from this run." >&2
  echo "==================================================================" >&2
  exit 1
fi

echo "=================================================================="
echo " EVAL OK -- $RUN_ID: $REPEATS robometer-scored rollouts"
echo " artifacts: $RESULTS"
echo "=================================================================="
rm -rf "$REALM_ROOT/tmp/$JOB"
exit 0
