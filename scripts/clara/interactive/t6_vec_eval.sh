#!/bin/bash
# Vectorized pi0.5 evaluation. Needs the pi0.5 server on :8000 and MODE=oglite (the scene z-offset
# fix that makes scenes 1..N-1 usable lives in the fork).
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
NUM_ENVS=${NUM_ENVS:-4}; REPEATS=${REPEATS:-25}; MAX_STEPS=${MAX_STEPS:-500}
RUN_ID=${RUN_ID:-vec}; EXPERIMENT=${EXPERIMENT:-vec_pi05}
ROD=${ROD:-1}
ROD_FLAG="--no-render_on_demand"
[ "$ROD" = "1" ] && ROD_FLAG="--render_on_demand"
python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT:-8000}))==0 else 1)" \
  || { echo "### no policy server on :${PORT:-8000}" >&2; exit 1; }
echo "### vec eval: num_envs=$NUM_ENVS repeats=$REPEATS max_steps=$MAX_STEPS run_id=$RUN_ID rod=$ROD"
MODE=oglite ./scripts/clara/interactive/rr \
  python -u examples/04_vector_evaluate.py \
    --num_envs "$NUM_ENVS" --repeats "$REPEATS" --max_steps "$MAX_STEPS" --horizon 8 \
    --task_id "${TASK_ID:-0}" --perturbation_id "${PERT_ID:-0}" \
    --model_type openpi --model_name "${MODEL_NAME:-checkpoints_pi05_droid_jointpos}" \
    --port "${PORT:-8000}" --host 127.0.0.1 \
    --experiment_name "$EXPERIMENT" --run_id "$RUN_ID" --log_dir /logs \
    --robot DROID --rendering_mode rt $ROD_FLAG
echo "### vec eval exit: $?"
