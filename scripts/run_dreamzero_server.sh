#!/usr/bin/env bash
# Launch the DreamZero inference server on 2x H200 GPUs.
#
# Two run modes:
#   1) --allocate   : submits the srun allocation itself, then re-execs this
#                     script inside the compute node.
#   2) (default)    : assumes you are already on a compute node (e.g. from an
#                     existing srun/sbatch shell) and starts the server.
#
# Usage examples:
#   # Request fresh allocation + launch server on port 5000
#   bash scripts/run_dreamzero_server.sh --allocate
#
#   # Already inside an srun shell — just launch on port 5001
#   bash scripts/run_dreamzero_server.sh --port 5001
#
#   # Custom checkpoint
#   bash scripts/run_dreamzero_server.sh --port 5002 \
#        --checkpoint checkpoints/dreamzero_droid_3epoch_h200
#
# Notes:
#   * Each concurrent server needs a DIFFERENT --port.
#   * Script prints the node's hostname + IP + port so the evaluation client
#     (realm/inference/dreamzero.py) can dial in.
#   * Expects the dreamzero repo at ../dreamzero relative to REALM, and a
#     working `dreamzero` conda env.

set -eo pipefail
# NOTE: intentionally NOT using `set -u` — dreamzero's conda-env activate
# hooks reference unset vars (e.g. NVCC_PREPEND_FLAGS) and would otherwise
# abort the server bringup.

DEFAULT_PORT=5000
DEFAULT_CHECKPOINT="checkpoints/DreamZero-DROID"
DREAMZERO_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../dreamzero" 2>/dev/null && pwd || echo "")"

PORT="$DEFAULT_PORT"
CHECKPOINT="$DEFAULT_CHECKPOINT"
DREAMZERO_DIR="$DREAMZERO_DIR_DEFAULT"
ALLOCATE=0
TIME_LIMIT="30:00:00"
CONDA_ENV="dreamzero"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)           PORT="$2"; shift 2 ;;
    --checkpoint)     CHECKPOINT="$2"; shift 2 ;;
    --dreamzero-dir)  DREAMZERO_DIR="$2"; shift 2 ;;
    --conda-env)      CONDA_ENV="$2"; shift 2 ;;
    --time)           TIME_LIMIT="$2"; shift 2 ;;
    --allocate)       ALLOCATE=1; shift 1 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DREAMZERO_DIR" || ! -d "$DREAMZERO_DIR" ]]; then
  echo "ERROR: dreamzero repo not found. Pass --dreamzero-dir /path/to/dreamzero" >&2
  exit 1
fi

#------------------------------------------------------------------------------
# Mode 1: request the srun allocation, then re-exec inside compute node.
#------------------------------------------------------------------------------
if [[ "$ALLOCATE" -eq 1 ]]; then
  echo "[allocate] requesting 2x H200 for ${TIME_LIMIT}..."
  exec srun \
    --partition=h200 \
    --gres=gpu:2 \
    --cpus-per-gpu=32 \
    --mem-per-gpu=250G \
    --gpu-bind=closest \
    --time="${TIME_LIMIT}" \
    --pty bash "$0" \
      --port "$PORT" \
      --checkpoint "$CHECKPOINT" \
      --dreamzero-dir "$DREAMZERO_DIR" \
      --conda-env "$CONDA_ENV"
fi

#------------------------------------------------------------------------------
# Mode 2: running on the compute node — start the server.
#------------------------------------------------------------------------------
NODE_HOST="$(hostname -s)"
NODE_FQDN="$(hostname -f 2>/dev/null || hostname)"
# Prefer a routable IP; fall back to hostname -I.
NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -z "$NODE_IP" ]] && NODE_IP="$(getent hosts "$NODE_FQDN" | awk '{print $1}' | head -n1)"

cat <<EOF
============================================================
 DreamZero server bringup
------------------------------------------------------------
 node host     : ${NODE_HOST}
 node fqdn     : ${NODE_FQDN}
 node ip       : ${NODE_IP}
 port          : ${PORT}
 checkpoint    : ${CHECKPOINT}
 dreamzero dir : ${DREAMZERO_DIR}
 conda env     : ${CONDA_ENV}
------------------------------------------------------------
 >>> CLIENT ENDPOINT: ${NODE_IP}:${PORT} (host=${NODE_HOST}) <<<
============================================================
EOF

cd "$DREAMZERO_DIR"

# Initialise conda in this non-interactive shell.
if [[ -z "${CONDA_EXE:-}" ]]; then
  for candidate in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
    [[ -f "$candidate" ]] && { source "$candidate"; break; }
  done
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES=0,1

exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  socket_test_optimized_AR.py \
    --port "$PORT" \
    --enable-dit-cache \
    --model-path "$CHECKPOINT"
