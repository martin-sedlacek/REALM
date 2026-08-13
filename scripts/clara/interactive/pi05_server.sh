#!/bin/bash
# Start the pi0.5 policy server on the held allocation and leave it resident.
#
# Needed for the contact-cache A/B: --model_type debug returns a CONSTANT action
# (np.zeros(8) for joint control, realm/inference/client.py:33), so the gripper never touches
# anything and the contact matrix never leaves its cheap regime. The perf doc measured the cache
# as bimodal -- ~23-28 ms most steps, ~300 ms on ~28% of them, the spikes appearing exactly when
# the gripper contacts an object. A debug-action A/B would measure only the cheap mode.
#
#   ./scripts/clara/interactive/pi05_server.sh          # foreground; run it under `go` in the background
#
# Recipe lifted from run_og391_smoke_pi05.sh section 1 / runbook entry 2026-08-12.
# ~70 s to come up, ~11.8 G VRAM at XLA_PYTHON_CLIENT_MEM_FRACTION=0.25, leaving ~34 G for the sim.
set -uo pipefail

PORT=${PORT:-8000}
CKPT=${CKPT:-/home/sedlam56/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos}
CONFIG=${CONFIG:-pi05_full_droid_finetune}

[ -d "$CKPT/params" ] || { echo "no params/ under $CKPT" >&2; exit 1; }

cd /mnt/home_lustre/sedlam56/projects/openpi || exit 1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export HF_HUB_OFFLINE=1          # compute nodes have no outbound internet
exec uv run scripts/serve_policy.py \
    --port="$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT"
