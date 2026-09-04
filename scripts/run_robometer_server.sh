#!/usr/bin/env bash
# Start a Robometer reward-model eval server from the pinned submodule at packages/robometer.
#
# This runs OUTSIDE the REALM simulation container, in robometer's own uv environment (Python 3.10,
# torch 2.8, transformers 4.57 -- none of which can coexist with OmniGibson's pins). Give it its
# own GPU: Robometer-4B needs roughly 10-12 GB. REALM reaches it through the vendored
# packages/robometer-client (robometer_client.RobometerClient), the same arrangement as the openpi
# policy server.
#
# First time only:
#     git submodule update --init packages/robometer
#
# Usage:
#     ./scripts/run_robometer_server.sh                       # Robometer-4B on 0.0.0.0:8010, 1 GPU
#     ROBOMETER_PORT=8020 ./scripts/run_robometer_server.sh
#     ./scripts/run_robometer_server.sh num_gpus=2            # extra args pass through to hydra
#
# Environment:
#     ROBOMETER_ROOT      submodule checkout (default packages/robometer)
#     ROBOMETER_MODEL     HF id or local checkpoint (default robometer/Robometer-4B)
#     ROBOMETER_HOST      bind address (default 0.0.0.0)
#     ROBOMETER_PORT      port (default 8010, the evaluators' --robometer_port default; the policy
#                         server usually holds 8000, so the two run side by side)
#     ROBOMETER_NUM_GPUS  model replicas, one per GPU (default 1)
#
# Verify from anywhere that can reach the host:
#     curl -s http://<host>:<port>/health        # {"status":"healthy",...}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-$REPO_ROOT/packages/robometer}"
ROBOMETER_MODEL="${ROBOMETER_MODEL:-robometer/Robometer-4B}"
ROBOMETER_HOST="${ROBOMETER_HOST:-0.0.0.0}"
ROBOMETER_PORT="${ROBOMETER_PORT:-8010}"
ROBOMETER_NUM_GPUS="${ROBOMETER_NUM_GPUS:-1}"

if [ ! -f "$ROBOMETER_ROOT/pyproject.toml" ]; then
    echo "robometer checkout not found at $ROBOMETER_ROOT" >&2
    echo "run: git submodule update --init packages/robometer   (or set ROBOMETER_ROOT)" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run the robometer server: https://github.com/astral-sh/uv" >&2
    exit 1
fi

cd "$ROBOMETER_ROOT"
echo "robometer @ $(git rev-parse --short HEAD 2>/dev/null || echo '?') -> ${ROBOMETER_MODEL} on ${ROBOMETER_HOST}:${ROBOMETER_PORT} (${ROBOMETER_NUM_GPUS} GPU)"

# First run: build the environment. robometer ships no lockfile, so this resolves to whatever is
# current on PyPI, and one thing is known to drift: torchao (pulled in through transformers'
# quantizer imports) >= 0.14 needs torch >= 2.9, while robometer pins torch == 2.8.0 -- the server
# then dies at import with "cannot import name 'ScalingType' from 'torch.nn.functional'". Pin it
# back to the last release built against 2.8. --extra robometer pulls the transformers/trl versions
# the released checkpoints load with; the bare `uv sync` in robometer's README installs neither.
if [ ! -x .venv/bin/python ]; then
    echo "--- building robometer env (first run; several GB) ---"
    uv sync --extra robometer
    uv pip install --python .venv/bin/python "torchao<0.14"
fi
# --no-sync: a plain `uv run` re-syncs and would undo the torchao pin above.
# scripts/robometer_server.py wraps robometer/evals/eval_server.py to load the checkpoint without
# unsloth -- with current unsloth the stock server dies on its first request; see its docstring.
exec uv run --no-sync python "$REPO_ROOT/scripts/robometer_server.py" \
    model_path="$ROBOMETER_MODEL" \
    server_url="$ROBOMETER_HOST" \
    server_port="$ROBOMETER_PORT" \
    num_gpus="$ROBOMETER_NUM_GPUS" \
    "$@"
