#!/bin/bash
# Policy server startup and shutdown utilities.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/lib/server.sh"
#
# start_policy_server requires these variables to be set by the caller:
#   MODEL_TYPE, port, REALM_ROOT
#   POLICY_RUN_DIR, POLICY_CONFIG, CHECKPOINT_PATH  (model-specific)
# Sets: SERVER_PID

start_policy_server() {
  if [ "$MODEL_TYPE" = "openpi" ]; then
    cd "$POLICY_RUN_DIR" || exit
    uv run scripts/serve_policy.py \
        --port=$port \
        policy:checkpoint \
        --policy.config=$POLICY_CONFIG \
        --policy.dir=$CHECKPOINT_PATH & SERVER_PID=$!
    sleep 120


  elif [ "$MODEL_TYPE" = "molmoact" ]; then
    cd "$POLICY_RUN_DIR" || exit
    export PYTHONPATH=$PYTHONPATH:$(pwd)/inference
    conda run --no-capture-output -n molmoact_inference python inference/run_molmoact_server.py \
        --ckpt allenai/MolmoAct-7B-D-0812 \
        --port $port \
        --host 127.0.0.1 \
        --host 0.0.0.0 & SERVER_PID=$!
    sleep 120

  elif [ "$MODEL_TYPE" == "GR00T" ]; then
    cd "$POLICY_RUN_DIR" || exit
    uv run scripts/serve_gr00t.py \
      --port=$port \
      --model_path $CHECKPOINT_PATH \
      --data-config droid_joint_pos & SERVER_PID=$!
    sleep 120

  elif [ "$MODEL_TYPE" == "GR00T_N16" ]; then
    export CUDA_HOME=/opt/apps/software/CUDA/12.8.0
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    cd "$POLICY_RUN_DIR" || exit
    uv run python gr00t/eval/run_gr00t_server.py \
      --embodiment-tag OXE_DROID \
      --use_sim_policy_wrapper \
      --model-path ${CHECKPOINT_PATH:-nvidia/GR00T-N1.6-DROID} \
      --port $port & SERVER_PID=$!
    sleep 120

  fi

  cd "$REALM_ROOT" || exit
}

# Kill the server started by start_policy_server, if any.
stop_policy_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
}
