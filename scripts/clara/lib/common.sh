#!/bin/bash
# Common defaults and utilities for REALM CLARA cluster scripts.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

#--- Default variable values ---------------------------------------------------


REALM_ROOT=$(pwd)
RUN_ID=$(date +%Y%m%d_%H%M%S)
DEBUG=false
RENDERING_MODE="rt"
MULTI_VIEW_FLAG=""
RESUME_FLAG=""
RESUME=false
TASK_CFG_PATH=""
NO_RENDER_FLAG=""
NO_RECORD_FLAG=""
ROBOT_FLAG=""
OG_LITE=false
OG_LITE_BIND=""
EXTRA_APPTAINER_ARGS=""
BASE_PORT=8000
MAX_STEPS=800
HORIZON=8
REPEATS=25
HOST="127.0.0.1"
SPP=8

#--- Utility functions ---------------------------------------------------------

# Expand comma-separated IDs and ranges into individual numbers.
# Example: "0,2-4,7" → "0\n2\n3\n4\n7"
expand_ids() {
  local input="$1"
  IFS=',' read -ra ADDR <<< "$input"
  for r in "${ADDR[@]}"; do
    if [[ "$r" =~ - ]]; then
      seq "${r%-*}" "${r#*-}"
    else
      echo "$r"
    fi
  done
}

# Parse SUPPORTED_TASKS and SUPPORTED_PERTURBATIONS from realm/eval.py via AST.
# Sets global bash arrays ALL_TASKS and ALL_PERTS.
extract_task_pert_names() {
  local pyout
  pyout=$(python3 << 'PYEOF'
import ast, sys
try:
    with open('realm/eval.py', 'r') as f:
        tree = ast.parse(f.read())
    tasks, perts = [], []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'SUPPORTED_TASKS':
                    tasks = ast.literal_eval(node.value)
                if isinstance(target, ast.Name) and target.id == 'SUPPORTED_PERTURBATIONS':
                    perts = ast.literal_eval(node.value)
    print('ALL_TASKS=(' + ' '.join([f'"{t}"' for t in tasks]) + ')')
    print('ALL_PERTS=(' + ' '.join([f'"{p}"' for p in perts]) + ')')
except Exception as e:
    print(f"Error parsing realm/eval.py: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
  if [ $? -ne 0 ]; then
    echo "Error: Failed to extract task/perturbation names from realm/eval.py. Aborting."
    exit 1
  fi
  eval "$pyout"
}

# Set OG_LITE_BIND and OG_LITE_FLAG based on the $OG_LITE flag.
# Call after arg parsing; results are read by apptainer_exec (apptainer.sh) and
# passed to 02_evaluate.py via $OG_LITE_FLAG.
compute_og_lite_bind() {
  OG_LITE_BIND=""
  OG_LITE_FLAG=""
  if [ "$OG_LITE" = "true" ]; then
    OG_LITE_BIND="--bind $REALM_ROOT/../OG-lite:/omnigibson-src"
    OG_LITE_FLAG="--og_lite"
  fi
}

# Derive MODEL_NAME from MODEL_TYPE / CHECKPOINT_PATH / DEBUG.
compute_model_name() {
  if [ "$DEBUG" = "true" ]; then
    MODEL_NAME="debug"
  elif [ "$MODEL_TYPE" = "molmoact" ]; then
    MODEL_NAME="molmoact"
  elif [ "$MODEL_TYPE" = "GR00T_N16" ]; then
    MODEL_NAME="GR00T_N16"
  elif [ "$MODEL_TYPE" = "hamster" ]; then
    MODEL_NAME="hamster"
  elif [ "$MODEL_TYPE" = "dreamzero" ]; then
    MODEL_NAME="dreamzero"
  else
    local clean="${CHECKPOINT_PATH%/}"
    MODEL_NAME=$(basename "$(dirname "${clean%/}")")_$(basename "${clean%/}")
  fi
}

# Export HuggingFace / XDG cache directories under REALM_ROOT.
setup_hf_cache() {
  export HF_HOME=$REALM_ROOT/hf_cache
  export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
  [[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"
  export XDG_CACHE_HOME=$REALM_ROOT/python_cache
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
}

# Create per-job temporary / cache directories.
setup_job_dirs() {
  mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
  mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
  mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
}

# Remove per-job temp dirs on success; preserve them on failure for debugging.
# Usage: cleanup_job_dirs <exit_code> [label]
cleanup_job_dirs() {
  local exit_code=$1
  local label="${2:-Job}"
  if [ "$exit_code" -eq 0 ]; then
    echo "$label finished successfully. Cleaning up..."
    rm -rf "$REALM_ROOT/tmp/$SLURM_JOB_ID"
    rm -rf "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
    rm -rf "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
  else
    echo "$label failed (exit code $exit_code). Preserving temporary directories for debugging."
  fi
}
