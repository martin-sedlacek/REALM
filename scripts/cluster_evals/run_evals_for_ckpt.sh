#!/bin/bash

unset EXPERIMENT_NAME T_RAW P_RAW TASK_IDS PERT_IDS
mkdir -p "$REALM_ROOT/tmp"

#---------------------------------------------------------------------------------

BASE_PORT=8000
MAX_STEPS=800
REPEATS=25
RUN_ID=$(date +%Y%m%d_%H%M%S)
DEBUG=false
MULTI_VIEW_FLAG=""
RESUME_FLAG=""
RESUME=false
ROBOT_FLAG=""
RENDERING_MODE="rt"


expand_ids() {
  echo "$1" | tr ',' '\n' | while read -r r; do
    [[ "$r" =~ - ]] && seq "${r%-*}" "${r#*-}" || echo "$r"
  done
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config) POLICY_CONFIG="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --policy_run_dir) POLICY_RUN_DIR="$2"; shift 2 ;;
    --base_port|--base-port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_ids) T_RAW="$2"; mapfile -t TASK_IDS < <(expand_ids "$2"); shift 2 ;;
    --task_cfg_path) TASK_CFG_PATH="$2"; shift 2 ;;
    --perturbation_ids) P_RAW="$2"; mapfile -t PERT_IDS < <(expand_ids "$2"); shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --debug) DEBUG=true; shift 1;;
    --multi-view) MULTI_VIEW_FLAG="--multi-view"; shift 1;;
    --no_render) NO_RENDER_FLAG="--no_render"; shift 1;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --resume) RESUME=true; RESUME_FLAG="--resume"; shift 1;;
    --rendering_mode) RENDERING_MODE="$2"; shift 2 ;;
    --robot) ROBOT_FLAG="--robot $2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ ${#TASK_IDS[@]} -eq 0 ]; then
  T_RAW="0-9"
  mapfile -t TASK_IDS < <(expand_ids "$T_RAW")
fi

if [ ${#PERT_IDS[@]} -eq 0 ]; then
  P_RAW="0-15"
  mapfile -t PERT_IDS < <(expand_ids "$P_RAW")
fi

if [ -z "$EXPERIMENT_NAME" ]; then
  EXPERIMENT_NAME="t${T_RAW//,/_}_p${P_RAW//,/_}_s${MAX_STEPS}_r${REPEATS}"
fi

METADATA_DIR="logs/$EXPERIMENT_NAME"
mkdir -p "$METADATA_DIR"
METADATA_FILE="$METADATA_DIR/metadata.json"

{
  echo "{"
  echo "  \"max_steps\": $MAX_STEPS,"
  echo "  \"repeats\": $REPEATS,"
  echo "  \"task_ids\": [${T_RAW}]",
  echo "  \"perturbation_ids\": [${P_RAW}]"
  echo "}"
} > "$METADATA_FILE"

# Extract mappings using lightweight AST parsing to avoid importing heavyweight modules
PYTHON_SCRIPT=$(cat <<'EOF'
import ast
import sys
try:
    with open('realm/eval.py', 'r') as f:
        tree = ast.parse(f.read())
    tasks = []
    perts = []
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
EOF
)
eval "$(python3 -c "$PYTHON_SCRIPT")"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract tasks/perturbations. Aborting."
    exit 1
fi


# Determine MODEL_NAME for path construction
if [ "$DEBUG" = "true" ]; then
  MODEL_NAME="debug"
elif [ "$MODEL_TYPE" = "molmoact" ]; then
  MODEL_NAME="molmoact"
else
  CLEAN_PATH="${CHECKPOINT_PATH%/}"
  MODEL_NAME=$(basename "$(dirname "${CLEAN_PATH%/}")")_$(basename "${CLEAN_PATH%/}")
fi
VIDEO_DIR="logs/$EXPERIMENT_NAME/$MODEL_NAME/$RUN_ID/videos"

#---------------------------------------------------------------------------------

DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
  DEBUG_FLAG="--debug"
fi

for i in "${TASK_IDS[@]}"; do
  for j in "${PERT_IDS[@]}"; do
    if [ "$RESUME" = "true" ]; then
        TASK_NAME=${ALL_TASKS[$i]}
        PERT_NAME=${ALL_PERTS[$j]}
        # Check specific task/pert completion
        COUNT=$(ls "$VIDEO_DIR/${TASK_NAME}_${PERT_NAME}_"*.mp4 2>/dev/null | wc -l)
        if [ "$COUNT" -ge "$REPEATS" ]; then
            echo "Skipping Task $i ($TASK_NAME) Pert $j ($PERT_NAME): Found $COUNT/$REPEATS videos."
            continue
        fi
    fi

    if [ -n "$TASK_CFG_PATH" ]; then
      TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH"
    else
      TASK_CFG_ARG=""
    fi

    sbatch scripts/cluster_evals/run_single_eval.sh \
      --task_id "$i" \
      $TASK_CFG_ARG \
      --perturbation_id "$j" \
      --repeats "$REPEATS" \
      --max_steps "$MAX_STEPS" \
      --policy_config "$POLICY_CONFIG" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --policy_run_dir "$POLICY_RUN_DIR" \
      --base_port "$BASE_PORT" \
      --experiment_name "$EXPERIMENT_NAME" \
      --run_id "$RUN_ID" \
      --model_type "$MODEL_TYPE" \
      --rendering_mode "$RENDERING_MODE" \
      $DEBUG_FLAG \
      $MULTI_VIEW_FLAG \
      $RESUME_FLAG \
      $NO_RENDER_FLAG \
      $ROBOT_FLAG
  done
done
