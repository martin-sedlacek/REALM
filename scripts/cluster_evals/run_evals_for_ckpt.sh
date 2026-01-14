#!/bin/bash

unset EXPERIMENT_NAME T_RAW P_RAW TASK_IDS PERT_IDS
mkdir -p "$REALM_ROOT/tmp"

#---------------------------------------------------------------------------------

BASE_PORT=8000
MAX_STEPS=800
REPEATS=25
RUN_ID=$(date +%Y%m%d_%H%M%S)
DEBUG=false

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
    --base_port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_ids) T_RAW="$2"; mapfile -t TASK_IDS < <(expand_ids "$2"); shift 2 ;;
    --perturbation_ids) P_RAW="$2"; mapfile -t PERT_IDS < <(expand_ids "$2"); shift 2 ;;
    --debug) DEBUG=true; shift 1;;
    *) shift ;;
  esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
  EXPERIMENT_NAME="t${T_RAW//,/_}_p${P_RAW//,/_}_s${MAX_STEPS}_r${REPEATS}"
fi

#---------------------------------------------------------------------------------

for i in "${TASK_IDS[@]}"; do
  for j in "${PERT_IDS[@]}"; do
    sbatch scripts/cluster_evals/run_single_eval.sh \
      --task_id "$i" \
      --perturbation_id "$j" \
      --repeats "$REPEATS" \
      --max_steps "$MAX_STEPS" \
      --policy_config "$POLICY_CONFIG" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --policy_run_dir "$POLICY_RUN_DIR" \
      --base_port "$BASE_PORT" \
      --experiment_name "$EXPERIMENT_NAME" \
      --run_id "$RUN_ID" \
      --debug "$DEBUG"
  done
done