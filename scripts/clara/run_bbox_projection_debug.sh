#!/bin/bash
#SBATCH --job-name realm-bbox-debug
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 00-00:30:00

# Single-step debug eval that draws the projected 3D bbox of the main/target
# objects on each external camera image. Output PNGs land in:
#   $REALM_ROOT/logs/bbox_debug/<timestamp>/
#
# Usage (defaults shown):
#   sbatch scripts/clara/run_bbox_projection_debug.sh \
#     --task_id 0 --perturbation_id 0 --multi-view

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/apptainer.sh"

TASK_ID=0
PERTURBATION_ID=0
MULTI_VIEW_FLAG=""
OUTPUT_DIR="/app/logs/bbox_debug"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --task_id)         TASK_ID="$2";                   shift 2 ;;
    --perturbation_id) PERTURBATION_ID="$2";           shift 2 ;;
    --multi-view)      MULTI_VIEW_FLAG="--multi_view"; shift 1 ;;
    --rendering_mode)  RENDERING_MODE="$2";            shift 2 ;;
    --output_dir)      OUTPUT_DIR="$2";                shift 2 ;;
    --og_lite)         OG_LITE=true;                   shift 1 ;;
    *) shift ;;
  esac
done

compute_og_lite_bind

cd "$REALM_ROOT" || exit
setup_job_dirs

echo "Running 2D bbox projection debug for task=$TASK_ID perturbation=$PERTURBATION_ID..."

apptainer_eval "python scripts/debug/debug_project_bbox.py \
  --task_id $TASK_ID \
  --perturbation_id $PERTURBATION_ID \
  --rendering_mode $RENDERING_MODE \
  --output_dir $OUTPUT_DIR \
  $MULTI_VIEW_FLAG"

EXIT_CODE=$?
cleanup_job_dirs $EXIT_CODE "BBox debug"
exit $EXIT_CODE
