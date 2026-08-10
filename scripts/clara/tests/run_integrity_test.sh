#!/bin/bash
#SBATCH --job-name realm-integrity-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 40G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 32
#SBATCH --time 00-01:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
source "$SCRIPT_DIR/../lib/apptainer.sh"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --og_lite) OG_LITE=true; shift 1 ;;
    *) shift ;;
  esac
done

compute_og_lite_bind

cd "$REALM_ROOT" || exit
setup_job_dirs

echo "Running Task Integrity Test..."

apptainer_eval "python tests/test_integrity.py"

EXIT_CODE=$?
cleanup_job_dirs $EXIT_CODE "Test"
exit $EXIT_CODE
