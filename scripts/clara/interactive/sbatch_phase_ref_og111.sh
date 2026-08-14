#!/bin/bash
# Reference phase profile on the PRE-PORT stack: REALM@dev + OG-lite, OmniGibson 1.1.1.
#
# Produces the cold-start / reset / warmup / per-step numbers that the og391 port gets compared
# against. Submit the matching og391 job (sbatch_phase_ref_og391.sh) with identical eval arguments;
# the two differ only in stack.
#
#   sbatch scripts/clara/interactive/sbatch_phase_ref_og111.sh
#   REPEATS=3 MAX_STEPS=100 sbatch scripts/clara/interactive/sbatch_phase_ref_og111.sh
#
# Runs on its own node -- deliberately NOT on the held interactive allocation, which is busy.
#
#SBATCH --job-name realm-phase-og111
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 01:30:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/phase_ref_og111_%j.log

set -uo pipefail

# DELIBERATELY the pre-port tree, and therefore deliberately NOT sourced from lib/paths.sh, which
# resolves the og391 stack. This is the one script here that is supposed to point elsewhere.
REALM_ROOT=${REALM_OG111_ROOT:-/mnt/home_lustre/sedlam56/projects/REALM}  # 1.1.1 checkout, branch dev
OGLITE_ROOT=/mnt/home_lustre/sedlam56/projects/OG-lite       # 1.1.1 fork
REALM_DATA=$REALM_ROOT/data                                  # datasets/ is the 1.1.1 tree
REALM_LOGS=$REALM_ROOT/logs
REALM_SIF=/home/sedlam56/apptainer/realm-dm.sif              # the 1.1.1 image (per ~/.bashrc)

# A GUARD, not decoration. ~/projects/REALM is the 1.1.1 checkout today, but it is also the
# DESTINATION of the pending REALM_og391 -> REALM rename. The moment that happens, the path above
# silently becomes the PORTED tree: this job would copy the profiler into it (see below) and profile
# 3.9.1 code with the 1.1.1 image while labelling the output "og111" -- a wrong historical reference
# that nothing else would catch. Two markers separate the trees: the 1.1.1 tree has
# realm/misc/modified_entity_prim.py; the ported one has realm/misc/*_og391.patch.
#
# After the rename, point REALM_OG111_ROOT at wherever the 1.1.1 checkout ended up. Do not edit the
# default to make this pass.
[ -f "$REALM_ROOT/realm/misc/modified_entity_prim.py" ] || {
  echo "ERROR: $REALM_ROOT is not the OmniGibson 1.1.1 checkout (no realm/misc/modified_entity_prim.py)." >&2
  echo "       Set REALM_OG111_ROOT to the 1.1.1 tree. This job is a reference for the OLD stack." >&2
  exit 1; }
if compgen -G "$REALM_ROOT/realm/misc/*_og391.patch" >/dev/null; then
  echo "ERROR: $REALM_ROOT is the PORTED og391 checkout, not the 1.1.1 one -- it carries" >&2
  echo "       realm/misc/*_og391.patch. The rename to ~/projects/REALM has happened; set" >&2
  echo "       REALM_OG111_ROOT to wherever the 1.1.1 tree now lives." >&2
  exit 1
fi

TASK_ID=${TASK_ID:-0}
PERT_ID=${PERT_ID:-0}
REPEATS=${REPEATS:-3}
MAX_STEPS=${MAX_STEPS:-100}
HORIZON=${HORIZON:-8}
# Stays on stock DROID: this is the PRE-PORT 1.1.1 checkout, which has no robolab config at all --
# neither v1 nor v2 (the robolab asset is og391-only work). The point of this job is a historical
# reference for the old stack, so its robot has to be one the old stack actually has.
ROBOT=${ROBOT:-DROID}
# OG-lite on-demand rendering, so this matches og391's render_on_demand default (ON).
OG_LITE_RENDER=${OG_LITE_RENDER:-1}
LABEL=${LABEL:-og111_oglite}
JOB=${SLURM_JOB_ID:-local}
OUT=/logs/phase_ref/${LABEL}_${JOB}.json

#--- fail fast, before booting a simulator --------------------------------------------------------
[ -f "$REALM_SIF" ]                     || { echo "ERROR: no SIF at $REALM_SIF" >&2; exit 1; }
[ -d "$REALM_DATA/datasets" ]           || { echo "ERROR: no 1.1.1 dataset at $REALM_DATA/datasets" >&2; exit 1; }
[ -d "$OGLITE_ROOT/omnigibson" ]        || { echo "ERROR: no OG-lite at $OGLITE_ROOT" >&2; exit 1; }
# The 1.1.1 checkout is a separate repo, so the profiler has to be staged into it rather than
# read from this one's scripts/. Keep the two byte-identical -- that is the whole point of running
# one profiler against both stacks.
cp scripts/clara/interactive/profile_phases.py "$REALM_ROOT/tmp/profile_phases.py" 2>/dev/null
[ -f "$REALM_ROOT/tmp/profile_phases.py" ] || { echo "ERROR: could not stage profiler into $REALM_ROOT/tmp" >&2; exit 1; }
# /tmp is node-local and wiped; every artifact must land on Lustre.
case "$REALM_LOGS" in /tmp/*) echo "ERROR: refusing to write artifacts under /tmp" >&2; exit 1;; esac

mkdir -p "$REALM_ROOT/tmp/$JOB" "$REALM_ROOT/mamba_cache/$JOB" "$REALM_ROOT/pip_cache/$JOB" \
         "$REALM_LOGS/phase_ref"

OG_LITE_FLAG=""
[ "$OG_LITE_RENDER" = "1" ] && OG_LITE_FLAG="--og_lite"

echo "=================================================================="
echo " PRE-PORT reference phase profile -- OmniGibson 1.1.1 + OG-lite"
echo " label=$LABEL  task=$TASK_ID  pert=$PERT_ID  repeats=$REPEATS  max_steps=$MAX_STEPS"
echo " og_lite render-on-demand: $OG_LITE_RENDER   robot=$ROBOT   model=debug"
echo " sif     = $REALM_SIF"
echo " dataset = $REALM_DATA/datasets"
echo " OG-lite = $OGLITE_ROOT  -> /omnigibson-src"
echo " out     = $REALM_LOGS/phase_ref/${LABEL}_${JOB}.json"
echo " node    = $(hostname)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# A GPU Slurm hands you is not necessarily empty -- check for squatters before trusting timings.
echo "--- other compute apps on this GPU (want: empty) ---"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

apptainer exec \
  --userns --nv --writable-tmpfs --pwd /app \
  --bind "$REALM_ROOT":/app \
  --bind "$OGLITE_ROOT":/omnigibson-src \
  --bind "$REALM_DATA"/datasets:/data \
  --bind "$REALM_DATA"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
  --bind "$REALM_DATA"/isaac-sim/cache/ov:/root/.cache/ov \
  --bind "$REALM_DATA"/isaac-sim/cache/pip:/root/.cache/pip \
  --bind "$REALM_DATA"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind "$REALM_DATA"/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
  --bind "$REALM_DATA"/isaac-sim/logs:/root/.nvidia-omniverse/logs \
  --bind "$REALM_DATA"/isaac-sim/config:/root/.nvidia-omniverse/config \
  --bind "$REALM_DATA"/isaac-sim/data:/root/.local/share/ov/data \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT"/tmp/"$JOB":/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONUNBUFFERED=1 \
  --env MAMBA_CACHE_DIR="$REALM_ROOT"/mamba_cache/"$JOB" \
  --env PIP_CACHE_DIR="$REALM_ROOT"/pip_cache/"$JOB" \
  "$REALM_SIF" \
  micromamba run -n omnigibson bash -c "
    pip install json_numpy --quiet &&
    python -u /app/tmp/profile_phases.py --out $OUT --label $LABEL -- \
      --task_id $TASK_ID \
      --perturbation_id $PERT_ID \
      --repeats $REPEATS \
      --max_steps $MAX_STEPS \
      --horizon $HORIZON \
      --model_name debug \
      --model_type debug \
      --port 8000 \
      --robot $ROBOT \
      --experiment_name phase_ref \
      --run_id ${LABEL}_${JOB} \
      --log_dir /logs \
      --rendering_mode rt \
      $OG_LITE_FLAG
  "
EXIT=$?

echo "[phase_ref] exited $EXIT"
echo "[phase_ref] json: $REALM_LOGS/phase_ref/${LABEL}_${JOB}.json"
[ "$EXIT" -eq 0 ] && rm -rf "$REALM_ROOT/tmp/$JOB" "$REALM_ROOT/mamba_cache/$JOB" "$REALM_ROOT/pip_cache/$JOB"
exit $EXIT
