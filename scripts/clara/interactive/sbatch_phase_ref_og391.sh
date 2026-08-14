#!/bin/bash
# Matching phase profile on the PORTED stack: REALM_og391 + OmniGibson 3.9.1.
#
# Identical eval arguments to sbatch_phase_ref_og111.sh so the two are directly comparable; the
# only difference is the stack. Run both, then diff with scripts/clara/interactive/compare_phases.py.
#
#   sbatch scripts/clara/interactive/sbatch_phase_ref_og391.sh                 # stock 3.9.1
#   OGLITE=1 sbatch scripts/clara/interactive/sbatch_phase_ref_og391.sh        # with the OG-lite fork bound
#
#SBATCH --job-name realm-phase-og391
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 01:30:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/phase_ref_og391_%j.log

set -uo pipefail

# REALM_ROOT / REALM_SIF / REALM_DATA / REALM_APPDATA / REALM_LOGS / REALM_OGLITE_ROOT, all derived
# from this script's own location rather than written out here, so a worktree profiles ITS OWN
# checkout and moving the tree costs no edit. Read paths.sh before adding a path of your own: the
# profile EXPORTS REALM_ROOT and REALM_SIF pointing at the pre-port 1.1.1 tree and image, so
# `${REALM_ROOT:-<default>}` silently selects the wrong stack.
#
# The #SBATCH --output above cannot use any of this -- Slurm parses those directives before a shell
# exists. It names the shared log tree, which the rename does not move, so it stays literal.
#
# Locating paths.sh needs more than ${BASH_SOURCE[0]} here. Under sbatch, Slurm ships this script's
# TEXT to the node and runs a copy at /var/spool/slurmd/job<N>/slurm_script, so BASH_SOURCE points
# into the spool dir -- verified 2026-08-14 by probe job 191043. `scontrol show job` still reports
# the absolute path sbatch was handed; $SLURM_SUBMIT_DIR is the last resort. Every candidate is
# tested before use, and not finding paths.sh is FATAL: set -e is off, so carrying on would leave
# $REALM_ROOT at the value the shell profile exports -- the PRE-PORT 1.1.1 checkout.
_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" 2>/dev/null && pwd)
if [ ! -f "${_lib:-/nonexistent}/paths.sh" ]; then
  _cmd=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
  _lib=$(cd "$(dirname "${_cmd:-/nonexistent}")/../lib" 2>/dev/null && pwd)
fi
[ -f "${_lib:-/nonexistent}/paths.sh" ] || _lib=${SLURM_SUBMIT_DIR:-$PWD}/scripts/clara/lib
[ -f "$_lib/paths.sh" ] || { echo "ERROR: cannot locate scripts/clara/lib/paths.sh (BASH_SOURCE=${BASH_SOURCE[0]} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 1; }
source "$_lib/paths.sh"
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source $_lib/paths.sh" >&2; exit 1; }

TASK_ID=${TASK_ID:-0}
PERT_ID=${PERT_ID:-0}
REPEATS=${REPEATS:-3}
MAX_STEPS=${MAX_STEPS:-100}
HORIZON=${HORIZON:-8}
ROBOT=${ROBOT:-DROID_robolab_v2}
# render_on_demand is ON by default in this branch, which is what --og_lite approximates on 1.1.1.
ROD=${ROD:-1}
OGLITE=${OGLITE:-0}
LABEL=${LABEL:-og391_$([ "$OGLITE" = "1" ] && echo oglite || echo stock)}
JOB=${SLURM_JOB_ID:-local}
OUT=/logs/phase_ref/${LABEL}_${JOB}.json

[ -f "$REALM_SIF" ]                        || { echo "ERROR: no SIF at $REALM_SIF" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ]    || { echo "ERROR: no og391 dataset at $REALM_DATA" >&2; exit 1; }
[ -f "$REALM_ROOT/scripts/clara/interactive/profile_phases.py" ] || { echo "ERROR: no profiler at $REALM_ROOT/scripts/clara/interactive" >&2; exit 1; }
case "$REALM_LOGS" in /tmp/*) echo "ERROR: refusing to write artifacts under /tmp" >&2; exit 1;; esac

mkdir -p "$REALM_ROOT/tmp/$JOB" "$REALM_APPDATA/appdata" "$REALM_LOGS/phase_ref"

OGLITE_BIND=()
[ "$OGLITE" = "1" ] && OGLITE_BIND=(--bind "$REALM_OGLITE_ROOT/omnigibson":/behavior-src/OmniGibson/omnigibson)
ROD_FLAG="--no-render_on_demand"
[ "$ROD" = "1" ] && ROD_FLAG="--render_on_demand"

echo "=================================================================="
echo " PORTED phase profile -- OmniGibson 3.9.1 (oglite=$OGLITE)"
echo " label=$LABEL  task=$TASK_ID  pert=$PERT_ID  repeats=$REPEATS  max_steps=$MAX_STEPS"
echo " render_on_demand=$ROD  robot=$ROBOT  model=debug"
echo " out  = $REALM_LOGS/phase_ref/${LABEL}_${JOB}.json"
echo " node = $(hostname)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "--- other compute apps on this GPU (want: empty) ---"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# `apptainer run`, not exec: the %runscript activates conda env `behavior`.
apptainer run --userns --nv --writable-tmpfs --pwd /app \
  --bind "$REALM_ROOT":/app \
  --bind "$REALM_DATA":/data \
  --bind "$REALM_APPDATA":/cache \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT/tmp/$JOB":/tmp \
  "${OGLITE_BIND[@]}" \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONUNBUFFERED=1 \
  "$REALM_SIF" \
  python -u /app/scripts/clara/interactive/profile_phases.py --out "$OUT" --label "$LABEL" -- \
    --task_id "$TASK_ID" \
    --perturbation_id "$PERT_ID" \
    --repeats "$REPEATS" \
    --max_steps "$MAX_STEPS" \
    --horizon "$HORIZON" \
    --model_name debug \
    --model_type debug \
    --port 8000 \
    --robot "$ROBOT" \
    --experiment_name phase_ref \
    --run_id "${LABEL}_${JOB}" \
    --log_dir /logs \
    --rendering_mode rt \
    $ROD_FLAG
EXIT=$?

echo "[phase_ref] exited $EXIT"
echo "[phase_ref] json: $REALM_LOGS/phase_ref/${LABEL}_${JOB}.json"
[ "$EXIT" -eq 0 ] && rm -rf "$REALM_ROOT/tmp/$JOB"
exit $EXIT
