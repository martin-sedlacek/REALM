#!/bin/bash
# Self-contained pi0.5 REALM eval: starts its OWN policy server on this node, runs the eval, tears
# the server down. One server per batch of evals, so several of these can run in parallel on
# different nodes without sharing a port or a GPU.
#
#   VEC=4 PERT_ID=0  MAX_STEPS=800 REPEATS=25 RUN_ID=def_vec4  sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
#   VEC=0 PERT_ID=13 MAX_STEPS=800 REPEATS=25 RUN_ID=vbpose_single sbatch scripts/clara/interactive/sbatch_eval_pi05.sh
#
# VEC=0 runs the single-env path (examples/02_evaluate.py); VEC=N>=1 runs the vectorized path
# (examples/04_vector_evaluate.py) in waves of N.
#
# PERTURBATION SAFETY: perturbations that call og.sim.stop()/play() are NOT safe vectorized, because
# those are global and REALM applies perturbations per member inside reset() -- one member's cycle
# disturbs every other member. Known offenders: VB-POSE (vb_pose.py:25,49), VB-MOBJ, VSB-NOBJ,
# SB-VRB. Run those with VEC=0 until the stop/play is batched the way apply_scene_fixes_from_cfg
# already is. Default is safe vectorized.
#
# WHY THIS SCRIPT DOES NOT TRUST THE EVAL'S EXIT CODE: Isaac's SimulationApp.close() hard-exits the
# process with status 0, so an unhandled Python exception still leaves $? at 0. Job 190683
# (2026-08-13) died on `AssertionError: droid_robolab_v2 is not a registered robot` after ~6 minutes,
# wrote no results at all, printed "[eval] exited 0", and Slurm recorded COMPLETED. Three runs were
# silently "successful" that day, one of which was read as a result. The authoritative signal is
# therefore the ARTIFACTS plus a scan of the run's own log -- see check_run.py, and the
# "eval gate" block at the bottom. Exit code is reported but never decides.
#
#SBATCH --job-name realm-pi05-eval
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 04:00:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/pi05_eval_%j.log

set -uo pipefail

# REALM_ROOT / REALM_SIF / REALM_DATA / REALM_APPDATA / REALM_LOGS / REALM_OGLITE_ROOT, all derived
# from this script's own location. $REALM_ROOT is what gets bound as /app AND where the gate's
# check_run.py is read from, so the two must not be allowed to drift apart -- deriving it from the
# script guarantees they cannot.
#
# This line used to read `REALM_ROOT=${REALM_ROOT:-<og391 default>}`, so that a worktree could point
# it at its own checkout. That override was a LIE: the shell profile EXPORTS REALM_ROOT (and
# REALM_SIF), both naming the pre-port 1.1.1 tree and image, so the default was never reached and an
# unset-looking variable silently selected the wrong stack. A worktree's copy of this script now
# derives the worktree, which is what the override was trying to buy in the first place. See
# scripts/clara/lib/paths.sh; override with REALM_SHARED_OG391= / REALM_SIF_OG391= if you must.
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
# OPENPI_ROOT, POLICY_CONFIG and CKPT are ONE unit -- a config name only resolves in the checkout
# that defines it, and only an *inference* config carries the DROID policy transforms. Serving a
# checkpoint under a training config yields a silent 0% run, not an error: see the retraction in
# ~/runbook/streams/realm_og391_port.md, 2026-08-17.
#
#   pi0.5    OPENPI_ROOT=.../openpi        POLICY_CONFIG=pi05_full_droid_finetune
#                                          CKPT=.../pi05_droid_jointpos
#   pi0-FAST OPENPI_ROOT=.../openpi_realm  POLICY_CONFIG=pi0_fast_droid_jointpos
#                                          CKPT=.../pi0_fast_droid_jointpos
#
# pi0_fast_droid_jointpos is defined ONLY in openpi_realm (config.py:643) and is what
# REALM/scripts/eval.sh:585 served for the original 1.1.1 benchmarks. It carries DroidInputs /
# DroidOutputs, prompt_from_task=True (without which the FAST model never emits the "Action: "
# marker its detokenizer needs) and AbsoluteActions(make_bool_mask(7, -1)).
# pi0_fast_full_droid_finetune in .../openpi is a FINE-TUNING recipe -- no policy transforms, an
# unfilled rlds_data_dir -- and must not be used to serve.
OPENPI_ROOT=${OPENPI_ROOT:-/mnt/home_lustre/sedlam56/projects/openpi}
CKPT=${CKPT:-/home/sedlam56/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_full_droid_finetune}

# REALM_LIGHT_FIX=1 restores OG 1.1.1's light configuration (FORCE_LIGHT_INTENSITY 150000 and no
# inputs:normalize write). ON BY DEFAULT FOR EVALS as of 2026-08-18: Martin reviewed the rendered
# task-3 comparison and picked the flag-on look ("the one called on_baseline looks good lets use
# that"). Pass REALM_LIGHT_FIX=0 to reproduce a run made before that call -- that path is
# bit-identical to stock OG 3.9.1 lighting, verified at 184.510 vs an unpatched container's 184.508.
# omnigibson/macros.py now defaults it on too, so this line only pins what an eval gets regardless of
# which OG-lite revision is bound.
#
# It exists because 3.9.1's two lighting changes (intensity /15, emission x1/area via
# normalize=True) cancel only at light area 1/15 m^2, which leaves a PER-SCENE error rather than a
# global one. Measured over the 6 comparable tasks on exterior cam1 (job 192356, ten tasks off and
# on): as shipped the per-task brightness ratio against the 1.1.1 references spans x1.013-x1.564
# (spread 0.551), and with the flag on it tightens to x1.130-x1.331 (spread 0.201) -- 2.74x tighter,
# mean x1.176 -> x1.201. It does NOT match 1.1.1 -- a uniform ~13-33% excess remains -- it makes the
# residual uniform, which is what a single exposure term could absorb and a per-scene shift never
# could. Do not pair it with an appearance-tuned intensity scale; that undoes exactly that property.
#
# NUMBERS CORRECTED TWICE. (1) 2026-08-18: the original 0.53 -> 0.11 from x1.199-x1.313 was three
# tasks wide and measured off ladder rungs carrying a ~+10 luma under-settle bias. (2) 2026-08-19: the
# 7-task replacement, 2.2x, included open_drawer -- whose cabinet was purpose="guide" and reached no
# camera until 8598e59, 90 min after the table was written -- so the comparable set is SIX and the
# figure is 2.74x. Both retractions in full at the FORCE_LIGHT_INTENSITY definition in
# omnigibson/macros.py; the table is ~/projects/REALM/logs/lightfix_10task/lightfix_table.{txt,json}.
#
# This script always binds OG-lite over the image's package (see --bind below), so the flag is live
# here in both states. Every run prints one "[REALM_LIGHT_FIX] ..." line at startup, and the value is
# echoed in the config block below, so no result is ever ambiguous about which lighting produced it.
# CHANGING IT INVALIDATES COMPARISONS against runs made with the other setting.
REALM_LIGHT_FIX=${REALM_LIGHT_FIX:-1}

VEC=${VEC:-4}
PERT_ID=${PERT_ID:-0}
TASK_ID=${TASK_ID:-0}
REPEATS=${REPEATS:-25}
MAX_STEPS=${MAX_STEPS:-800}
HORIZON=${HORIZON:-8}
ROBOT=${ROBOT:-DROID_robolab_v2}
MODEL_NAME=${MODEL_NAME:-checkpoints_pi05_droid_jointpos}
EXPERIMENT=${EXPERIMENT:-vec_pi05_verify}
JOB=${SLURM_JOB_ID:-local}
RUN_ID=${RUN_ID:-run_$JOB}
# Own port per job so two of these never collide even if Slurm co-locates them.
PORT=${PORT:-$((8000 + (JOB % 1000)))}
SERVER_WAIT=${SERVER_WAIT:-300}

[ -f "$REALM_SIF" ]                     || { echo "ERROR: no SIF" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ] || { echo "ERROR: no dataset" >&2; exit 1; }
[ -d "$CKPT/params" ]                   || { echo "ERROR: no params/ under $CKPT" >&2; exit 1; }
[ -f "$OPENPI_ROOT/scripts/serve_policy.py" ] || { echo "ERROR: no serve_policy.py under OPENPI_ROOT=$OPENPI_ROOT" >&2; exit 1; }
# Fail here rather than serve a checkpoint under a config that does not exist in this checkout --
# tyro would error, but the pairing is easy to get wrong and cheap to assert.
grep -q "name=\"$POLICY_CONFIG\"" "$OPENPI_ROOT/src/openpi/training/config.py" \
  || { echo "ERROR: POLICY_CONFIG=$POLICY_CONFIG is not defined in $OPENPI_ROOT/src/openpi/training/config.py" >&2; exit 1; }
# OGLITE_BIND=0 runs against an image that has OG-lite BAKED IN (built from .docker/realm_og391.def
# since 2026-08-20), so no bind is needed and /behavior-src is the image's own installed fork.
#
# Guarded, not just permitted: turning the bind off against an OLD image would silently run stock,
# unpatched OmniGibson with stock lighting -- the drawer tasks would not even load and every number
# would be wrong without anything failing. The provenance file only exists in images built from the
# new recipe, so it is the discriminator. Same check rr makes, for the same reason.
OGLITE_BIND=${OGLITE_BIND:-1}
if [ "$OGLITE_BIND" = 0 ]; then
  apptainer exec --userns "$REALM_SIF" test -f /behavior-src/OmniGibson/OGLITE_PROVENANCE 2>/dev/null \
    || { echo "ERROR: OGLITE_BIND=0 but $(basename "$REALM_SIF") has no baked-in OG-lite" >&2
         echo "       (no /behavior-src/OmniGibson/OGLITE_PROVENANCE -- it is an old patch-based image)." >&2
         echo "       Either drop OGLITE_BIND=0, or point REALM_SIF_OG391 at an image built from" >&2
         echo "       .docker/realm_og391.def after running scripts/stage_oglite_for_build.sh." >&2
         exit 1; }
  OGLITE_BIND_ARGS=()
  echo "OG-lite: BAKED INTO THE IMAGE, no bind ($(basename "$REALM_SIF"))"
else
  [ -d "$REALM_OGLITE_ROOT/omnigibson" ]  || { echo "ERROR: no OG-lite" >&2; exit 1; }
  OGLITE_BIND_ARGS=(--bind "$REALM_OGLITE_ROOT/omnigibson":/behavior-src/OmniGibson/omnigibson)
  echo "OG-lite: BOUND from $REALM_OGLITE_ROOT"
fi
mkdir -p "$REALM_ROOT/tmp/$JOB" "$REALM_APPDATA/appdata" "$REALM_LOGS/$EXPERIMENT"

echo "=================================================================="
echo " pi0.5 eval  vec=$VEC  pert_id=$PERT_ID  task=$TASK_ID"
echo " repeats=$REPEATS  max_steps=$MAX_STEPS  horizon=$HORIZON  robot=$ROBOT"
echo " policy_config=$POLICY_CONFIG  openpi_root=$OPENPI_ROOT"
echo " ckpt=$CKPT"
echo " run_id=$RUN_ID  port=$PORT  node=$(hostname)"
echo " light_fix=$REALM_LIGHT_FIX  ($([ "$REALM_LIGHT_FIX" != 0 ] && echo 'OG 1.1.1 lighting: intensity 150000, no normalize' || echo 'off -- stock OG 3.9.1 lighting'))"
echo "=================================================================="

#--- own policy server -----------------------------------------------------------------------------
SERVER_LOG="$REALM_LOGS/pi05_server_${JOB}.log"
(
  cd "$OPENPI_ROOT" || exit 1
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  export HF_HUB_OFFLINE=1
  exec uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
      --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval] server pid=$SERVER_PID port=$PORT log=$SERVER_LOG"
trap 'kill $SERVER_PID 2>/dev/null' EXIT

SERVER_UP=0
for i in $(seq 1 "$SERVER_WAIT"); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[eval] server died:" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
  python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null && { echo "[eval] server up after ${i}s"; SERVER_UP=1; break; }
  sleep 1
done
# Falling out of the loop without a connection used to run the eval anyway against a dead port. The
# eval then fails ~6 minutes later inside the container, which the gate at the bottom now catches --
# but there is no reason to spend those minutes, or to report the failure as an eval bug.
[ "$SERVER_UP" -eq 1 ] || {
  echo "[eval] FATAL: policy server did not accept connections on port $PORT within ${SERVER_WAIT}s" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
}

#--- eval, inside the container with the OG-lite bind ----------------------------------------------
if [ "$VEC" -ge 1 ]; then
  ENTRY=(python -u /app/examples/04_vector_evaluate.py --num_envs "$VEC")
else
  ENTRY=(python -u /app/examples/02_evaluate.py)
fi

RESULTS="$REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
EVAL_LOG="$REALM_LOGS/pi05_evalout_${JOB}_${RUN_ID}.log"
# Taken BEFORE the eval so the gate can tell this run's artifacts from ones an earlier run with the
# same RUN_ID left in $RESULTS. Without it, re-running a RUN_ID that previously succeeded makes any
# crash look like a pass, because the directory is already full of a valid-looking eval.
START_EPOCH=$(date +%s)

apptainer run --userns --nv --writable-tmpfs --pwd /app \
  --bind "$REALM_ROOT":/app \
  --bind "$REALM_DATA":/data \
  --bind "$REALM_APPDATA":/cache \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT/tmp/$JOB":/tmp \
  "${OGLITE_BIND_ARGS[@]}" \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONUNBUFFERED=1 \
  --env REALM_LIGHT_FIX="$REALM_LIGHT_FIX" \
  "$REALM_SIF" \
  "${ENTRY[@]}" \
    --task_id "$TASK_ID" --perturbation_id "$PERT_ID" \
    --repeats "$REPEATS" --max_steps "$MAX_STEPS" --horizon "$HORIZON" \
    --model_type openpi --model_name "$MODEL_NAME" \
    --port "$PORT" --host 127.0.0.1 \
    --experiment_name "$EXPERIMENT" --run_id "$RUN_ID" --log_dir /logs \
    --robot "$ROBOT" --rendering_mode rt \
  2>&1 | tee "$EVAL_LOG"
# ${PIPESTATUS[0]}, not $?: $? is the exit of `tee`. The eval's own output is teed to its own file so
# the gate below has an exact, self-contained log to scan even when this runs outside Slurm (JOB=local,
# no %j log) -- and so the scan cannot be confused by anything this wrapper printed.
EXIT=${PIPESTATUS[0]}

echo "[eval] exited $EXIT  (NOT a success signal -- see the gate below)"
printf '### EXIT_CODE=%s\n' "$EXIT" >> "$EVAL_LOG"
echo "[eval] artifacts: $RESULTS"
ls "$RESULTS" 2>/dev/null

#--- eval gate: did this run actually produce a complete set of results? ----------------------------
# The only evidence that counts. check_run.py requires all four artifacts, exactly $REPEATS rollout
# rows in reports/*.csv (so a run that died half way through is not reported as complete), artifact
# mtimes newer than this job's start (so a previous run's leftovers under the same RUN_ID cannot
# stand in for this one), and a log free of Traceback / AssertionError / Segmentation fault / CUDA
# OOM / 'row mismatch'.
python3 "$REALM_ROOT/scripts/clara/interactive/check_run.py" \
    "$RESULTS" "$EVAL_LOG" --repeats "$REPEATS" --newer-than "$START_EPOCH"
GATE=$?

if [ "$GATE" -ne 0 ]; then
  echo "==================================================================" >&2
  echo " EVAL FAILED -- $RUN_ID did NOT produce a complete, verified set of results" >&2
  echo "   job=$JOB  run_id=$RUN_ID  vec=$VEC  pert_id=$PERT_ID  task=$TASK_ID  repeats=$REPEATS" >&2
  echo "   reported exit code was $EXIT, which proves nothing: SimulationApp.close() exits 0" >&2
  echo "   even after an unhandled exception. The gate above is the real verdict." >&2
  echo "   eval log:  $EVAL_LOG" >&2
  echo "   artifacts: $RESULTS" >&2
  echo "   DO NOT report numbers from this run." >&2
  echo "   Keeping $REALM_ROOT/tmp/$JOB for debugging." >&2
  echo "==================================================================" >&2
  exit 1
fi

echo "=================================================================="
echo " EVAL OK -- $RUN_ID: $REPEATS rollouts, all artifacts present, clean log"
echo " artifacts: $RESULTS"
echo "=================================================================="
rm -rf "$REALM_ROOT/tmp/$JOB"
exit 0
