#!/bin/bash
# Smoke test for the OmniGibson 3.9.1 port: pi0.5 (openpi, pi05_droid_jointpos) on ONE task under the
# Default perturbation, a handful of repeats. Enough to prove the port boots, the DROID robot loads
# from its RobotDefinition, the policy drives it, videos record and SR is scored.
#
# What makes this different from scripts/clara/run_eval_single.sh (the pre-3.9.1 driver):
#   - a different image: realm_og391.sif, built from stanfordvl/behavior:3.9.1
#   - `apptainer run`, not `exec`: the image's %runscript activates the conda env "behavior".
#     `exec` bypasses it and lands on the base python with no omnigibson.
#   - one /data bind replaces the old dataset/asset/key paths, and /cache backs
#     OMNIGIBSON_APPDATA_PATH. Isaac Sim 5.1 is a pip package, so the old /isaac-sim binds are gone.
#   - no pip installs at runtime: openpi-client and everything else is baked into the image.
#   - no --og_lite: OG-lite does not exist for 3.9.1 and this branch has no such flag.
#
# Datasets and results are NOT duplicated into this repo -- see REALM_DATA / REALM_LOGS below. Both
# point into ~/projects/REALM, which is where the 36 GB og391 dataset was rsynced and where every
# REALM eval writes its reports.
#
# Usage (from the REALM_og391 root):
#   sbatch scripts/clara/run_og391_smoke_pi05.sh
#   TASK_ID=1 REPEATS=5 sbatch scripts/clara/run_og391_smoke_pi05.sh
#   MULTI_VIEW=1 sbatch scripts/clara/run_og391_smoke_pi05.sh   # second video panel for review only
#   RENDER_ON_DEMAND=0 sbatch scripts/clara/run_og391_smoke_pi05.sh  # render every step (full-rate video)
#
#SBATCH --job-name realm-og391-smoke-pi05
#SBATCH --partition l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G
#SBATCH --time 03:00:00
#SBATCH --output=/mnt/home_lustre/sedlam56/projects/REALM/logs/og391_smoke_pi05_%j.log

set -uo pipefail

REALM_ROOT=/mnt/home_lustre/sedlam56/projects/REALM_og391
OPENPI_ROOT=/mnt/home_lustre/sedlam56/projects/openpi

# Shared with ~/projects/REALM on purpose: the 36 GB behavior-1k-assets + robot assets live there and
# must not be copied, and results belong in the same log tree as every other REALM eval.
REALM_DATA=$REALM_ROOT/data/datasets          # -> REALM/data/datasets_og391
REALM_LOGS=/mnt/home_lustre/sedlam56/projects/REALM/logs
REALM_SIF=$REALM_ROOT/realm_og391.sif         # -> REALM/realm_og391.sif
APPDATA=$REALM_ROOT/data/cache                # OMNIGIBSON_APPDATA_PATH=/cache/appdata

TASK_ID=${TASK_ID:-0}                 # 0 = put_green_block_into_bowl
PERT_ID=${PERT_ID:-0}                 # 0 = Default (no perturbation)
# Selects realm/config/robots/$ROBOT.yaml. "DROID" is the stock droid_mounted asset; "DROID_robolab"
# is the robolab Franka + Robotiq 2F-85 with the compliant mimic-joint gripper. A non-stock robot
# also needs its definition registered in the dataset -- 3.9.1 globs <data>/*/models/<name>/<name>.yaml
# -- which is a symlink NOT tracked in git:
#   ln -s /app/realm/robots/definitions/<name> data/datasets/omnigibson-robot-assets/models/<name>
ROBOT=${ROBOT:-DROID}
REPEATS=${REPEATS:-3}
MAX_STEPS=${MAX_STEPS:-300}
HORIZON=${HORIZON:-8}                 # pi0.5 action-chunk execution horizon
PORT=${PORT:-8500}
EXPERIMENT=${EXPERIMENT:-og391_smoke_pi05}
MODEL_NAME=${MODEL_NAME:-checkpoints_pi05_droid_jointpos}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_full_droid_finetune}
RENDERING_MODE=${RENDERING_MODE:-rt}
# pi0.5 reads ONE exterior camera. realm/inference/client.py's openpi branch sends only
# observation/exterior_image_1_left (= base_im = external_sensor0) plus the wrist image; the second
# exterior view is never passed. The one path that would use it, use_base_im_second in eval.py:228,
# tests `task_type == "open_close_drawer"`, a value no task config declares (the drawer tasks say
# "open_drawer" / "close_drawer"), so it never fires for any REALM_DROID10 task.
# So --multi-view only renders a second 1280x720 camera per step and adds a video panel. Off by
# default; set MULTI_VIEW=1 when you want the second panel for reviewing footage. Do NOT copy this
# default into a cosmos3 / DreamZero / GR00T launcher -- those DO consume the second view.
MULTI_VIEW=${MULTI_VIEW:-0}
MULTI_VIEW_FLAG=""
[ "$MULTI_VIEW" = "1" ] && MULTI_VIEW_FLAG="--multi-view"
# Render only on the steps whose observation feeds inference (1 in HORIZON); physics-only on the
# rest, via OG 3.9.1's native og.sim.render_on_step().
# ON by default. Microbenchmark (tmp/dbg_session/bench_step2.py, interleaved medians, 1 exterior +
# 1 wrist camera at 1280x720, rt):
#     env.step render ON  + obs      348.8 ms
#     env.step render OFF + obs      242.3 ms   -> render costs ~106 ms, ~30% of a step
#     get_obs() alone                  5.4 ms   -> the obs/annotator read is NOT the cost
#     blind step + extract_from_obs   263.3 ms
#   => 7-of-8 blind predicts 348.8 -> 274 ms/step, about -21%.
# NOTE: an earlier end-to-end A/B (jobs 187497 vs 187532, 422 vs 419 ms/step) appeared to show no
# gain and this defaulted off. That comparison was UNDERPOWERED, not a null: per-rollout ms/step
# scatters +/-90 ms run to run, against a ~75 ms expected effect. Do not re-flip this off on the
# strength of a handful of rollouts -- instrument per-step wall time inside the loop instead.
RENDER_ON_DEMAND=${RENDER_ON_DEMAND:-1}
ROD_FLAG=""
[ "$RENDER_ON_DEMAND" = "1" ] && ROD_FLAG="--render_on_demand"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
# Local cache -- compute nodes have NO outbound internet, so never point at gs:// here.
CKPT=${CKPT:-/home/sedlam56/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos}
SERVER_WAIT=${SERVER_WAIT:-240}

JOB=${SLURM_JOB_ID:-local}

#--- Fail fast on anything missing, before booting a simulator ------------------------------------
[ -f "$REALM_SIF" ]            || { echo "ERROR: no SIF at $REALM_SIF" >&2; exit 1; }
[ -d "$REALM_DATA" ]           || { echo "ERROR: no dataset at $REALM_DATA" >&2; exit 1; }
[ -d "$REALM_DATA/behavior-1k-assets" ] || { echo "ERROR: $REALM_DATA has no behavior-1k-assets/" >&2; exit 1; }
[ -d "$REALM_LOGS" ]           || { echo "ERROR: no log dir at $REALM_LOGS" >&2; exit 1; }
[ -d "$CKPT/params" ]          || { echo "ERROR: no params/ under $CKPT" >&2; exit 1; }
# /tmp is node-local (200 G, wiped when the job ends). Every artifact path must be on Lustre.
case "$REALM_ROOT$REALM_LOGS" in /tmp/*) echo "ERROR: refusing to write artifacts under /tmp" >&2; exit 1;; esac

mkdir -p "$REALM_ROOT/tmp/$JOB" "$APPDATA/appdata" "$REALM_LOGS/$EXPERIMENT"

echo "=================================================================="
echo " REALM og391 smoke test -- pi0.5"
echo " robot=$ROBOT  task_id=$TASK_ID  perturbation=$PERT_ID (Default)  repeats=$REPEATS"
echo " max_steps=$MAX_STEPS  horizon=$HORIZON  rendering=$RENDERING_MODE  port=$PORT"
echo " sif        = $(readlink -f "$REALM_SIF")"
echo " dataset    = $(readlink -f "$REALM_DATA")"
echo " results    = $REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
echo " checkpoint = $CKPT"
echo "=================================================================="

#--- 1. policy server (shares the single L40S with the sim) ---------------------------------------
SERVER_LOG="$REALM_LOGS/og391_pi05_server_${JOB}.log"
(
  cd "$OPENPI_ROOT" || exit 1
  # pi0.5 (~3B) + the OmniGibson sim share ONE 48 GB L40S; this is what the pre-3.9.1 pi0 scripts do.
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
  exec uv run scripts/serve_policy.py \
      --port="$PORT" \
      policy:checkpoint \
      --policy.config="$POLICY_CONFIG" \
      --policy.dir="$CKPT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[og391] pi05 server pid=$SERVER_PID on GPU 0 (log: $SERVER_LOG)"

for i in $(seq 1 "$SERVER_WAIT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[og391] ERROR: policy server died during startup. Last lines:" >&2
    tail -30 "$SERVER_LOG" >&2; exit 1
  fi
  python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)
" 2>/dev/null && { echo "[og391] server up after ${i}s"; break; }
  sleep 1
done

#--- 2. REALM eval client, inside the 3.9.1 container ---------------------------------------------
# --log_dir /logs rather than letting it default to /app/logs: REALM_og391/logs is a symlink to
# REALM/logs, and a symlink to an unbound host path dangles inside the container.
cd "$REALM_ROOT" || exit 1

apptainer run --userns --nv --writable-tmpfs \
  --bind "$REALM_ROOT":/app \
  --bind "$REALM_DATA":/data \
  --bind "$APPDATA":/cache \
  --bind "$REALM_LOGS":/logs \
  --bind "$REALM_ROOT/tmp/$JOB":/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env CUDA_VISIBLE_DEVICES=0 \
  "$REALM_SIF" \
  python examples/02_evaluate.py \
    --task_id "$TASK_ID" \
    --perturbation_id "$PERT_ID" \
    --repeats "$REPEATS" \
    --max_steps "$MAX_STEPS" \
    --horizon "$HORIZON" \
    --model_type openpi \
    --robot "$ROBOT" \
    --model_name "$MODEL_NAME" \
    --port "$PORT" --host 127.0.0.1 \
    --experiment_name "$EXPERIMENT" \
    --run_id "$RUN_ID" \
    --log_dir /logs \
    $MULTI_VIEW_FLAG \
    $ROD_FLAG \
    --rendering_mode "$RENDERING_MODE"
EXIT=$?

echo "[og391] eval client exited $EXIT"
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null

RESULTS="$REALM_LOGS/$EXPERIMENT/$MODEL_NAME/$RUN_ID"
echo "[og391] artifacts under $RESULTS"
ls -R "$RESULTS" 2>/dev/null | head -40

# Temp dirs are kept on failure so the omniverse logs under them stay inspectable.
[ "$EXIT" -eq 0 ] && rm -rf "$REALM_ROOT/tmp/$JOB"
exit $EXIT
