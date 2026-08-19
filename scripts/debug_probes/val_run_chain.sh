#!/bin/bash
# Validate the FIXED build against the criterion that actually matters: VISIBLE inward fingertip curl.
#
#   usage: ./scripts/debug_probes/val_run_chain.sh <jobid> [phase ...]
#          phases: vis  gate  deep      (default: all three, in that order)
#
# Everything runs MODE=stock -- the image's own OmniGibson, NO loader patch. `droid_robolab_xflat`
# is the asset-side fix, so a stock loader is the point: nothing but the .usd differs from the
# `DROID_robolab_v2` control, and the control is taken in the same session with the same flags.
#
# PHASE vis -- three Isaac starts, and the reason there are three rather than two:
#   1. val_xflat_a   fixed build, TWO identical rungs, --cam-freeze. Latches the camera and prints
#                    CAM_LATCH. The two identical rungs are the WITHIN-process noise floor.
#   2. val_xflat_b   the same build again in a SEPARATE process, camera forced to run 1's latch.
#                    This is the CROSS-process floor, and it is the floor the verdict is taken
#                    against, because the signal pair is itself cross-process.
#   3. val_v2_a      the control build, same forced camera, same flags. This is the SIGNAL.
#   Without run 2 a cross-process comparison would be judged against a within-process floor, which
#   flatters it. That is the whole reason the run exists.
#
# PHASE gate -- the grasp gate, which has never completed for any configuration. Tasks 0 and 4
#   through tests/test_vector_integrity.py at --num_envs 2, on the fixed build and on the control.
#   `is_grasping` intersects contacts with finger_link_names = [left_inner_finger,
#   right_inner_finger], so a changed pad response feeds grasp detection directly.
#
# PHASE deep -- the LOAD ladder, which answers "if it is not visible, what would make it so".
#   Overtravel past first contact is the honest knob: it is a property of the SCENE (how hard and
#   how far the object is driven into the tip), not of the robot. Lowering naturalFrequency below
#   its authored 1000 would be compensating for a bug that is now fixed, and is not done anywhere
#   in this chain -- every run here is at the authored nf=1000/dr=0.05.
#   --still-every 20 dumps a raw frame every 10 mm of overtravel, so one press per build covers the
#   whole ladder at depths that match between builds. val_xflat_deep2 is the cross-process floor at
#   every depth.
#
# Isaac exits 139 at teardown regardless of outcome -- grep the verdict lines, NEVER the exit code.
set -uo pipefail
JOB=${1:?usage: val_run_chain.sh <jobid> [vis|gate|deep ...]}
shift
PHASES=("$@"); [ ${#PHASES[@]} -eq 0 ] && PHASES=(vis gate deep)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L=/mnt/home_lustre/sedlam56/projects/REALM/logs/gripper_squeeze
RR="$HERE/scripts/clara/interactive/rr"
CHAIN="$L/val_chain.log"

# The measurement configuration, held FIXED across every run in phase vis so that the only variable
# between xflat and v2 is the .usd. Identical to the flags that produced the +0.335/+0.383 deg
# result, plus the three new rendering flags.
COMMON=(--load tip --states open --tip-fingers both
        --hold-steps 3 --traverse-steps 0 --rest-steps 8 --retract-steps 8
        --tip-gap 0.020 --tip-dz 0.0005 --tip-past 40 --tip-steps 80
        --video 1 --raw-stills 1 --cam-dist 0.13)

say() { echo "=== $(date -Is) $* ===" | tee -a "$CHAIN"; }

run() {  # run <logname> <args...>
  local name=$1; shift
  say "START $name"
  srun --jobid="$JOB" --overlap -n1 "$RR" python -u "$@" > "$L/$name.log" 2>&1
  say "DONE  $name (srun exit $?)"
}

# The camera latch, lifted out of run 1's log. Every later run is forced to this exact viewpoint, so
# the only thing that can differ between two images is the ROBOT. Each forced run also prints
# CAM_NATURAL and the delta, which is a free check that the two assets really do sit in the same
# place at rest.
latch() {
  local v
  v=$(grep -m1 '^  CAM_LATCH ' "$L/val_xflat_a.log" | sed 's/^  CAM_LATCH //' | tr -s ' ' ',')
  [ -n "$v" ] || { echo "NO CAM_LATCH in val_xflat_a.log -- cannot force the camera" | tee -a "$CHAIN"; return 1; }
  echo "$v"
}

for ph in "${PHASES[@]}"; do
case $ph in

vis)
  say "PHASE vis"
  run val_xflat_a /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_xflat --tag val_xflat_a --cam-freeze 1 \
      --rungs "nf1000a=1000/0.05,nf1000b=1000/0.05" "${COMMON[@]}"
  CP=$(latch) || exit 1
  say "CAM_POSE = $CP"
  run val_xflat_b /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_xflat --tag val_xflat_b --cam-pose="$CP" \
      --rungs "nf1000a=1000/0.05" "${COMMON[@]}"
  run val_v2_a /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_v2 --tag val_v2_a --cam-pose="$CP" \
      --rungs "nf1000a=1000/0.05,nf1000b=1000/0.05" "${COMMON[@]}"
  ;;

gate)
  say "PHASE gate"
  for rb in DROID_robolab_xflat DROID_robolab_v2; do
    run "val_gate_${rb}" /app/tests/test_vector_integrity.py \
        --cells 0:Default,4:Default --num_envs 2 --robot "$rb" \
        --experiment_name "ship_gate_${rb}" --log_dir /logs
  done
  ;;

deep)
  say "PHASE deep"
  CP=$(latch) || exit 1
  DEEP=(--load tip --states open --tip-fingers both
        --hold-steps 3 --traverse-steps 0 --rest-steps 8 --retract-steps 8
        --tip-gap 0.020 --tip-dz 0.0005 --tip-past 400 --tip-steps 500
        --video 1 --raw-stills 1 --still-every 20 --cam-dist 0.13 --cam-pose="$CP")
  run val_xflat_deep /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_xflat --tag val_xflat_deep --rungs "nf1000a=1000/0.05" "${DEEP[@]}"
  run val_v2_deep /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_v2 --tag val_v2_deep --rungs "nf1000a=1000/0.05" "${DEEP[@]}"
  run val_xflat_deep2 /app/scripts/debug_probes/curl_press_direction.py \
      --robot DROID_robolab_xflat --tag val_xflat_deep2 --rungs "nf1000a=1000/0.05" "${DEEP[@]}"
  ;;

*) echo "unknown phase '$ph' (want vis|gate|deep)" | tee -a "$CHAIN"; exit 2 ;;
esac
done

say "VAL_CHAIN_COMPLETE phases=${PHASES[*]}"
