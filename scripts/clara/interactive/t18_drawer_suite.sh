#!/bin/bash
# The full drawer re-measurement: facing, joint travel, openness reset, and rendered frames,
# on BOTH drawer tasks at num_envs 1 and 4.
#
# One script because the acceptance bar for the base_link-frame fix is a set of numbers that have to
# be produced together and compared against the pre-fix values -- a facing fix that silently changed
# which world axis the drawer slides along, or broke the openness reset, would be a worse bug than
# the one it replaced. Run it before and after the asset edit with identical arguments.
#
#   TAG=after ./scripts/clara/interactive/t18_drawer_suite.sh
set -uo pipefail

TAG=${TAG:-after}
TASKS=${TASKS:-"8 9"}
ENVS=${ENVS:-"1 4"}
ROBOT=${ROBOT:-DROID_robolab}
HOLD=${HOLD:-200}
MODE=${MODE:-oglite}
export MODE

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
RR=./scripts/clara/interactive/rr

# 1. Facing: is the drawer front pointing at external_sensor0, the camera that feeds the policy?
for t in $TASKS; do
  echo "########## [$TAG] t16 facing: task $t ##########"
  $RR python -u scripts/clara/interactive/t16_drawer_facing.py --num_envs 1 --task_id "$t" --robot "$ROBOT"
  echo "########## [$TAG] t16 task $t exit=$? ##########"
done

# 2. Joint: travel distance and axis, openness reset, stability over $HOLD free steps.
for t in $TASKS; do
  for n in $ENVS; do
    echo "########## [$TAG] t13 joint: task $t num_envs $n ##########"
    $RR python -u scripts/clara/interactive/t13_drawer_stop.py \
      --num_envs "$n" --task_id "$t" --robot "$ROBOT" --hold_steps "$HOLD" --resets 1
    echo "########## [$TAG] t13 task $t n$n exit=$? ##########"
  done
done

# 3. Frames from the policy camera, the visual arbiter.
for t in $TASKS; do
  echo "########## [$TAG] frames: task $t ##########"
  $RR python -u examples/03_vector_first_frames.py \
    --num_envs 1 --task_id "$t" --robot "$ROBOT" --out_dir "/logs/drawer_asset/frames_${TAG}/t${t}_n1"
  echo "########## [$TAG] frames task $t exit=$? ##########"
done

echo "########## [$TAG] SUITE COMPLETE ##########"
