#!/bin/bash
# Render the policy camera's first frame for both drawer tasks, at one or more --num_envs.
#
# The cabinet-facing bug is a rendering claim, so the arbiter has to be a rendered frame from the
# camera that actually feeds the policy (external_sensor0). This wraps 03_vector_first_frames.py so
# the before/after pair is produced by one command with identical arguments -- the only difference
# between the two runs being the asset under test.
#
#   TAG=before ./scripts/clara/interactive/t17_drawer_frames.sh
#   TAG=after  NUM_ENVS="1 4" ./scripts/clara/interactive/t17_drawer_frames.sh
set -uo pipefail

TAG=${TAG:-before}
NUM_ENVS=${NUM_ENVS:-1}
TASKS=${TASKS:-"8 9"}
ROBOT=${ROBOT:-DROID_robolab}
MODE=${MODE:-oglite}
export MODE

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1

for t in $TASKS; do
  for n in $NUM_ENVS; do
    out=/logs/drawer_asset/frames_${TAG}/t${t}_n${n}
    echo "=== TAG=$TAG task=$t num_envs=$n -> $out ==="
    ./scripts/clara/interactive/rr python -u examples/03_vector_first_frames.py \
      --num_envs "$n" --task_id "$t" --robot "$ROBOT" --out_dir "$out"
    echo "=== exit=$? task=$t num_envs=$n ==="
  done
done
