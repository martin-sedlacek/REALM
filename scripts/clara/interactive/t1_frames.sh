#!/bin/bash
# Exact reproduction of the run that produced docs/vector_env/frames/, so the new montages are
# directly comparable to the committed ones.
set -uo pipefail
cd /mnt/home_lustre/sedlam56/projects/REALM_og391
NUM_ENVS=${NUM_ENVS:-4}
OUT=${OUT:-/logs/vector_first_frames_190155}
echo "### t1 frames: num_envs=$NUM_ENVS out=$OUT"
MODE=${MODE:-stock} ./scripts/clara/interactive/rr \
  python -u examples/03_vector_first_frames.py \
    --num_envs "$NUM_ENVS" --task_id 0 --out_dir "$OUT"
echo "### frames exit: $?"
