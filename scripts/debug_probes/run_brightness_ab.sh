#!/bin/bash
# Run a debug probe inside EITHER stack's container, on an already-held allocation.
#
#   STACK=og391 ./scripts/debug_probes/run_brightness_ab.sh --label og391_rt
#   STACK=og111 ./scripts/debug_probes/run_brightness_ab.sh --label og111_rt
#
# Everything after the script name is passed through to the probe.
#
# PROBE= selects which file in THIS directory runs, defaulting to the one this launcher was written
# for. Both container invocations mount the whole debug_probes dir, so any probe there works; the
# name was hardcoded only because there used to be one.
#
#   STACK=og391 PROBE=post_tone_sweep.py ./scripts/debug_probes/run_brightness_ab.sh --label x
#
# og391 goes through scripts/clara/interactive/rr, so it inherits that path resolution unchanged.
#
# og111 needs its own invocation, modelled on scripts/clara/interactive/sbatch_phase_ref_og111.sh:
# a different image, a different OmniGibson mount point (/omnigibson-src, not /behavior-src), and
# `micromamba run -n omnigibson`. THE 1.1.1 TREE IS READ-ONLY here -- unlike that script, this one
# does NOT copy the probe into it. The probe is bound in at /dbg from THIS checkout, and /app is
# bound :ro so a stray write fails loudly instead of mutating the pre-port tree. Artifacts go to the
# SHARED log store, which both stacks already write to.

set -uo pipefail

STACK=${STACK:-og391}
PROBE=${PROBE:-render_brightness_ab.py}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HERE/$PROBE" ] || { echo "ERROR: no probe at $HERE/$PROBE" >&2; exit 1; }
source "$HERE/../clara/lib/paths.sh"
[ "${REALM_PATHS_SH:-}" = 1 ] || { echo "ERROR: could not source scripts/clara/lib/paths.sh" >&2; exit 1; }

OUT_HOST=$REALM_SHARED/logs/render_bright_ab
mkdir -p "$OUT_HOST"

case "$STACK" in
og391)
  echo "=== stack og391 :: $REALM_SIF ==="
  exec "$HERE/../clara/interactive/rr" python -u "/app/scripts/debug_probes/$PROBE" \
    --out /logs/render_bright_ab "$@"
  ;;

og111)
  # Deliberately NOT from paths.sh -- this is the one place that must point at the pre-port stack.
  OG111_ROOT=${REALM_OG111_ROOT:-/mnt/home_lustre/sedlam56/projects/REALM}
  OG111_OGLITE=/mnt/home_lustre/sedlam56/projects/OG-lite
  OG111_SIF=${REALM_OG111_SIF:-/mnt/home_lustre/sedlam56/apptainer/realm-dm.sif}

  # The same guard sbatch_phase_ref_og111.sh carries, and for the same reason: ~/projects/REALM is
  # the 1.1.1 checkout today AND the destination of the pending REALM_og391 -> REALM rename. After
  # that rename this path silently becomes the PORTED tree and the job would label 3.9.1 numbers
  # "og111". Two markers separate them.
  [ -f "$OG111_ROOT/realm/misc/modified_entity_prim.py" ] || {
    echo "ERROR: $OG111_ROOT is not the OmniGibson 1.1.1 checkout. Set REALM_OG111_ROOT." >&2; exit 1; }
  if compgen -G "$OG111_ROOT/realm/misc/*_og391.patch" >/dev/null; then
    echo "ERROR: $OG111_ROOT is the PORTED checkout, not 1.1.1. Set REALM_OG111_ROOT." >&2; exit 1
  fi
  [ -f "$OG111_SIF" ]                 || { echo "ERROR: no SIF at $OG111_SIF" >&2; exit 1; }
  [ -d "$OG111_ROOT/data/datasets" ]  || { echo "ERROR: no 1.1.1 dataset" >&2; exit 1; }
  [ -d "$OG111_OGLITE/omnigibson" ]   || { echo "ERROR: no 1.1.1 OG-lite at $OG111_OGLITE" >&2; exit 1; }

  # The OptiX denoiser data blob. scripts/run_pt_vs_rt_bench.sh binds it for the docker path;
  # sbatch_phase_ref_og111.sh does not, and would never have noticed because it only measures
  # timings. Bound when the node has it, skipped when it does not.
  NVOPTIX=$([ -f /usr/share/nvidia/nvoptix.bin ] && echo /usr/share/nvidia/nvoptix.bin || echo "")

  # /tmp and every cache live under THIS checkout, so the 1.1.1 tree is never written to.
  JOB=${SLURM_JOB_ID:-interactive}
  TMPROOT=$REALM_ROOT/tmp/og111_bright_$JOB
  mkdir -p "$TMPROOT" "$REALM_ROOT/mamba_cache/$JOB" "$REALM_ROOT/pip_cache/$JOB"

  D=$OG111_ROOT/data
  echo "=== stack og111 :: $OG111_SIF  (tree $OG111_ROOT, read-only) ==="
  exec apptainer exec --userns --nv --writable-tmpfs --pwd /app \
    --bind "$OG111_ROOT":/app:ro \
    --bind "$OG111_OGLITE":/omnigibson-src:ro \
    --bind "$HERE":/dbg:ro \
    --bind "$D"/datasets:/data \
    --bind "$D"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
    --bind "$D"/isaac-sim/cache/ov:/root/.cache/ov \
    --bind "$D"/isaac-sim/cache/pip:/root/.cache/pip \
    --bind "$D"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
    --bind "$D"/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
    --bind "$D"/isaac-sim/logs:/root/.nvidia-omniverse/logs \
    --bind "$D"/isaac-sim/config:/root/.nvidia-omniverse/config \
    --bind "$D"/isaac-sim/data:/root/.local/share/ov/data \
    --bind "$OUT_HOST":/logs/render_bright_ab \
    --bind "$TMPROOT":/tmp \
    ${NVOPTIX:+--bind $NVOPTIX:/usr/share/nvidia/nvoptix.bin:ro} \
    --env TMPDIR=/tmp \
    --env OMNIGIBSON_HEADLESS=1 \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PYTHONUNBUFFERED=1 \
    --env MAMBA_CACHE_DIR="$REALM_ROOT/mamba_cache/$JOB" \
    --env PIP_CACHE_DIR="$REALM_ROOT/pip_cache/$JOB" \
    "$OG111_SIF" \
    micromamba run -n omnigibson python -u "/dbg/$PROBE" \
      --out /logs/render_bright_ab "$@"
  ;;

*) echo "ERROR: STACK must be og391 or og111 (got '$STACK')" >&2; exit 1 ;;
esac
