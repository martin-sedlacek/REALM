#!/bin/bash
# Produce the ONE patched OmniGibson file the vectorized eval path needs on top of stock, so
# `MODE=stockfix ./scripts/clara/interactive/rr ...` can run without binding the whole OG-lite fork.
#
#   ./scripts/clara/interactive/make_stock_patch.sh            # needs a Slurm alloc, uses $ALLOC
#   ALLOC=190763 ./scripts/clara/interactive/make_stock_patch.sh
#
# WHY THIS EXISTS. OG-lite differs from the image's own OmniGibson 3.9.1 in 7 files / ~570 lines,
# but only 25 of those lines are needed for CORRECTNESS: scenes/scene_base.py re-applies object
# poses after the scene prim reaches its final position. Without it, scene-file objects load ~100 m
# too high in every scene except index 0 -- scene 0's origin IS the world origin, which is why no
# single-env run ever showed it. Measured on this image, num_envs=2, Default and V-SC:
#
#     scene 0: every check passes
#     scene 1: "main object left the table (z=0.015)", a distractor displaced 3.51 m
#
# and that was the ONLY failure in either run: no crash, no contact-view damage, and the sibling
# init-queue eviction was repaired by REALM's own code, since that bug is upstream OmniGibson's
# (Simulator._pre_remove_object matches by name) and not something the fork introduced.
#
# Everything else in OG-lite -- incremental contact cache, proximity gate, descriptor-pool raise,
# render-on-demand plumbing -- is performance at these scene counts. Visible here as 268 contact
# rows on stock versus 49-51 with the fork.
#
# The patch is also wired into .docker/realm_og391.def with a build-time grep guard, so a rebuilt
# SIF ships it and this script plus MODE=stockfix become unnecessary. This exists to test the change
# BEFORE anyone pays for a rebuild, and because `apptainer build --fakeroot` fails on Lustre here
# ("failed to change uid and gids on /image/rootfs/var/mail"), so the rebuild has to happen
# elsewhere.
set -uo pipefail

REALM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# NOT ${REALM_SIF:-...}. The shell profile on this machine exports
# REALM_SIF=/home/sedlam56/apptainer/realm-dm.sif -- the PRE-PORT 1.1.1 image, where OmniGibson
# lives at /omnigibson-src rather than /behavior-src. Defaulting through that variable silently
# selected the wrong container and the failure looked like "the patch no longer applies" rather
# than "wrong image". Same trap as $REALM_ROOT, which points at the pre-port checkout. Override
# with REALM_SIF_OG391= if you really need a different image.
REALM_SIF=${REALM_SIF_OG391:-/mnt/home_lustre/sedlam56/projects/REALM/realm_og391.sif}
OUT_DIR=${OUT_DIR:-/mnt/home_lustre/sedlam56/projects/REALM/stock_patch}
PATCH=$REALM_ROOT/realm/misc/scene_base_zoffset_og391.patch
ALLOC=${ALLOC:-${SLURM_JOB_ID:-}}

[ -f "$PATCH" ]     || { echo "no patch at $PATCH" >&2; exit 1; }
[ -f "$REALM_SIF" ] || { echo "no SIF at $REALM_SIF" >&2; exit 1; }
[ -n "$ALLOC" ]     || { echo "set ALLOC to a running interactive job id (apptainer needs a node)" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# Patch a COPY of the image's own file, inside the image, so the result matches the stock version
# this SIF actually ships. Patching a host copy of some other checkout's file would silently drift.
#
# `exec`, not `run`: the %runscript sources the container's docker2singularity env script, which
# fails to parse under srun here ("reached EOF without closing quote") and leaves /behavior-src
# unreachable. exec skips it. Nothing below needs the conda env -- only cp and patch.
#
# ONE LINE on purpose: a multi-line `bash -c '...'` through srun gets its newlines collapsed.
# Documented in rr's header.
srun --jobid="$ALLOC" --overlap -n1 apptainer exec --userns --pwd /app --bind "$REALM_ROOT":/app --bind "$OUT_DIR":/out "$REALM_SIF" bash -c 'set -e; mkdir -p /tmp/w/omnigibson/scenes && cp /behavior-src/OmniGibson/omnigibson/scenes/scene_base.py /tmp/w/omnigibson/scenes/ && cd /tmp/w && patch -p1 < /app/realm/misc/scene_base_zoffset_og391.patch && cp /tmp/w/omnigibson/scenes/scene_base.py /out/scene_base.py' || { echo "patching failed -- wrong image, or the patch no longer applies. SIF=$REALM_SIF" >&2; exit 1; }

grep -q "Re-apply the object poses now that the scene prim is at its final position" \
  "$OUT_DIR/scene_base.py" || { echo "marker missing from $OUT_DIR/scene_base.py" >&2; exit 1; }

echo "wrote $OUT_DIR/scene_base.py"
echo "use with:  MODE=stockfix ./scripts/clara/interactive/rr <cmd>"
