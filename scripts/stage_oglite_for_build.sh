#!/bin/bash
# Stage the OG-lite fork's `omnigibson` package into the build context, so an image built from
# .docker/realm.{def,Dockerfile} contains it and needs NO runtime bind.
#
# WHY A STAGING STEP AT ALL. Both recipes take the REALM repo root as their build context and the fork
# is a SEPARATE repository living outside it (github.com/martin-sedlacek/REALM_OG_lite, checked out at
# ../OG-lite_og391 on Clara). `docker build` cannot COPY from outside its context, and hardcoding a host
# path into the .def would make the recipe work on exactly one machine. So the fork is copied into
# .docker/vendor/ (gitignored) first, and both recipes read it from there -- one mechanism for both.
#
#     ./scripts/stage_oglite_for_build.sh                     # uses ../OG-lite_og391
#     OG_LITE_ROOT=/path/to/OG-lite_og391 ./scripts/stage_oglite_for_build.sh
#
# WHAT GOES IN THE IMAGE, AND WHY WHOLESALE RATHER THAN MORE PATCHES.
# The fork is a FULL OmniGibson checkout that is a strict SUPERSET of the nine patches the recipes used
# to apply: measured 2026-08-20, all nine grep guards are present in the fork, and it additionally
# carries the performance work the patch set never covered (incremental contact cache, proximity gate,
# descriptor-pool raise, render-on-demand plumbing -- envs/env_base.py, envs/vec_env_base.py,
# prims/rigid_prim.py, utils/usd_utils.py). Replacing the package wholesale therefore reproduces what
# `MODE=oglite` gives today BIT-FOR-BIT BY CONSTRUCTION, because that mode literally bind-mounts this
# same directory over the image's copy. A patch set could only ever approximate it, and this repo has
# already been bitten twice by a patch set drifting from the fork it was generated from.
#
# THE COST, stated so it can be reversed knowingly: the image no longer documents what changed relative
# to stock OmniGibson, and a future base image's upstream fixes to these files would be silently
# overwritten. Mitigations kept in place: the recipes still run all nine grep guards (so a stage that
# lost a semantic change fails the build), and the fork's exact commit is recorded in the image.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OG_LITE_ROOT=${OG_LITE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/OG-lite_og391}
VENDOR="$REPO_ROOT/.docker/vendor"

[ -d "$OG_LITE_ROOT/omnigibson" ] || {
  echo "ERROR: no omnigibson package at $OG_LITE_ROOT/omnigibson" >&2
  echo "       set OG_LITE_ROOT to the fork checkout." >&2; exit 1; }

echo "staging  $OG_LITE_ROOT/omnigibson"
echo "     ->  $VENDOR/omnigibson"
mkdir -p "$VENDOR"

# --delete so a file the fork DROPPED cannot survive in a stale vendor dir. __pycache__ excluded on
# purpose: stale .pyc in an image can shadow source that no longer matches it, which is a genuinely
# nasty failure to debug inside a container.
rsync -a --delete \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/' \
      "$OG_LITE_ROOT/omnigibson/" "$VENDOR/omnigibson/" || { echo "ERROR: rsync failed" >&2; exit 1; }

# Provenance, baked into the image. MODE=oglite binds whatever the checkout happens to be at run time,
# so today "which OG-lite did this run use?" is unanswerable after the fact. Once the fork is IN the
# image, its commit is a build-time fact and worth recording where a running container can print it.
( cd "$OG_LITE_ROOT" && {
    echo "og_lite_remote: $(git config --get remote.origin.url 2>/dev/null || echo unknown)"
    echo "og_lite_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "og_lite_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    d=$(git status --porcelain 2>/dev/null | grep -v '^?? ' | wc -l)
    echo "og_lite_dirty_tracked_files: $d"
    [ "$d" != 0 ] && echo "WARNING: staged from a DIRTY checkout -- the image will not match any commit"
    echo "staged_at: $(date -Is)"
  } ) > "$VENDOR/OGLITE_PROVENANCE"
cat "$VENDOR/OGLITE_PROVENANCE"

# Fail HERE rather than at build time if the stage lost a semantic change. Same nine guards the recipes
# assert after installing; checking them twice costs nothing and localises the fault.
echo "--- verifying the nine REALM changes survived the stage ---"
miss=0
check() {  # marker, label, file
  if grep -q "$1" "$VENDOR/omnigibson/$3" 2>/dev/null; then printf "  ok   %s\n" "$2"
  else printf "  MISS %s  (%s)\n" "$2" "$3"; miss=$((miss+1)); fi
}
check "REALM: relaxed" "entity_prim relaxed assertions" prims/entity_prim.py
check "REALM: relaxed" "usd_object relaxed assertions" objects/usd_object.py
check "Re-apply the object poses now that the scene prim is at its final position" \
      "scene_base z-offset" scenes/scene_base.py
check "initialize_obj is obj" "simulator init-queue identity" simulator.py
check "preset_name=None" "material_prim preset default" prims/material_prim.py
check "current_orientation = XFormPrim.get_position_orientation(self)" "xform_prim root-link" \
      prims/xform_prim.py
check "root_local = T.pose2mat(get_local_pose(self.root_link.prim_path))" "entity_prim root-link" \
      prims/entity_prim.py
check "gm.REALM_LIGHT_FIX = " "light fix macros" macros.py
check "if not light_fix:" "light fix dataset_object" objects/dataset_object.py
[ "$miss" = 0 ] || { echo "ERROR: $miss expected change(s) missing from the stage" >&2; exit 1; }

n=$(find "$VENDOR/omnigibson" -name '*.py' | wc -l)
echo "staged $n python files. Now build:"
echo "  apptainer build realm.sif .docker/realm.def"
echo "  docker build -f .docker/realm.Dockerfile -t realm:latest ."
