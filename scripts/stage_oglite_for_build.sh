#!/bin/bash
# RETIRED 2026-08-28. The image build no longer stages a sibling OG-lite checkout: .docker/patches/
# is the complete delta from stock OmniGibson 3.9.1 and is applied by .docker/realm.Dockerfile /
# .docker/realm.def directly. See .docker/patches/PROVENANCE.
#
# This script also cannot succeed as written: its own guard requires both halves of REALM_LIGHT_FIX
# in the fork, and they were never committed to it.
echo "stage_oglite_for_build.sh is retired -- the build applies .docker/patches/ directly." >&2
echo "See .docker/patches/PROVENANCE." >&2
exit 1

# ---- original script kept below for reference ----
# Stage OG-lite inside the container build context; builders cannot copy from a sibling checkout.
#
#     ./scripts/stage_oglite_for_build.sh                     # uses ../OG-lite_og391
#     OG_LITE_ROOT=/path/to/OG-lite_og391 ./scripts/stage_oglite_for_build.sh
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

# Remove stale source and bytecode from previous stages.
rsync -a --delete \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/' \
      "$OG_LITE_ROOT/omnigibson/" "$VENDOR/omnigibson/" || { echo "ERROR: rsync failed" >&2; exit 1; }

# Record the exact staged checkout in the image.
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

# Verify the REALM-specific OmniGibson changes before building.
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
