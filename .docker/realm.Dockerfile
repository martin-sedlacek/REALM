FROM stanfordvl/behavior:3.9.1

# The base image already provides:
#   - conda env "behavior" (python 3.11) activated by /entrypoint.sh
#   - OmniGibson 3.9.1 + bddl installed editable from /behavior-src
#   - OMNIGIBSON_DATA_PATH=/data, OMNIGIBSON_APPDATA_PATH=/cache/appdata
# so the only path we still need to add is our own source tree at /app.
ENV PYTHONPATH=/app \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]
ENV CONDA_PIP="/opt/conda/envs/behavior/bin/pip"

# The OG-lite fork's omnigibson package, staged into the build context by
# scripts/stage_oglite_for_build.sh. RUN IT FIRST -- this path does not exist otherwise, and a
# docker build cannot COPY from outside its context, which is why the fork is vendored rather than
# referenced at ../OG-lite_og391.
COPY .docker/vendor/omnigibson /opt/oglite/omnigibson
COPY .docker/vendor/OGLITE_PROVENANCE /opt/oglite/OGLITE_PROVENANCE
COPY packages/openpi-client /opt/openpi-client

# INSTALL THE OG-LITE FORK WHOLESALE. THIS IMAGE NEEDS NO RUNTIME BIND.
#
# Was: nine `patch -p1` invocations. The fork already contains all nine (verified 2026-08-20 -- every
# grep guard below is present in it) AND carries work the patch set never covered: the incremental
# contact cache, proximity gate, descriptor-pool raise and render-on-demand plumbing, in
# envs/env_base.py, envs/vec_env_base.py, prims/rigid_prim.py, utils/usd_utils.py. Measured
# fork-vs-image delta: 8 files, 691 changed lines.
#
# Wholesale rather than more patches because `MODE=oglite` bind-mounts this same directory over the
# image's copy, so installing it reproduces that runtime BIT-FOR-BIT BY CONSTRUCTION; a patch set could
# only approximate it, and a generated patch tree has drifted from this fork twice already. There is no
# pristine 3.9.1 locally to diff against either: realm_og391.sif already carries two of the nine,
# realm_og391_v2.sif seven.
#
# COST, so it can be reversed knowingly: the image no longer documents its delta from stock, and a
# future base image's upstream fixes to those eight files would be silently overwritten. The nine grep
# guards are the mitigation, with the fork commit recorded in OGLITE_PROVENANCE.
#
# WHAT THE NINE ARE: relaxed kinematic-tree assertions (entity_prim, usd_object); scene-file objects
# loading ~100 m too high in scenes idx != 0, which makes the vector path unusable; the object-init
# queue pruned by NAME so one member's removal evicted a sibling's uninitialised object;
# OmniSurfaceMaterialPrim requiring preset_name positionally; articulations placed by whichever prim was
# read rather than by their ROOT LINK (impact_drawer's base_link is at Ry(180) + 4 cm, so the drawer
# tasks could not load); and REALM_LIGHT_FIX, ON BY DEFAULT -- 3.9.1 took FORCE_LIGHT_INTENSITY
# 150000 -> 10000 while also writing inputs:normalize=True, which cancel exactly at area 1/15 m^2 and
# leave a PER-SCENE error; the flag restores 1.1.1's configuration, tightening the per-task spread
# 0.551 -> 0.201 over the 6 comparable tasks (2.74x). AN IMAGE BUILT FROM THIS RECIPE IS 1.1.1-LIT
# UNLESS A RUN SETS REALM_LIGHT_FIX=0.
#
# rm -rf then cp -a, never a merge: a file the fork DELETED must not survive from the stock tree.
# The greps then fail the build if any semantic change did not survive the stage.
# EXISTING IMAGES STILL NEED THE BIND: realm_og391.sif / _v2.sif came from the old patch-based recipe
# (2 and 7 of the nine, none of the perf work), so MODE=oglite stays REQUIRED against them. rr's modes
# are deliberately unchanged -- flipping the default would silently change what every run loads against
# images that lack this. With an image built from THIS recipe, MODE=stock suffices.
RUN test -d /opt/oglite/omnigibson || { echo "OG-lite not staged; run scripts/stage_oglite_for_build.sh" >&2; exit 1; } && \
    rm -rf /behavior-src/OmniGibson/omnigibson && \
    cp -a /opt/oglite/omnigibson /behavior-src/OmniGibson/omnigibson && \
    cp -a /opt/oglite/OGLITE_PROVENANCE /behavior-src/OmniGibson/OGLITE_PROVENANCE && \
    find /behavior-src/OmniGibson/omnigibson -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    rm -rf /opt/oglite/omnigibson && \
    cat /behavior-src/OmniGibson/OGLITE_PROVENANCE && \
    grep -q "REALM: relaxed" /behavior-src/OmniGibson/omnigibson/prims/entity_prim.py && \
    grep -q "REALM: relaxed" /behavior-src/OmniGibson/omnigibson/objects/usd_object.py && \
    grep -q "Re-apply the object poses now that the scene prim is at its final position" \
        /behavior-src/OmniGibson/omnigibson/scenes/scene_base.py && \
    grep -q "initialize_obj is obj" /behavior-src/OmniGibson/omnigibson/simulator.py && \
    grep -q "preset_name=None" /behavior-src/OmniGibson/omnigibson/prims/material_prim.py && \
    grep -q "current_orientation = XFormPrim.get_position_orientation(self)" \
        /behavior-src/OmniGibson/omnigibson/prims/xform_prim.py && \
    grep -q "root_local = T.pose2mat(get_local_pose(self.root_link.prim_path))" \
        /behavior-src/OmniGibson/omnigibson/prims/entity_prim.py && \
    grep -q "gm.REALM_LIGHT_FIX = " /behavior-src/OmniGibson/omnigibson/macros.py && \
    grep -q "if not light_fix:" /behavior-src/OmniGibson/omnigibson/objects/dataset_object.py

# Keep OmniGibson's own pins (numpy<2, torch 2.7, pydantic) intact.
COPY .docker/constraints.txt /opt/realm-constraints.txt

# dm_control / dm_robotics stack. Pinned to 0.9.0 because 0.10.0 requires numpy>=2,
# which OmniGibson 3.9.1 forbids (numpy<2.0.0,>=1.23.5). 0.9.0 pulls dm-control 1.0.15.
RUN $CONDA_PIP install --no-cache-dir -c /opt/realm-constraints.txt \
    dm-robotics-transformations==0.9.0 \
    dm-robotics-geometry==0.9.0 \
    dm-robotics-controllers==0.9.0

RUN $CONDA_PIP install --no-cache-dir -c /opt/realm-constraints.txt \
    dm-robotics-moma==0.9.0 \
    dm-robotics-manipulation==0.9.0

# Remaining REALM runtime deps not present in the base image.
RUN $CONDA_PIP install --no-cache-dir -c /opt/realm-constraints.txt \
    wandb moviepy openai fastparquet

RUN $CONDA_PIP install --no-cache-dir -c /opt/realm-constraints.txt /opt/openpi-client

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
