FROM stanfordvl/behavior:3.9.1

# The base image already provides:
#   - conda env "behavior" (python 3.11) activated by /entrypoint.sh
#   - OmniGibson 3.9.1 + bddl installed editable from /behavior-src
#   - OMNIGIBSON_DATA_PATH=/data, OMNIGIBSON_APPDATA_PATH=/cache/appdata
# so the only path we still need to add is our own source tree at /app.
#
# THIS BUILD IS SELF-CONTAINED: only this repository, no sibling OG-lite checkout, no staging step,
# no runtime bind. Apptainer counterpart: .docker/realm.def -- keep the two in sync.
ENV PYTHONPATH=/app \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]
ENV CONDA_PIP="/opt/conda/envs/behavior/bin/pip"

COPY .docker/patches /opt/realm-patches
COPY packages/openpi-client /opt/openpi-client
COPY packages/robometer-client /opt/robometer-client

# APPLY REALM'S DELTA FROM STOCK OMNIGIBSON 3.9.1.
#
# .docker/patches/ is the image's COMPLETE delta from stock; .docker/patches/PROVENANCE records how
# it was derived from the base revision and how to regenerate it. Twelve patches in filename order,
# then verified two ways: MANIFEST.sha256 proves every patched file hashes to the tree the patches
# were generated from (a patch applying with fuzz, or a base image that moved under us, fails the
# build), and nine grep guards name the semantic changes so a failure says WHICH behaviour is gone.
#
# A future base image's upstream fixes to these twelve files will make a patch FAIL TO APPLY. That
# is deliberate: the build stops rather than silently discarding an upstream fix, which is what the
# wholesale-copy recipe this replaces did.
#
# WHAT THE TWELVE CARRY: render-on-demand plumbing (envs/env_base, envs/vec_env_base); the perf
# switches realm/sim_config.py sets, and REALM_LIGHT_FIX (macros); that flag's other half, the
# inputs:normalize gate (objects/dataset_object); relaxed kinematic-tree assertions for REALM's
# custom robot USDs (objects/usd_object, prims/entity_prim); placing an articulation by its ROOT
# LINK, without which impact_drawer's base_link at Ry(180) + 4 cm stopped the drawer tasks loading
# (prims/entity_prim, prims/xform_prim); OmniSurfaceMaterialPrim's preset_name default, without
# which any asset resolving to OmniSurface failed to load (prims/material_prim); the incremental
# contact cache (prims/rigid_prim); the scene z-offset, without which scene-file objects load
# ~100 m high in every scene but 0 and the vectorized path is unusable (scenes/scene_base); the
# object-init queue matched by identity rather than by NAME, so removing one vector member's object
# no longer evicts a sibling's uninitialised one, plus the proximity gate (simulator); and the
# descriptor-pool raise (utils/usd_utils).
#
# REALM_LIGHT_FIX IS ON BY DEFAULT, so THIS IMAGE IS 1.1.1-LIT unless a run sets REALM_LIGHT_FIX=0.
# 3.9.1 took FORCE_LIGHT_INTENSITY 150000 -> 10000 while also writing inputs:normalize=True; the two
# cancel exactly at light area 1/15 m^2, leaving a PER-SCENE error. Restoring 1.1.1's configuration
# tightens the per-task spread 0.551 -> 0.201 over the 6 comparable tasks (2.74x), turning a
# per-scene domain shift into one global gain. macros.py prints the resolved state at import.
RUN cd /behavior-src/OmniGibson && \
    for p in /opt/realm-patches/*.patch; do \
        echo "--- applying $(basename "$p")" && \
        patch -p1 --forward --batch --no-backup-if-mismatch < "$p" || exit 1; \
    done && \
    sha256sum -c /opt/realm-patches/MANIFEST.sha256 && \
    cp /opt/realm-patches/PROVENANCE /behavior-src/OmniGibson/REALM_PATCH_PROVENANCE && \
    find /behavior-src/OmniGibson/omnigibson -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
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

# Robometer reward-model CLIENT only (numpy + requests, both already pinned above). The model server
# is packages/robometer, run in its own env via scripts/run_robometer_server.sh -- its torch 2.8 /
# transformers 4.57 / python==3.10 requirements cannot be installed here. Same arrangement as openpi.
RUN $CONDA_PIP install --no-cache-dir -c /opt/realm-constraints.txt /opt/robometer-client

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
