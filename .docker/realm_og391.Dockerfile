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

COPY realm/misc/entity_prim_og391.patch /opt/entity_prim_og391.patch
COPY realm/misc/usd_object_og391.patch /opt/usd_object_og391.patch
COPY realm/misc/scene_base_zoffset_og391.patch /opt/scene_base_zoffset_og391.patch
COPY realm/misc/simulator_initqueue_og391.patch /opt/simulator_initqueue_og391.patch
COPY realm/misc/material_prim_preset_og391.patch /opt/material_prim_preset_og391.patch
COPY packages/openpi-client /opt/openpi-client

# Relax the OmniGibson kinematic-tree assertions that reject our custom robot USDs, and fix
# scene-file objects loading ~100 m too high in every scene except index 0 -- without that third
# patch the vectorized eval path is unusable, because the task object has nothing to rest on and
# falls to z=0.015 in scenes 1..N. Measured on this image at num_envs=2 for both Default and V-SC;
# scene 0 passed every check because its origin IS the world origin, which is why no single-env run
# ever showed it. Taken from the OG-lite fork (ef7442b); it is the ONLY part of that fork the
# perturbations need, the rest being performance.
#
# Fails the build loudly if a patch no longer applies to this OmniGibson version, and the greps
# below fail it if a patch silently applied nothing.
RUN patch -p1 -d /behavior-src/OmniGibson < /opt/entity_prim_og391.patch && \
    patch -p1 -d /behavior-src/OmniGibson < /opt/usd_object_og391.patch && \
    patch -p1 -d /behavior-src/OmniGibson < /opt/scene_base_zoffset_og391.patch && \
    patch -p1 -d /behavior-src/OmniGibson < /opt/simulator_initqueue_og391.patch && \
    patch -p1 -d /behavior-src/OmniGibson < /opt/material_prim_preset_og391.patch && \
    rm /opt/entity_prim_og391.patch /opt/usd_object_og391.patch /opt/scene_base_zoffset_og391.patch \
       /opt/simulator_initqueue_og391.patch /opt/material_prim_preset_og391.patch && \
    grep -q "REALM: relaxed" /behavior-src/OmniGibson/omnigibson/prims/entity_prim.py && \
    grep -q "REALM: relaxed" /behavior-src/OmniGibson/omnigibson/objects/usd_object.py && \
    grep -q "Re-apply the object poses now that the scene prim is at its final position" \
        /behavior-src/OmniGibson/omnigibson/scenes/scene_base.py && \
    grep -q "initialize_obj is obj" /behavior-src/OmniGibson/omnigibson/simulator.py && \
    grep -q "preset_name=None" /behavior-src/OmniGibson/omnigibson/prims/material_prim.py

# Keep OmniGibson's own pins (numpy<2, torch 2.7, pydantic) intact.
COPY .docker/og391-constraints.txt /opt/og391-constraints.txt

# dm_control / dm_robotics stack. Pinned to 0.9.0 because 0.10.0 requires numpy>=2,
# which OmniGibson 3.9.1 forbids (numpy<2.0.0,>=1.23.5). 0.9.0 pulls dm-control 1.0.15.
RUN $CONDA_PIP install --no-cache-dir -c /opt/og391-constraints.txt \
    dm-robotics-transformations==0.9.0 \
    dm-robotics-geometry==0.9.0 \
    dm-robotics-controllers==0.9.0

RUN $CONDA_PIP install --no-cache-dir -c /opt/og391-constraints.txt \
    dm-robotics-moma==0.9.0 \
    dm-robotics-manipulation==0.9.0

# Remaining REALM runtime deps not present in the base image.
RUN $CONDA_PIP install --no-cache-dir -c /opt/og391-constraints.txt \
    wandb moviepy openai fastparquet

RUN $CONDA_PIP install --no-cache-dir -c /opt/og391-constraints.txt /opt/openpi-client

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
