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
COPY packages/openpi-client /opt/openpi-client

# Relax the OmniGibson kinematic-tree assertions that reject our custom robot USDs.
# Fails the build loudly if a patch no longer applies to this OmniGibson version.
RUN patch -p1 -d /behavior-src/OmniGibson < /opt/entity_prim_og391.patch && \
    patch -p1 -d /behavior-src/OmniGibson < /opt/usd_object_og391.patch && \
    rm /opt/entity_prim_og391.patch /opt/usd_object_og391.patch

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
