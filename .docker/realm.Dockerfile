FROM stanfordvl/omnigibson:1.1.1

ENV OMNIGIBSON_DATASET_PATH=/data/og_dataset \
    OMNIGIBSON_ASSET_PATH=/data/assets \
    GIBSON_DATASET_PATH=/data/g_dataset \
    OMNIGIBSON_KEY_PATH=/data/omnigibson.key \
    PYTHONPATH=$PYTHONPATH:/app \
    MAMBA_ROOT_PREFIX=/micromamba \
    PATH="/micromamba/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY realm/misc/modified_entity_prim.py /opt/modified_entity_prim.py
COPY packages/openpi-client /opt/openpi-client

RUN micromamba install -n omnigibson -y -c conda-forge wandb moviepy && \
    micromamba run -n omnigibson pip install /opt/openpi-client && \
    cp /opt/modified_entity_prim.py /omnigibson-src/omnigibson/prims/entity_prim.py && \
    rm /opt/modified_entity_prim.py

WORKDIR /omnigibson-src

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]
CMD ["/bin/bash", "--login"]