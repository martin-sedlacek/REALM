# REALM Project Context

This project uses an Apptainer container for evaluations and simulation with OmniGibson.

## Container Skill
The `realm-container` skill is installed at the user level. Activate it to get instructions for working with the container.

```bash
/skills enable realm-container
```

## Interactive Session
To run an interactive session in the REALM container:

```bash
./scripts/run_apptainer.sh
```

Or use the direct command:

```bash
apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd)":/app \
  --bind "$REALM_DATA_PATH"/datasets:/data \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/ov:/root/.cache/ov \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/pip:/root/.cache/pip \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
  --bind "$REALM_DATA_PATH"/isaac-sim/logs:/root/.nvidia-omniverse/logs \
  --bind "$REALM_DATA_PATH"/isaac-sim/config:/root/.nvidia-omniverse/config \
  --bind "$REALM_DATA_PATH"/isaac-sim/data:/root/.local/share/ov/data \
  --bind "$REALM_DATA_PATH"/isaac-sim/documents:/root/Documents \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  $REALM_SIF \
  micromamba run -n omnigibson bash
```

## Verification
Inside the container, verify the environment with:
```bash
python -c "import omnigibson; print('Import omnigibson successful'); print(omnigibson.__version__)"
```
