#!/usr/bin/env bash
set -e -o pipefail

# Image built from .docker/realm.Dockerfile (base: stanfordvl/behavior:3.9.1).
REALM_IMAGE="${REALM_IMAGE:-realm:latest}"
# OmniGibson lives at /behavior-src/OmniGibson in the 3.9.1 image (was /omnigibson-src in 1.1.1).
OG_SRC_MOUNT=/behavior-src/OmniGibson

# Parse command line arguments
TASK_ID=0
OG_LITE=false
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task_id)
        TASK_ID="$2"
        shift; shift
        ;;
        --og-lite|--og_lite)
        OG_LITE=true
        shift
        ;;
        *)
        REALM_DATA_PATH="$1"
        shift
        ;;
    esac
done

if [ -z "$REALM_DATA_PATH" ]; then
    echo "Usage: $0 <REALM_DATA_PATH> [--task_id N]"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

cd "$REALM_ROOT"
mkdir -p "$REALM_DATA_PATH"/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
mkdir -p "$REALM_DATA_PATH"/isaac-sim/{logs,config,data,documents}
# OMNIGIBSON_APPDATA_PATH=/cache/appdata in the 3.9.1 image -- persist it across runs.
mkdir -p "$REALM_DATA_PATH"/cache

OG_LITE_BIND=""
if [ "$OG_LITE" = true ]; then
    OG_LITE_BIND="-v $REALM_ROOT/../OG-lite:${OG_SRC_MOUNT}:rw"
fi

echo "Running single-task integrity test (task_id=$TASK_ID) inside Docker..."

docker run \
    --gpus all \
    --privileged \
    -e OMNIGIBSON_HEADLESS=1 \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e TORCH_CUDA_ARCH_LIST="12.0" \
    -e CUDA_FORCE_PTX_JIT=1 \
    -v $(pwd):/app:rw \
    ${OG_LITE_BIND} \
    -v "$REALM_DATA_PATH"/datasets:/data \
    -v "$REALM_DATA_PATH"/cache:/cache:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v "$REALM_DATA_PATH"/isaac-sim/documents:/root/Documents:rw \
    -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
    --network=host --rm ${REALM_IMAGE} \
    python /app/tests/test_single_task.py --task_id "$TASK_ID"

echo ""
if [ $? -eq 0 ]; then
    echo "TEST PASSED"
else
    echo "TEST FAILED"
    exit 1
fi
