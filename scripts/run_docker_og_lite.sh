#!/usr/bin/env bash
set -e -o pipefail

BYellow='\033[1;33m'
BRed='\033[1;31m'
BGreen='\033[1;32m'
Color_Off='\033[0m'

# Image built from .docker/realm_og391.Dockerfile (base: stanfordvl/behavior:3.9.1).
REALM_IMAGE="${REALM_IMAGE:-realm:og391}"
# OmniGibson lives at /behavior-src/OmniGibson in the 3.9.1 image (it was /omnigibson-src
# in the old stanfordvl/omnigibson:1.1.1 image). It is a PEP-660 editable install whose
# finder maps 'omnigibson' -> /behavior-src/OmniGibson/omnigibson, so bind-mounting over
# this path swaps the source with no reinstall. OG-lite's repo root is the OmniGibson
# package root (it is a flattened fork), so it mounts here directly.
OG_SRC_MOUNT=/behavior-src/OmniGibson
# The OmniGibson version the image was built against; OG-lite must match it.
IMAGE_OG_VERSION=3.9.1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

# Host paths, all overridable from the environment.
OG_LITE_PATH="${OG_LITE_PATH:-$REALM_ROOT/../OG-lite}"
# What gets mounted at /data (OMNIGIBSON_DATA_PATH in the image). 3.9.1 replaced the old
# assets/ + og_dataset/ layout with behavior-1k-assets/ + omnigibson-robot-assets/, so this
# is NOT the same tree the 1.1.1 script used.
OG_DATA_PATH="${OG_DATA_PATH:-}"

# Parse the command line arguments.
GUI=true
OG_LITE=true  # OG-lite enabled by default in this script
APPLY_PATCHES=false

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--headless)
        GUI=false
        shift
        ;;
        --og-lite|--og_lite)
        OG_LITE=true
        shift
        ;;
        --no-og-lite)
        OG_LITE=false
        shift
        ;;
        --patch-og-lite)
        APPLY_PATCHES=true
        shift
        ;;
        *)
        REALM_DATA_PATH="$1"
        shift
        ;;
    esac
done

REALM_DATA_PATH="${REALM_DATA_PATH:-$REALM_ROOT/data}"
if [ ! -d "$REALM_DATA_PATH" ]; then
    echo -e "${BRed}ERROR: data path '$REALM_DATA_PATH' does not exist.${Color_Off}"
    echo "Pass it as the first positional argument, e.g.: $0 data"
    exit 1
fi
REALM_DATA_PATH=$( cd -- "$REALM_DATA_PATH" &> /dev/null && pwd )
OG_DATA_PATH="${OG_DATA_PATH:-$REALM_DATA_PATH/datasets}"

# ---------------------------------------------------------------------------
# Preflight: OG-lite source
# ---------------------------------------------------------------------------
OG_LITE_BIND=""
if [ "$OG_LITE" = true ]; then
    if [ ! -d "$OG_LITE_PATH/omnigibson" ]; then
        echo -e "${BRed}ERROR: '$OG_LITE_PATH' does not look like an OG-lite checkout"
        echo -e "(no omnigibson/ package inside). Set OG_LITE_PATH or pass --no-og-lite.${Color_Off}"
        exit 1
    fi
    OG_LITE_PATH=$( cd -- "$OG_LITE_PATH" &> /dev/null && pwd )
    OG_LITE_BIND="-v $OG_LITE_PATH:${OG_SRC_MOUNT}:rw"

    # Version gate. The image's site-packages metadata, its dependency pins and the REALM
    # patches below are all tied to 3.9.1; a mismatched source will import but misbehave.
    OG_LITE_VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$OG_LITE_PATH/omnigibson/__init__.py" | head -1)
    if [ "$OG_LITE_VERSION" != "$IMAGE_OG_VERSION" ]; then
        echo -e "${BYellow}WARNING: OG-lite reports OmniGibson $OG_LITE_VERSION but ${REALM_IMAGE} was built"
        echo -e "against $IMAGE_OG_VERSION. Mounting it over ${OG_SRC_MOUNT} will shadow the image's"
        echo -e "source. Pass --no-og-lite to run the image's own OmniGibson instead.${Color_Off}"
    else
        echo -e "${BGreen}OG-lite $OG_LITE_VERSION -> ${OG_SRC_MOUNT} (matches image)${Color_Off}"
    fi

    # The image bakes the REALM patches into /behavior-src/OmniGibson at build time, but the
    # bind mount hides that patched tree. The mounted source therefore needs the same patches
    # applied on the host, or our custom robot USDs will trip OmniGibson's kinematic-tree
    # assertions at load time.
    MISSING_PATCHES=()
    for patchfile in "$REALM_ROOT"/realm/misc/*_og391.patch; do
        [ -e "$patchfile" ] || continue
        # Already applied? Then it reverse-applies cleanly.
        if patch -p1 -d "$OG_LITE_PATH" --dry-run --reverse --force < "$patchfile" > /dev/null 2>&1; then
            continue
        fi
        MISSING_PATCHES+=("$patchfile")
    done

    if [ ${#MISSING_PATCHES[@]} -gt 0 ]; then
        if [ "$APPLY_PATCHES" = true ]; then
            for patchfile in "${MISSING_PATCHES[@]}"; do
                echo "Applying $(basename "$patchfile") to $OG_LITE_PATH"
                patch -p1 -d "$OG_LITE_PATH" --forward < "$patchfile"
            done
        else
            echo -e "${BYellow}WARNING: these REALM patches are baked into the image but NOT present in the"
            echo -e "mounted OG-lite source (the mount shadows the patched copy):${Color_Off}"
            for patchfile in "${MISSING_PATCHES[@]}"; do
                echo -e "${BYellow}  - $(basename "$patchfile")${Color_Off}"
            done
            echo -e "${BYellow}Custom robot USDs will fail OmniGibson's root-link / joint-count assertions."
            echo -e "Re-run with --patch-og-lite to apply them to the OG-lite working tree.${Color_Off}"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Preflight: dataset layout
# ---------------------------------------------------------------------------
if [ ! -d "$OG_DATA_PATH" ]; then
    echo -e "${BRed}ERROR: dataset path '$OG_DATA_PATH' does not exist. Set OG_DATA_PATH.${Color_Off}"
    exit 1
fi
OG_DATA_PATH=$( cd -- "$OG_DATA_PATH" &> /dev/null && pwd )
for required in behavior-1k-assets omnigibson-robot-assets; do
    if [ ! -d "$OG_DATA_PATH/$required" ]; then
        echo -e "${BYellow}WARNING: '$OG_DATA_PATH' has no $required/. OmniGibson $IMAGE_OG_VERSION expects the"
        echo -e "behavior-1k-assets/ + omnigibson-robot-assets/ layout, not the 1.1.1 assets/ + og_dataset/ one."
        echo -e "Fetch it with: python -m omnigibson.utils.asset_utils --download_assets${Color_Off}"
    fi
done

# ---------------------------------------------------------------------------
echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before"
echo "Omniverse Kit can start. The license terms for this product can be viewed at"
echo "https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"

while true; do
    read -p "Do you accept the Omniverse EULA? [y/n] " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

OMNIGIBSON_HEADLESS=1
DOCKER_DISPLAY=""
if [ "$GUI" = true ] ; then
    xhost +local:root
    DOCKER_DISPLAY=$DISPLAY
    OMNIGIBSON_HEADLESS=0
fi

cd $REALM_ROOT
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/kit
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/ov
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/pip
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/glcache
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/computecache
mkdir -p $REALM_DATA_PATH/isaac-sim/logs
mkdir -p $REALM_DATA_PATH/isaac-sim/config
mkdir -p $REALM_DATA_PATH/isaac-sim/data
mkdir -p $REALM_DATA_PATH/isaac-sim/documents
# OMNIGIBSON_APPDATA_PATH=/cache/appdata in the 3.9.1 image -- persist it so shader/asset
# caches survive across runs instead of landing in a throwaway anonymous volume.
mkdir -p $REALM_DATA_PATH/cache

docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DOCKER_DISPLAY} \
    -e OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS} \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e TORCH_CUDA_ARCH_LIST="12.0" \
    -e CUDA_FORCE_PTX_JIT=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/app:rw \
    ${OG_LITE_BIND} \
    -v $OG_DATA_PATH:/data \
    -v $REALM_DATA_PATH/cache:/cache:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v $REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $REALM_DATA_PATH/isaac-sim/documents:/root/Documents:rw \
    -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
    --network=host --rm -it ${REALM_IMAGE}
