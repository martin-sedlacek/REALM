#!/bin/bash

# fail fast
set -e

# --- default values ---
# get script's directory to use as project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# dev container type, can be 'docker' or 'apptainer'
CONTAINER_TYPE=""
# by default, we don't download the dataset
DOWNLOAD_DATASET=false
# by default, we don't force build
FORCE_BUILD=false
# default data path for the datasets
DATA_PATH="$SCRIPT_DIR/data"
# custom path for the SIF file
SIF_PATH=""

# --- script arguments ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --docker)
      if [ -n "$CONTAINER_TYPE" ]; then
        echo "Error: --docker and --apptainer cannot be used together." >&2
        exit 1
      fi
      CONTAINER_TYPE="docker"
      shift # past argument
      ;;
    --apptainer)
      if [ -n "$CONTAINER_TYPE" ]; then
        echo "Error: --docker and --apptainer cannot be used together." >&2
        exit 1
      fi
      CONTAINER_TYPE="apptainer"
      shift # past argument
      ;;
    --force-build)
      FORCE_BUILD=true
      shift # past argument
      ;;
    --dataset)
      DOWNLOAD_DATASET=true
      shift # past argument
      ;;
    --data-path)
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    --sif-path)
      SIF_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# check for sif-path and container type consistency
if [ -n "$SIF_PATH" ] && [ "$CONTAINER_TYPE" != "apptainer" ]; then
  echo "Error: --sif-path can only be used with --apptainer." >&2
  exit 1
fi

# if no container type is specified, default to docker
if [ -z "$CONTAINER_TYPE" ]; then
  CONTAINER_TYPE="docker"
fi

# --- helper functions ---

# this function adds or updates an environment variable in ~/.bashrc
add_to_bashrc() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: add_to_bashrc VAR_NAME VAR_VALUE" >&2
        return 1
    fi

    local VAR_NAME="$1"
    local VAR_VALUE="$2"
    local SCRIPT_NAME="setup.sh"

    case "$VAR_NAME" in
      [a-zA-Z_][a-zA-Z0-9_]*) ;;
      *)
        echo "Invalid variable name: $VAR_NAME" >&2
        return 1
        ;;
    esac

    local BASHRC="$HOME/.bashrc"
    printf -v EXPORT_LINE 'export %s=%q' "$VAR_NAME" "$VAR_VALUE"
    local MARKER_BEGIN="# >>> $VAR_NAME (managed by REALM:$SCRIPT_NAME) >>>"
    local MARKER_END="# <<< $VAR_NAME (managed by REALM:$SCRIPT_NAME) <<<"

    touch "$BASHRC"

    local tmp
    tmp="$(mktemp "${BASHRC}.XXXXXX")" || {
      echo "Failed to create temporary file for $BASHRC" >&2
      return 1
    }

    if ! awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '\
    BEGIN { inblock = 0 }
    $0 == b { inblock = 1; next }
    $0 == e { inblock = 0; next }
    !inblock { print }
    ' "$BASHRC" > "$tmp"; then
      echo "Failed to process $BASHRC with awk" >&2
      rm -f "$tmp"
      return 1
    fi

    if ! mv "$tmp" "$BASHRC"; then
      echo "Failed to replace $BASHRC with updated version" >&2
      rm -f "$tmp"
      return 1
    fi

    {
      echo
      echo "$MARKER_BEGIN"
      echo "$EXPORT_LINE"
      echo "$MARKER_END"
    } >> "$BASHRC" || {
      echo "Failed to append managed block to $BASHRC" >&2
      return 1
    }

    echo "$VAR_NAME added to ~/.bashrc:"
    echo "  $EXPORT_LINE"
    echo "Run:  source \"$BASHRC\"  or open a new shell to apply it."
}

build_apptainer_image() {
    local OUT_SIF
    if [ -n "$SIF_PATH" ]; then
        OUT_SIF="$SIF_PATH"
    else
        OUT_SIF="$SCRIPT_DIR/realm.sif"
    fi

    # Check if the image already exists and we are not forcing a build
    if [ "$FORCE_BUILD" = false ] && [ -f "$OUT_SIF" ]; then
        echo "Image $OUT_SIF already exists, skipping build."
        export REALM_SIF=$OUT_SIF
        add_to_bashrc "REALM_SIF" "$OUT_SIF"
        return
    fi

    local DEF_FILE="$SCRIPT_DIR/.docker/realm.def"
    if [[ ! -f "$DEF_FILE" ]]; then
      echo "ERROR: Definition file not found: $DEF_FILE" >&2
      exit 1
    fi

    local BUILDER=""
    if command -v apptainer >/dev/null 2>&1; then
      BUILDER="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
      BUILDER="singularity"
    else
      echo "ERROR: Neither 'apptainer' nor 'singularity' found in PATH." >&2
      exit 1
    fi

    mkdir -p "$(dirname "$OUT_SIF")"

    echo "Building Singularity image"
    echo "  Definition file : $DEF_FILE"
    echo "  Output image    : $OUT_SIF"
    echo "  Builder         : $BUILDER"
    echo

    if ! "$BUILDER" build "$OUT_SIF" "$DEF_FILE"; then
      echo "Build failed without privileges, retrying with sudo..."
      sudo "$BUILDER" build "$OUT_SIF" "$DEF_FILE"
    fi

    echo
    echo "Image successfully built: $OUT_SIF"
    echo

    export REALM_SIF=$OUT_SIF
    add_to_bashrc "REALM_SIF" "$OUT_SIF"
}

# --- main logic ---

if [ "$CONTAINER_TYPE" == "docker" ]; then
    if [ "$FORCE_BUILD" = false ] && docker image inspect realm:latest &> /dev/null; then
        echo "Docker image realm:latest already exists, skipping build."
    else
        echo "Building docker image from .docker/realm.Dockerfile..."
        docker build -t realm:latest -f .docker/realm.Dockerfile .
    fi

elif [ "$CONTAINER_TYPE" == "apptainer" ]; then
    build_apptainer_image
fi

if [ "$DOWNLOAD_DATASET" = true ]; then
    echo "Downloading dataset to $DATA_PATH..."
    mkdir -p "$DATA_PATH/datasets"

    # THE DOWNLOAD ENTRY POINT MOVED IN OMNIGIBSON 3.9.1. There is no `omnigibson.download_datasets`
    # module any more (and no micromamba env named `omnigibson` -- 3.9.1 ships conda env `behavior`
    # on python 3.11). Downloads now go through omnigibson/utils/asset_utils.py's __main__:
    #   --download_behavior_1k_assets      behavior-1k-assets 3.9.0  (~29 GiB zip -> ~33 GB)
    #   --download_omnigibson_robot_assets omnigibson-robot-assets 3.8.2  (~2.8 GB)
    #   --accept_license                   skips the interactive BEHAVIOR EULA prompt; the
    #                                      decryption key is fetched only if <data>/omnigibson.key
    #                                      is absent
    # Both are pulled from HuggingFace (behavior-1k/zipped-datasets), so the host needs outbound
    # HTTPS to huggingface.co.
    OG_DOWNLOAD_ARGS="--download_behavior_1k_assets --download_omnigibson_robot_assets"

    # download_and_unpack_zipped_dataset() stages the whole zip in tempfile.mkdtemp() before
    # extracting, so TMPDIR needs ~30 GiB of REAL disk. Under apptainer's --writable-tmpfs that
    # would be a 64 MB in-RAM overlay and the download would die with ENOSPC part-way, which is why
    # a host directory is bound over /tmp below rather than leaving it to the container.
    mkdir -p "$DATA_PATH/download_tmp"

    if [ "$CONTAINER_TYPE" == "docker" ]; then
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
        add_to_bashrc "OMNIVERSE_EULA_ACCEPTED" "1"

        docker run \
            --gpus all \
            --privileged \
            -e OMNIVERSE_EULA_ACCEPTED=1 \
            -e OMNIGIBSON_HEADLESS=1 \
            -e OMNI_KIT_ALLOW_ROOT=1 \
            -e TMPDIR=/download_tmp \
            -v "${DATA_PATH}/datasets:/data" \
            -v "${DATA_PATH}/download_tmp:/download_tmp" \
            --network=host --rm -it realm:latest \
            python -m omnigibson.utils.asset_utils $OG_DOWNLOAD_ARGS --accept_license

    elif [ "$CONTAINER_TYPE" == "apptainer" ]; then
        if [[ -z "${REALM_SIF:-}" ]]; then
          echo "REALM_SIF is not set. Please set it to the location of your singularity image."
          exit 1
        fi
        
        BUILDER=""
        if command -v apptainer >/dev/null 2>&1; then
          BUILDER="apptainer"
        elif command -v singularity >/dev/null 2>&1; then
          BUILDER="singularity"
        else
          echo "ERROR: Neither 'apptainer' nor 'singularity' found in PATH." >&2
          exit 1
        fi
        
        # `run`, not `exec`: only the %runscript activates the conda env OmniGibson lives in.
        # No --writable-tmpfs and no --nv: the download needs neither a GPU nor a writable rootfs,
        # and the tmpfs overlay is only 64 MB (see the TMPDIR note above).
        $BUILDER run \
            --userns \
            --bind "${DATA_PATH}/datasets:/data" \
            --bind "${DATA_PATH}/download_tmp:/download_tmp" \
            --env TMPDIR=/download_tmp \
            "$REALM_SIF" \
            python -m omnigibson.utils.asset_utils $OG_DOWNLOAD_ARGS --accept_license
    fi

    # ensure Isaac Sim directories exist (may be missing after download)
    mkdir -p "$DATA_PATH/isaac-sim/cache"
    mkdir -p "$DATA_PATH/isaac-sim/config"
    mkdir -p "$DATA_PATH/isaac-sim/documents"
    mkdir -p "$DATA_PATH/isaac-sim/logs"
    mkdir -p "$DATA_PATH/isaac-sim/data"

    add_to_bashrc "REALM_DATA_PATH" "$DATA_PATH"
    echo "Finished downloading datasets."
fi

echo "Setup complete."
