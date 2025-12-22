#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# --------------------------------------------------------------------------------------
# Help / usage
# --------------------------------------------------------------------------------------
print_help() {
    cat <<EOF
Usage: $(basename "$0") -c PATH [OPTIONS]

Required:
    -c, --ckpt-path PATH       Host path to the model checkpoint

Rest of options:
  -p, --perturbation-id ID   Perturbation ID [0–15]. Default: 0
                               0: Default
                               1: V-AUG
                               2: V-VIEW
                               3: V-SC
                               4: V-LIGHT
                               5: S-PROP
                               6: S-LANG
                               7: S-MO
                               8: S-AFF
                               9: S-INT
                              10: B-HOBJ
                              11: SB-NOUN
                              12: SB-VRB
                              13: VB-POSE
                              14: VB-MOBJ
                              15: VSB-NOBJ

  -t, --task-id ID           Task ID [0–9]. Default: 0
                               0: put_green_block_in_bowl
                               1: put_banana_into_box
                               2: rotate_marker
                               3: rotate_mug
                               4: pick_spoon
                               5: pick_water_bottle
                               6: stack_cubes
                               7: push_switch
                               8: open_drawer
                               9: close_drawer

  -r, --repeats N            Number of episodes. Default: 25
  -s, --max-steps N          Max steps per episode before termination. Default: 500

  -m, --model MODEL          Either:
                               - one of: pi0 | pi0_FAST | GR00T
                               - or path to an executable script that starts a model server
                             Default: pi0

  -e, --environment ENV      Evaluation environment: singularity | docker | current
                             Default: singularity

Other:
  -h, --help                 Show this help and exit

Examples:
  # pi0 evaluation with default settings on 'put_green_block_in_bowl' task
  $0 -c /path/to/pi0/checkpoint
  # Small run of GR00T evaluation on 'rotate_marker' task with 'V-SC' (distractors) visual perturbation through docker
  $0 -p 3 -t 2 -r 1 -s 50 -m GR00T -c /path/to/gr00t/checkpoint -e docker
  # Standard run of pi0_FAST evaluation on 'stark_cubes' task with 'S_MO' (spatial ref.) semantic perturbation through docker
  $0 -p 7 -t 6 -m pi0_FAST -c /path/to/pi0_FAST/checkpoint
  # Custom model run
  $0 -m /path/to/model/server/script -c /path/to/checkpoint


EOF
}

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------
PERTURBATION_ID=0
TASK_ID=0
REPEATS=25
MAX_STEPS=500
MODEL="pi0"
CKPT_PATH=""             # required
EVAL_ENV="singularity"

# --------------------------------------------------------------------------------------
# Parse flag arguments
# --------------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--perturbation-id)
            if [[ $# -lt 2 ]]; then
                echo "Error: --perturbation-id requires a value" >&2
                exit 1
            fi
            PERTURBATION_ID="$2"
            shift 2
            ;;
        -t|--task-id)
            if [[ $# -lt 2 ]]; then
                echo "Error: --task-id requires a value" >&2
                exit 1
            fi
            TASK_ID="$2"
            shift 2
            ;;
        -r|--repeats)
            if [[ $# -lt 2 ]]; then
                echo "Error: --repeats requires a value" >&2
                exit 1
            fi
            REPEATS="$2"
            shift 2
            ;;
        -s|--max-steps)
            if [[ $# -lt 2 ]]; then
                echo "Error: --max-steps requires a value" >&2
                exit 1
            fi
            MAX_STEPS="$2"
            shift 2
            ;;
        -m|--model)
            if [[ $# -lt 2 ]]; then
                echo "Error: --model requires a value" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -c|--ckpt-path)
            if [[ $# -lt 2 ]]; then
                echo "Error: --ckpt-path requires a value" >&2
                exit 1
            fi
            CKPT_PATH="$2"
            shift 2
            ;;
        -e|--environment)
            if [[ $# -lt 2 ]]; then
                echo "Error: --environment requires a value" >&2
                exit 1
            fi
            EVAL_ENV="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: Unknown option '$1'" >&2
            echo
            print_help
            exit 1
            ;;
        *)
            echo "Error: Unexpected positional argument '$1'" >&2
            echo
            print_help
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------------------
# Required argument checks after parsing
# --------------------------------------------------------------------------------------
if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: --ckpt-path is required." >&2
    echo
    print_help
    exit 1
fi

# Validating args -----------------------------------------------------------------------------------
if ! [[ "$PERTURBATION_ID" =~ ^[0-9]+$ ]] || [ "$PERTURBATION_ID" -lt 0 ] || [ "$PERTURBATION_ID" -gt 15 ]; then
    echo "Error: PERTURBATION_ID must be an integer between 0 and 15 (got '$PERTURBATION_ID')."
    exit 1
fi

case "$TASK_ID" in
    [0-9]) ;;
    *) echo "task_id must be an integer in [0-9], got '$TASK_ID'"; exit 1;;
esac

[[ "$REPEATS" =~ ^[0-9]+$ ]] || { echo "repeats must be integer"; exit 1; }
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "max_steps must be integer"; exit 1; }
[ -e "$CKPT_PATH" ] || { echo "checkpoint $CKPT_PATH does not exist"; exit 1; }

# Validate env
case "$EVAL_ENV" in
    singularity|docker|current)
        ;;
    *)
        echo "Invalid value for environment: '$EVAL_ENV'"
        echo "Allowed values: singularity, docker, current"
        exit 1
        ;;
esac

case "$MODEL" in
    pi0|pi0_FAST|GR00T)
        ;;
    *)
        echo "$MODEL is not in (pi0|pi0_FAST|GR00T). Assuming that it is an executable file."
        if [ ! -e "$MODEL" ]; then
            echo "Error: MODEL '$MODEL' does not exist."
            exit 1
        fi

        if [ ! -f "$MODEL" ]; then
            echo "Error: MODEL '$MODEL' exists but is not a regular file."
            exit 1
        fi

        if [ ! -x "$MODEL" ]; then
            echo "Error: MODEL '$MODEL' is not executable. Run: chmod +x \"$MODEL\""
            exit 1
        fi
        ;;
esac

# Checking dependencies ------------------------------------------------------------------------------------

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

if [[ -z "${REALM_DATA_PATH:-}" ]]; then
  echo "REALM_DATA_PATH is not set."
  echo "Set it to the path where omnigibson dataset and IsaacSim cache located to use, e.g.:"
  echo "  export REALM_DATA_PATH=\"/path/to/the/realm/data\""
  echo "Run ./scripts/download_dataset.sh , if you haven't downloaded data yet"
  exit 1
fi

case "$EVAL_ENV" in
    singularity)
        if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
            echo "Need either 'singularity' or 'apptainer' available to run through singularity environment."
            exit 1
        fi
        ;;
    docker)
        if ! command -v docker >/dev/null 2>&1; then
            echo "Missing 'docker' in the \$PATH, not able to run through 'docker' environment"
            exit 1
        fi
        if [ "${OMNIVERSE_EULA_ACCEPTED:-0}" -ne 1 ]; then
            if [ ! -t 0 ]; then
                echo "Non-interactive shell detected. Please set OMNIVERSE_EULA_ACCEPTED=1 to skip the prompt."
                exit 1
            else
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
                export OMNIVERSE_EULA_ACCEPTED=1
                #"$SCRIPT_DIR/add_to_bashrc.sh" OMNIVERSE_EULA_ACCEPTED 1 EULA
            fi
        fi
        ;;
esac

SIF_CMD=""
if command -v apptainer >/dev/null 2>&1; then
    SIF_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    SIF_CMD="singularity"
fi

if [ -z "${SIF_CMD:-}" ] && { [ "$MODEL" = "pi0" ] || [ "$MODEL" = "pi0_FAST" ]; }; then
    echo "pi0 and pi0_FAST models require singularity/apptainer to be installed"
    exit 1
fi

# Port logic ---------------------------------------------------------------------------------------

# Pick a port-checking command once
PORT_CHECK_CMD=""

if command -v nc >/dev/null 2>&1; then
    PORT_CHECK_CMD="nc"
elif command -v ss >/dev/null 2>&1; then
    PORT_CHECK_CMD="ss"
else
    echo "Need either 'nc' or 'ss' available to check the port."
    exit 1
fi


# Helper: is a port free? Returns 0 if free, 1 if taken
port_is_free() {
    local port="$1"
    case "$PORT_CHECK_CMD" in
        nc)
            # nc -z: check if something is listening; if nothing, return nonzero
            if nc -z localhost "$port" 2>/dev/null; then
                return 1  # something is using the port
            else
                return 0  # free
            fi
            ;;
        ss)
            # ss: search for anything listening on that port
            if ss -ltn "sport = :$port" 2>/dev/null | grep -q "$port"; then
                return 1  # taken
            else
                return 0  # free
            fi
            ;;
        *)
            echo "Internal error: unsupported PORT_CHECK_CMD '$PORT_CHECK_CMD'" >&2
            return 1
            ;;
    esac
}

# Helper: wait until a port is LISTENING. Returns 0 if listening, 1 on timeout
wait_for_port() {
    local port="$1"
    local timeout="${2:-180}"
    local interval=2
    local elapsed=0

    while (( elapsed < timeout )); do
        case "$PORT_CHECK_CMD" in
            nc)
                if nc -z localhost "$port" 2>/dev/null; then
                    return 0
                fi
                ;;
            ss)
                if ss -ltn "sport = :$port" 2>/dev/null | grep -q "$port"; then
                    return 0
                fi
                ;;
        esac
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done

    return 1
}

# Helper: choose a port + lock it
DEFAULT_LOCK_DIR="/tmp/model_server_ports"
LOCK_DIR="$DEFAULT_LOCK_DIR"

# Ensure LOCK_DIR exists and is usable by this user.
# If not, fall back to a per-user directory and warn about cross-user collisions.
if ! mkdir -p "$LOCK_DIR" 2>/dev/null; then
    # Cannot create or access the shared lock dir at all -> fall back
    LOCK_DIR="/tmp/model_server_ports_${USER:-uid_$(id -u)}"
    mkdir -p "$LOCK_DIR"
    echo "WARNING: cannot use shared lock dir '$DEFAULT_LOCK_DIR'; using per-user lock dir '$LOCK_DIR'." >&2
    echo "         Port uniqueness will NOT be enforced across different users on this node." >&2
elif [ ! -w "$LOCK_DIR" ] || [ ! -x "$LOCK_DIR" ]; then
    # Exists but not writable/executable by current user -> fall back
    LOCK_DIR="/tmp/model_server_ports_${USER:-uid_$(id -u)}"
    mkdir -p "$LOCK_DIR"
    echo "WARNING: shared lock dir '$DEFAULT_LOCK_DIR' is not writable/executable by this user; using '$LOCK_DIR' instead." >&2
    echo "         Port uniqueness will NOT be enforced across different users on this node." >&2
fi

choose_model_port() {
    local base_port=20000
    local max_port=25000
    local lock_dir="$LOCK_DIR"

    mkdir -p "$lock_dir"

    for (( i=0; i<1000; i++ )); do
        local candidate=$(( base_port + RANDOM % (max_port - base_port) ))

        # if port already has a listener, skip
        if ! port_is_free "$candidate"; then
            continue
        fi

        # Try to atomically "lock" this port
        if mkdir "$lock_dir/$candidate.lock" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
        # else: another process won the race; try again
    done

    echo "ERROR: Could not allocate a free port in range ${base_port}-${max_port}" >&2
    return 1
}


# Launching a model ---------------------------------------------------------------------------------

HF_HOME="${HF_HOME:-$REALM_ROOT/.cache/hugging_face}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REALM_ROOT/.cache/xdg}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE
export XDG_CACHE_HOME

mkdir -p "$HF_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
mkdir -p "$XDG_CACHE_HOME"

#"$SCRIPT_DIR/add_to_bashrc.sh" HF_HOME "$HF_HOME" eval.sh
#"$SCRIPT_DIR/add_to_bashrc.sh" HUGGINGFACE_HUB_CACHE "$HUGGINGFACE_HUB_CACHE" eval.sh
#"$SCRIPT_DIR/add_to_bashrc.sh" XDG_CACHE_HOME "$XDG_CACHE_HOME" eval.sh

PORT=""

cleanup() {
    # Kill the whole process group if we know it
    if [[ -n "${SERVER_PGID:-}" ]]; then
        echo "Cleaning up server process group $SERVER_PGID" >&2
        kill -TERM "-${SERVER_PGID}" 2>/dev/null || true
        sleep 2
        kill -KILL "-${SERVER_PGID}" 2>/dev/null || true
    elif [[ -n "${SERVER_PID:-}" ]]; then
        # fallback: kill just the main PID
        echo "Cleaning up server PID $SERVER_PID" >&2
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi

    # Release port lock
    if [[ -n "${PORT:-}" ]]; then
        rm -rf "$LOCK_DIR/$PORT.lock" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# TODO: OPENPI_ROOT, GR00T_ROOT, should be a script which setup these models
#   ideally inside setup.sh invoking setup_pi.sh and setup_gr00t.sh
# TODO: replace this if statement by running external model script.
#   Implemented models should have own sh script.
PORT=$(choose_model_port)
echo "Using port $PORT for model server"
SERVER_PGID=""
if [ "$MODEL" == "pi0" ]; then
    if [[ -z "${OPENPI_ROOT:-}" ]]; then
        echo "OPENPI_ROOT is not set."
        echo "Set it to the OpenPi root to use, e.g.:"
        echo "  export OPENPI_ROOT=\"/path/to/the/openpi\""
        exit 1
    fi
    cd "$OPENPI_ROOT"

    # --- Singularity free version. Getting error:
    # warning: The `tool.uv.dev-dependencies` field (used in `packages/openpi-client/pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
    # error: Distribution `rerun-sdk==0.23.1 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform
    # hint: You're on Linux (`manylinux_2_28_x86_64`), but `rerun-sdk` (v0.23.1) only has wheels for the following platforms: `manylinux_2_31_aarch64`, `manylinux_2_31_x86_64`, `macosx_10_12_x86_64`, `macosx_11_0_arm64`, `win_amd64`; consider adding "sys_platform == 'linux' and platform_machine == 'x86_64'" to `tool.uv.required-environments` to ensure uv resolves to a version with compatible wheels
    # export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
    # setsid uv run scripts/serve_policy.py \
    #     --port=$PORT policy:checkpoint \
    #     --policy.config=pi0_droid_jointpos \
    #     --policy.dir="$CKPT_PATH" & SERVER_PID=$!

    # --- Singularity dependent version
    if [[ -z "${OPENPI_SIF:-}" ]]; then
        echo "OPENPI_SIF is not set."
        echo "Set it to the OpenPi singularity image to use, e.g.:"
        echo "  export OPENPI_SIF=\"/path/to/the/openpi/sif/file\""
        exit 1
    fi
    setsid $SIF_CMD exec \
        --writable-tmpfs \
        --nv \
        --bind "$(pwd):/app" \
        --bind "$XDG_CACHE_HOME":/app/.cache/xdg \
        --bind "$CKPT_PATH":/checkpoint \
        --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        --env XDG_CACHE_HOME=/app/.cache/xdg \
        "$OPENPI_SIF" uv run /app/scripts/serve_policy.py \
            --port=$PORT \
            policy:checkpoint \
            --policy.config=pi0_droid_jointpos \
            --policy.dir=/checkpoint & SERVER_PID=$!

    # capture process group of the server
    SERVER_PGID="$SERVER_PID"
elif [ "$MODEL" == "pi0_FAST" ]; then
    if [[ -z "${OPENPI_ROOT:-}" ]]; then
        echo "OPENPI_ROOT is not set."
        echo "Set it to the OpenPi root to use, e.g.:"
        echo "  export OPENPI_ROOT=\"/path/to/the/openpi\""
        exit 1
    fi
    if [[ -z "${OPENPI_SIF:-}" ]]; then
        echo "OPENPI_SIF is not set."
        echo "Set it to the OpenPi singularity image to use, e.g.:"
        echo "  export OPENPI_SIF=\"/path/to/the/openpi/sif/file\""
        exit 1
    fi
    cd "$OPENPI_ROOT"
    setsid $SIF_CMD exec \
        --writable-tmpfs \
        --nv \
        --bind "$(pwd):/app" \
        --bind "$XDG_CACHE_HOME":/app/.cache/xdg \
        --bind "$CKPT_PATH":/checkpoint \
        --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
        --env XDG_CACHE_HOME=/app/.cache/xdg \
        "$OPENPI_SIF" uv run /app/scripts/serve_policy.py \
            --port=$PORT \
            policy:checkpoint \
            --policy.config=pi0_fast_droid_jointpos \
            --policy.dir=/checkpoint & SERVER_PID=$!

    # capture process group of the server
    SERVER_PGID="$SERVER_PID"
elif [ "$MODEL" == "GR00T" ]; then
    if [[ -z "${GR00T_ROOT:-}" ]]; then
        echo "GR00T_ROOT is not set."
        echo "Set it to the GR00T root to use, e.g.:"
        echo "  export GR00T_ROOT=\"/path/to/the/GR00T\""
        exit 1
    fi
    cd "$GR00T_ROOT"
    setsid uv run scripts/serve_gr00t.py \
        --port=$PORT \
        --model_path "$CKPT_PATH"  \
        --data-config droid_joint_pos & SERVER_PID=$!

    # capture process group of the server
    SERVER_PGID="$SERVER_PID"
else
    # MODEL script must take CKPT_PATH and PORT as args
    setsid bash "$MODEL" "$CKPT_PATH" "$PORT" & SERVER_PID=$!
    # capture process group of the server
    SERVER_PGID="$SERVER_PID"
fi

# Waiting for the model server to start if it will timeout then exit

: "${MODEL_SERVER_TIMEOUT:=180}"

echo "Waiting for the model server to start (maximum: ${MODEL_SERVER_TIMEOUT}s)"
if wait_for_port "$PORT" "$MODEL_SERVER_TIMEOUT"; then
    echo "Server is listening on port ${PORT}"
else
    echo "Server did not start listening on ${PORT} within ${MODEL_SERVER_TIMEOUT}s"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "DEBUG: Server running"

# Running evaluation ---------------------------------------------------------------------------------
cd "$REALM_ROOT"
mkdir -p "$REALM_ROOT/tmp"
mkdir -p "$REALM_ROOT/logs"
mkdir -p "$REALM_ROOT/.cache/mamba"
mkdir -p "$REALM_ROOT/.cache/pip"

# TODO: evaluation part should not depend on model or ckpt_path
# If you want to use it only for naming then just a prefix should be passed

case "$EVAL_ENV" in
    singularity)
        if [[ -z "${REALM_SIF:-}" ]]; then
            echo "REALM_SIF is not set."
            echo "Set it to the Singularity image to use, e.g.:"
            echo "  export REALM_SIF=\"/path/to/the/sif/file\""
            exit 1
        fi

        $SIF_CMD exec \
            --userns \
            --nv \
            --writable-tmpfs \
            --bind "$REALM_ROOT:/app" \
            --bind "$REALM_DATA_PATH/datasets:/data" \
            --bind "$REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit" \
            --bind "$REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov" \
            --bind "$REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip" \
            --bind "$REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache" \
            --bind "$REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache" \
            --bind "$REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs" \
            --bind "$REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config" \
            --bind "$REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data" \
            --bind "$REALM_DATA_PATH/isaac-sim/documents:/root/Documents" \
            --bind "$REALM_ROOT/tmp:/tmp" \
            --env TMPDIR=/tmp \
            --env OMNIGIBSON_HEADLESS=1 \
            --env NVIDIA_DRIVER_CAPABILITIES=all \
            --env "MAMBA_CACHE_DIR=/app/.cache/mamba/${SERVER_PID}" \
            --env "PIP_CACHE_DIR=/app/.cache/pip/${SERVER_PID}" \
            "$REALM_SIF" \
            /app/scripts/karolina_evals/gen_eval_body.sh \
                "$PERTURBATION_ID" \
                "$TASK_ID" \
                "$REPEATS" \
                "$MAX_STEPS" \
                "$MODEL" \
                "$PORT"
        ;;
    docker)
        docker run \
            --gpus all \
            --privileged \
            -e OMNIGIBSON_HEADLESS=1 \
            -e OMNI_KIT_ALLOW_ROOT=1 \
            -v "$REALM_ROOT:/app:rw" \
            -v "$REALM_DATA_PATH/datasets:/data" \
            -v "$REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw" \
            -v "$REALM_DATA_PATH/isaac-sim/documents:/root/Documents:rw" \
            --network=host --rm stanfordvl/omnigibson:1.1.1 \
            /app/scripts/karolina_evals/gen_eval_body.sh \
                "$PERTURBATION_ID" \
                "$TASK_ID" \
                "$REPEATS" \
                "$MAX_STEPS" \
                "$MODEL"\
                "$PORT"
        ;;
    current)
        ./scripts/karolina_evals/gen_eval_body.sh \
            "$PERTURBATION_ID" \
            "$TASK_ID" \
            "$REPEATS" \
            "$MAX_STEPS" \
            "$MODEL" \
            "$PORT"
        ;;
esac