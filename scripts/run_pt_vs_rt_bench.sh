#!/usr/bin/env bash
# Benchmark PT vs RT render time for single-view and multi-view, headless.
# Usage: scripts/run_pt_vs_rt_bench.sh <REALM_DATA_PATH>
#
# Runs two Docker invocations (multi_view fixed at env init):
#   1) MULTI_VIEW=0 → bench_pt_vs_rt_single.json
#   2) MULTI_VIEW=1 → bench_pt_vs_rt_multi.json
# Each sweeps render modes internally (rt, r, pt:1/4/8/16/32 spp).
set -e -o pipefail

REALM_DATA_PATH="$1"
if [ -z "$REALM_DATA_PATH" ]; then
    echo "Usage: $0 <REALM_DATA_PATH>"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )
OG_LITE_DIR="$REALM_ROOT/../OG-lite"

# Image built from .docker/realm.Dockerfile (base: stanfordvl/behavior:3.9.1).
REALM_IMAGE="${REALM_IMAGE:-realm:latest}"
# OmniGibson lives at /behavior-src/OmniGibson in the 3.9.1 image (was /omnigibson-src in 1.1.1).
OG_SRC_MOUNT=/behavior-src/OmniGibson

cd "$REALM_ROOT"
mkdir -p "$REALM_DATA_PATH"/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
mkdir -p "$REALM_DATA_PATH"/isaac-sim/{logs,config,data,documents}
# OMNIGIBSON_APPDATA_PATH=/cache/appdata in the 3.9.1 image -- persist it across runs.
mkdir -p "$REALM_DATA_PATH"/cache

# EULA prompt once for the whole sweep
echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before"
echo "Omniverse Kit can start."
read -p "Do you accept the Omniverse EULA? [y/n] " yn
case $yn in [Yy]*) ;; *) exit 1;; esac

run_one () {
    local view_flag="$1"   # 0 or 1
    local label="$2"       # single|multi
    echo
    echo "─── PT/RT bench: $label-view ─────────────────────────────"
    docker run \
        --gpus all \
        --privileged \
        -e OMNIGIBSON_HEADLESS=1 \
        -e OMNI_KIT_ALLOW_ROOT=1 \
        -e TORCH_CUDA_ARCH_LIST="12.0" \
        -e CUDA_FORCE_PTX_JIT=1 \
        -e MULTI_VIEW="$view_flag" \
        -e STEPS="${STEPS:-50}" \
        -e WARMUP="${WARMUP:-5}" \
        -e TASK_ID="${TASK_ID:-0}" \
        -e SWEEP="${SWEEP:-rt,r,pt:1,pt:4,pt:8,pt:16,pt:32}" \
        -e OUTPUT_PATH="${OG_SRC_MOUNT}/bench_pt_vs_rt_${label}.json" \
        -v $(pwd):/app:rw \
        -v "$OG_LITE_DIR":${OG_SRC_MOUNT}:rw \
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
        python ${OG_SRC_MOUNT}/benchmark_pt_vs_rt.py
}

run_one 0 single
run_one 1 multi

echo
echo "Done. Results:"
echo "  $OG_LITE_DIR/bench_pt_vs_rt_single.json"
echo "  $OG_LITE_DIR/bench_pt_vs_rt_multi.json"
