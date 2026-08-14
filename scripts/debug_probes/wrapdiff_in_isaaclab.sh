#!/bin/bash
# Run a python script inside the Isaac Lab 2.2.0 SIF (Isaac Sim 5.0) with both robot asset trees
# and the REALM log dir bound, so the RoboLab and REALM sides of the wrapper diff can be measured
# with the SAME reader. Mirrors rl_run_client_in_sif() from RoboLab's scripts/lib/robolab_eval_lib.sh
# (identical binds and env) and adds: the REALM worktree at /realm, the shared log tree at /logs.
set -uo pipefail
ROBOLAB_ROOT=/mnt/home_lustre/sedlam56/projects/RoboLab
ISAAC_SIF=/mnt/home_lustre/sedlam56/apptainer/isaac-lab-2.2.0.sif
ISAAC_STATE=${ROBOLAB_ROOT}/.isaac_state
SIF_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
REALM_WT=${REALM_WT:-/mnt/home_lustre/sedlam56/projects/wt/realm_r}
REALM_LOGS=/mnt/home_lustre/sedlam56/projects/REALM/logs
mkdir -p "${ISAAC_STATE}"/{kit_cache,kit_data,ov_cache,ov_data,ov_logs,home_documents}
cd "${ROBOLAB_ROOT}"
exec apptainer exec --nv --writable-tmpfs \
  --bind "${ROBOLAB_ROOT}:${ROBOLAB_ROOT}" \
  --bind "${REALM_WT}:/realm" \
  --bind "${REALM_LOGS}:/logs" \
  --bind "${ISAAC_STATE}/kit_cache:/workspace/isaaclab/_isaac_sim/kit/cache" \
  --bind "${ISAAC_STATE}/kit_data:/workspace/isaaclab/_isaac_sim/kit/data" \
  --bind "${ISAAC_STATE}/ov_cache:/root/.cache/ov" \
  --bind "${ISAAC_STATE}/ov_data:/root/.local/share/ov/data" \
  --bind "${ISAAC_STATE}/ov_logs:/root/.nvidia-omniverse/logs" \
  --bind "${ISAAC_STATE}/home_documents:/root/Documents" \
  --env "PYTHONUSERBASE=${ROBOLAB_ROOT}/.robolab_user_site" \
  --env "PYTHONPATH=${ROBOLAB_ROOT}" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  --env "OMNI_KIT_ACCEPT_EULA=YES" \
  "${ISAAC_SIF}" "${SIF_PYTHON}" "$@"
