
import os
import random

import numpy as np
import torch

import omnigibson.lazy as lazy
from omnigibson.macros import gm


def set_sim_config(robot="DROID"):
    if robot == "WidowX":
        gm.DEFAULT_SIM_STEP_FREQ = 5
        gm.DEFAULT_RENDERING_FREQ = 5
    elif "UR5" in robot:
        gm.DEFAULT_SIM_STEP_FREQ = 30
        gm.DEFAULT_RENDERING_FREQ = 30
    else:
        gm.DEFAULT_SIM_STEP_FREQ = 15
        gm.DEFAULT_RENDERING_FREQ = 15

    gm.DEFAULT_PHYSICS_FREQ = 120
    # Transition rules trigger an upstream sludge-state collision bug.
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = True
    # ToggledOn is the only state REALM updates each frame; kinematic states remain on demand.
    gm.OBJECT_STATE_UPDATE_WHITELIST = ["ToggledOn"]
    gm.ENABLE_VISUAL_UPDATES = False
    gm.INCREMENTAL_CONTACT_CACHE = os.environ.get("REALM_INCREMENTAL_CONTACT_CACHE", "1") == "1"
    # Proximity-gate membership is fixed at initialization; disable it for mobile robots.
    gm.PROXIMITY_GATE_ENABLED = os.environ.get("REALM_PROXIMITY_GATE", "1") == "1"
    # GPU dynamics changes trajectories and is therefore opt-in.
    if "REALM_GPU_DYNAMICS" in os.environ:
        gm.USE_GPU_DYNAMICS = os.environ["REALM_GPU_DYNAMICS"] == "1"
    gm.RENDER_VIEWER_CAMERA=False
    # OmniGibson requires at least 60 Hz for HQ isosurface rendering.
    gm.ENABLE_HQ_RENDERING = False

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_rendering_mode(rendering_mode):
    carb_settings = lazy.carb.settings.get_settings()
    if rendering_mode == "pt":
        def enable_interactive_path_tracing(carb_settings, samples_per_pixel=8):
            carb_settings.set("/rtx/rendermode", "PathTracing")
            if samples_per_pixel is not None:
                carb_settings.set_int("/rtx/pathtracing/spp", samples_per_pixel)
                carb_settings.set_int("/rtx/pathtracing/totalSpp", samples_per_pixel)
                carb_settings.set_int(
                    "/rtx/pathtracing/useDirectLightingCache", False
                )
            carb_settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)

        enable_interactive_path_tracing(carb_settings, samples_per_pixel=8)
    elif rendering_mode == "r":
        carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")
        carb_settings.set_bool("/rtx/translucency/enabled", True)
        carb_settings.set_bool("/rtx/reflections/enabled", False)
        carb_settings.set_bool("/rtx/indirectDiffuse/enabled", False)
        carb_settings.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
        carb_settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 1)
        carb_settings.set_bool("/rtx/shadows/enabled", False)
        carb_settings.set_int("/rtx/post/dlss/execMode", 0)
        carb_settings.set_bool("/rtx/ambientOcclusion/enabled", False)
        carb_settings.set_bool("/rtx-transient/dlssg/enabled", False)
        carb_settings.set_float("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.6)
        carb_settings.set_float("/rtx/sceneDb/ambientLightIntensity", 1.0)
        carb_settings.set_bool("/exts/omni.renderer.core/present/enabled", False)
        carb_settings.set_string("/isaaclab/rendering/rendering_mode", "performance")
    else:
        assert rendering_mode == "rt", f"rendering mode must be 'pt', 'rt', or 'r'"
