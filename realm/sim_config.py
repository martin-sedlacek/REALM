"""Global simulator and renderer configuration.

Two separate concerns that both have to happen before/around env creation:
  set_sim_config()     -- OmniGibson macros (frequencies, object states, seeding). Must run
                          BEFORE og.Environment() is constructed, since the macros are read there.
  set_rendering_mode() -- carb RTX settings. Applied after the env exists.
"""
import random

import numpy as np
import torch

import omnigibson.lazy as lazy
from omnigibson.macros import gm


def set_sim_config(robot="DROID"):
    if robot == "WidowX": # TODO: just read this from the yamls...
        gm.DEFAULT_SIM_STEP_FREQ = 5
        gm.DEFAULT_RENDERING_FREQ = 5
    elif "UR5" in robot:
        gm.DEFAULT_SIM_STEP_FREQ = 30
        gm.DEFAULT_RENDERING_FREQ = 30
    else:
        gm.DEFAULT_SIM_STEP_FREQ = 15
        gm.DEFAULT_RENDERING_FREQ = 15

    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state
    # Of the 13 state types OmniGibson steps every frame, ToggledOn is the only one REALM reads. The
    # rest are pure overhead in a kitchen scene, and several are expensive: HeatSourceOrSink issues a
    # PhysX overlap query per heat source per step (stove/oven/microwave/fridge), AttachedTo does a
    # full-row contact scan per attachable object, and Temperature/MaxTemperature run a tensorized
    # update over every object in the scene.
    #
    # Safe because Touching / OnTop / Inside -- the states REALM actually queries -- are computed on
    # demand via KinematicsMixin and never appear in the per-step update list. Every state type is
    # still globally initialized, so on-demand queries are unaffected.
    #
    # The one capability this removes is water: ParticleSource/ParticleSink stop producing, so a
    # faucet would no longer run. No REALM task uses one.
    gm.OBJECT_STATE_UPDATE_WHITELIST = ["ToggledOn"]
    # Texture/emitter updates for Cooked, Burnt, Frozen, OnFire etc. Nothing REALM renders depends on
    # them, and the sweep touches every initialized object in the scene each step.
    gm.ENABLE_VISUAL_UPDATES = False
    gm.RENDER_VIEWER_CAMERA=False
    # OG 3.9.1 asserts that isosurface HQ rendering runs at >=60 FPS, but REALM renders at 5-30 Hz
    # (see above), so enabling it aborts at env creation. Disabled unconditionally until the
    # rendering frequency is raised to 60.
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

        #carb_settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
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
