"""Global simulator and renderer configuration.

Two separate concerns that both have to happen before/around env creation:
  set_sim_config()     -- OmniGibson macros (frequencies, object states, seeding). Must run
                          BEFORE og.Environment() is constructed, since the macros are read there.
  set_rendering_mode() -- carb RTX settings. Applied after the env exists.
"""
import os
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
    # OG-lite-only macro: folds each physics substep into (R, C) accumulators instead of
    # materializing an (N, R, C) stack per step. Stock OmniGibson never reads it, so setting it in
    # the stock container is a harmless no-op. Off by default until the win is measured on REALM's
    # own workload -- export REALM_INCREMENTAL_CONTACT_CACHE=1 to turn it on.
    gm.INCREMENTAL_CONTACT_CACHE = os.environ.get("REALM_INCREMENTAL_CONTACT_CACHE", "0") == "1"
    # Same deal: OG-lite drops bodies further than PROXIMITY_GATE_RADIUS from every robot out of the
    # contact matrix. It defaults ON in OG-lite; set REALM_PROXIMITY_GATE=0 to rule it out if
    # contact-dependent metrics (collisions_env, is_grasping) start behaving oddly.
    if "REALM_PROXIMITY_GATE" in os.environ:
        gm.PROXIMITY_GATE_ENABLED = os.environ["REALM_PROXIMITY_GATE"] == "1"
    # Which device runs rigid-body physics. OmniGibson defaults this to False, i.e. the CPU solver
    # with the MBP broadphase; True switches PhysX to GPU dynamics with the GPU broadphase
    # (simulator.py: enable_gpu_dynamics / set_broadphase_type). Unlike the two macros above this is
    # NOT an OG-lite addition -- it is stock 3.9.1 and works in either container.
    #
    # Two things to know before using it:
    #   - it changes the solver, so trajectories are not bit-identical to a CPU run and results are
    #     not directly comparable to CPU-collected baselines;
    #   - the GPU path is bounded by gm.GPU_*_CAPACITY macros. If a scene exceeds them PhysX warns
    #     and can drop contacts rather than failing, so check the log for capacity messages before
    #     trusting collision-dependent metrics.
    # Off by default; export REALM_GPU_DYNAMICS=1 to switch.
    if "REALM_GPU_DYNAMICS" in os.environ:
        gm.USE_GPU_DYNAMICS = os.environ["REALM_GPU_DYNAMICS"] == "1"
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
