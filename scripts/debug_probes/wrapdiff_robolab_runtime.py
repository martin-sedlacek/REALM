"""Read back what the RoboLab / Isaac Lab stack ACTUALLY runs, for the wrapper diff against REALM.

Everything here is a read-back, never a restatement of the cfg. The reason is concrete: RoboLab's
robolab/core/environments/base.py applies its PhysX settings through

    for attr_name, value in physx_settings.items():
        if hasattr(physx, attr_name): setattr(physx, attr_name, value)

so any key absent from the installed PhysxCfg is silently dropped. Isaac Lab 2.2's PhysxCfg has no
contact_offset, rest_offset, num_position_iterations, num_velocity_iterations, num_threads,
relaxation, warm_start or shape_collision_* field, so a third of that dict never reaches PhysX and
reading the cfg would report settings the simulation does not have.

The one that matters most: the robot's ArticulationRootPropertiesCfg asks for 64 position
iterations, but the SCENE sets max_position_iteration_count = 32, and PhysX CLAMPS the per-
articulation count to the scene's min/max window. So the question "does RoboLab solve the mimic
constraint with 64 iterations where REALM uses 32" is answered by the numbers below, not by the cfg.

Run:
    scripts/debug_probes/wrapdiff_in_isaaclab.sh scripts/debug_probes/wrapdiff_robolab_runtime.py \
        --out /logs/gripper_squeeze/wrapdiff_robolab_runtime.json

Isaac exits non-zero at teardown; grep for WRAPDIFF_ROBOLAB_OK, never the exit code.
"""
import argparse
import json
import os

import cv2  # noqa: F401  must be imported before isaaclab

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/logs/gripper_squeeze/wrapdiff_robolab_runtime.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
# A bare SimulationApp({"headless": True}) hangs in this container; AppLauncher is the supported
# entry point and is what every RoboLab example uses.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402

from robolab.core.environments.base import RobolabDefaultEnvCfg  # noqa: E402
from robolab.robots.droid import DroidCfg  # noqa: E402

OUT = {}


def np_(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        try:
            import warp as wp
            x = wp.to_torch(x)
        except Exception:
            return None
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


# ------------------------------------------------------------------ RoboLab's real sim cfg
# Instantiating the env cfg runs __post_init__, which is where dt / render_interval / use_fabric and
# the physx_settings block are applied. Taking .sim off the instance therefore gives the SAME
# SimulationCfg a RoboLab task would run, without loading any scene assets.
env_cfg = RobolabDefaultEnvCfg()
sim_cfg = env_cfg.sim
sim_cfg.device = args_cli.device or "cuda:0"

hdr("ROBOLAB SimulationCfg AS CONFIGURED (after __post_init__)")
cfgd = {}
for k in ("dt", "render_interval", "use_fabric", "device", "gravity", "physics_prim_path",
          "enable_scene_query_support", "create_stage_in_memory"):
    v = getattr(sim_cfg, k, "<absent>")
    cfgd[k] = str(v)
    print(f"  sim.{k:<32} = {v}")
physxd = {}
for k in sorted(vars(sim_cfg.physx)):
    if k.startswith("_"):
        continue
    physxd[k] = str(getattr(sim_cfg.physx, k))
    print(f"  sim.physx.{k:<26} = {getattr(sim_cfg.physx, k)}")
print("\n  sim.physics_material (the DEFAULT material every shape without its own falls back to):")
matd = {}
for k in sorted(vars(sim_cfg.physics_material)):
    if k.startswith("_") or k == "func":
        continue
    matd[k] = str(getattr(sim_cfg.physics_material, k))
    print(f"    {k:<34} = {getattr(sim_cfg.physics_material, k)}")
print("\n  keys RoboLab's physx_settings dict tries to set that this PhysxCfg DOES NOT HAVE\n"
      "  (silently dropped by its `if hasattr(physx, attr_name)` guard):")
tried = ["gpu_temp_buffer_capacity", "gpu_heap_capacity", "gpu_collision_stack_size", "enable_ccd",
         "contact_offset", "rest_offset", "num_position_iterations", "num_velocity_iterations",
         "max_position_iteration_count", "max_velocity_iteration_count",
         "bounce_threshold_velocity", "max_depenetration_velocity", "solver_type", "num_threads",
         "relaxation", "warm_start", "shape_collision_distance", "shape_collision_margin"]
dropped = [k for k in tried if not hasattr(sim_cfg.physx, k)]
applied = [k for k in tried if hasattr(sim_cfg.physx, k)]
print(f"    DROPPED ({len(dropped)}): {dropped}")
print(f"    APPLIED ({len(applied)}): {applied}")
OUT["sim_cfg"] = cfgd
OUT["physx_cfg"] = physxd
OUT["default_physics_material_cfg"] = matd
OUT["physx_settings_dropped"] = dropped
OUT["physx_settings_applied"] = applied

# ------------------------------------------------------------------ bring up the sim + robot
sim = SimulationContext(sim_cfg)
robot_cfg = DroidCfg().robot.replace(prim_path="/World/envs/env_0/robot")
robot = Articulation(robot_cfg)
sim.reset()
stage = sim.stage if hasattr(sim, "stage") else Usd.Stage.GetCurrent()
if stage is None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()

hdr("PHYSICS SCENE AS THE SIMULATION ACTUALLY HAS IT (read back from the live prim)")
scene_attrs = {}
for prim in stage.Traverse():
    if str(prim.GetTypeName()) != "PhysicsScene":
        continue
    print(f"  {prim.GetPath()}   schemas={[str(s) for s in prim.GetAppliedSchemas()]}")
    for a in sorted(prim.GetAttributes(), key=lambda x: x.GetName()):
        n = a.GetName()
        if not (n.startswith("physxScene:") or n.startswith("physics:")):
            continue
        try:
            v = a.Get()
        except Exception:
            continue
        if v is None:
            continue
        scene_attrs[n] = str(v)
        print(f"    {n:<58} = {v}")
OUT["physics_scene_live"] = scene_attrs

pc = sim.get_physics_context()
ctx = {}
for name in ("is_gpu_dynamics_enabled", "get_broadphase_type", "get_solver_type",
             "is_ccd_enabled", "get_bounce_threshold", "get_friction_offset_threshold",
             "get_friction_correlation_distance", "get_physics_dt", "get_gravity",
             "is_fabric_enabled", "is_stablization_enabled", "is_stabilization_enabled",
             "get_gpu_max_rigid_contact_count", "get_invert_collision_group_filter"):
    fn = getattr(pc, name, None)
    if fn is None:
        continue
    try:
        ctx[name] = str(fn())
    except Exception as e:
        ctx[name] = f"<{type(e).__name__}>"
print("\n  PhysicsContext read-back (this is the GPU-vs-CPU and TGS-vs-PGS answer):")
for k, v in ctx.items():
    print(f"    {k:<42} = {v}")
OUT["physics_context"] = ctx

# ------------------------------------------------------------------ articulation solver counts
hdr("ARTICULATION SOLVER ITERATION COUNTS -- requested vs what the scene window allows")
art_path = "/World/envs/env_0/robot"
art_prim = stage.GetPrimAtPath(art_path)
art = {}
if art_prim and art_prim.IsValid():
    api = PhysxSchema.PhysxArticulationAPI(art_prim)
    for nm, getter in (("solverPositionIterationCount", "GetSolverPositionIterationCountAttr"),
                       ("solverVelocityIterationCount", "GetSolverVelocityIterationCountAttr"),
                       ("enabledSelfCollisions", "GetEnabledSelfCollisionsAttr"),
                       ("sleepThreshold", "GetSleepThresholdAttr"),
                       ("stabilizationThreshold", "GetStabilizationThresholdAttr")):
        try:
            art[nm] = str(getattr(api, getter)().Get())
        except Exception as e:
            art[nm] = f"<{type(e).__name__}>"
for k, v in art.items():
    print(f"  articulation {k:<34} = {v}")
print(f"\n  scene window: min_position={scene_attrs.get('physxScene:minPositionIterationCount')} "
      f"max_position={scene_attrs.get('physxScene:maxPositionIterationCount')}  "
      f"min_velocity={scene_attrs.get('physxScene:minVelocityIterationCount')} "
      f"max_velocity={scene_attrs.get('physxScene:maxVelocityIterationCount')}")
print("  PhysX clamps the per-articulation count into that window, so the EFFECTIVE position\n"
      "  iteration count is min(articulation_request, scene_max) -- compare that number, not 64,\n"
      "  against OmniGibson's 32.")
OUT["articulation"] = art

# ------------------------------------------------------------------ gripper joints
hdr("GRIPPER JOINTS AS THE SIMULATION HAS THEM")
names = list(robot.data.joint_names)
st = np_(robot.data.joint_stiffness)
dp = np_(robot.data.joint_damping)
el = np_(getattr(robot.data, "joint_effort_limits", None))
vl = np_(getattr(robot.data, "joint_velocity_limits", None))
am = np_(getattr(robot.data, "joint_armature", None))
fr = np_(getattr(robot.data, "joint_friction", None))
pl = np_(getattr(robot.data, "joint_pos_limits", None))
joints = {}
print(f"  {'joint':<34} {'stiff':>12} {'damp':>12} {'effort':>9} {'maxvel':>8} "
      f"{'armature':>9} {'friction':>9}  limits")
for i, n in enumerate(names):
    row = dict(
        stiffness=None if st is None else float(st[0, i]),
        damping=None if dp is None else float(dp[0, i]),
        effort_limit=None if el is None else float(el[0, i]),
        velocity_limit=None if vl is None else float(vl[0, i]),
        armature=None if am is None else float(am[0, i]),
        friction=None if fr is None else float(fr[0, i]),
        pos_limits=None if pl is None else [float(pl[0, i, 0]), float(pl[0, i, 1])],
    )
    joints[n] = row
    print(f"  {n:<34} {row['stiffness']!s:>12.12} {row['damping']!s:>12.12} "
          f"{row['effort_limit']!s:>9.9} {row['velocity_limit']!s:>8.8} "
          f"{row['armature']!s:>9.9} {row['friction']!s:>9.9}  {row['pos_limits']}")
OUT["joints_runtime"] = joints

hdr("GRIPPER JOINT PRIMS -- live mimic attributes and drive block")
jp = {}
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith(art_path) or not prim.IsA(UsdPhysics.Joint):
        continue
    nm = p.split("/")[-1]
    if not any(k in nm for k in ("finger", "knuckle")):
        continue
    d = dict(type=str(prim.GetTypeName()),
             schemas=[str(s) for s in prim.GetAppliedSchemas()], attrs={}, mimic={})
    for a in sorted(prim.GetAttributes(), key=lambda x: x.GetName()):
        an = a.GetName()
        if not (an.startswith("drive:") or an.startswith("physxJoint:")
                or an.startswith("physxLimit:") or "imicJoint" in an
                or an.startswith("physics:")):
            continue
        try:
            v = a.Get()
        except Exception:
            continue
        if v is None and not a.HasAuthoredValue():
            continue
        d["attrs"][an] = str(v)
    for r in prim.GetRelationships():
        if "imicJoint" in r.GetName():
            d["mimic"][r.GetName()] = [str(t) for t in r.GetTargets()]
    jp[nm] = d
    print(f"\n  {nm}  ({d['type']})")
    print(f"    schemas: {[s for s in d['schemas'] if 'Mimic' in s or 'Drive' in s]}")
    for k, v in d["attrs"].items():
        print(f"    {k:<52} = {v}")
    for k, v in d["mimic"].items():
        print(f"    REL {k:<48} -> {v}")
OUT["joint_prims_live"] = jp

hdr("PAD COLLISION GEOMS -- offsets, approximation, and the BOUND PHYSICS MATERIAL")
print("  (REALM's geom_prim.get_applied_physics_material() returns nothing for either pad; this is\n"
      "   the same question asked on the Isaac Lab side.)")
pads = {}
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if "inner_finger" not in p or not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    e = {}
    for an in ("physxCollision:contactOffset", "physxCollision:restOffset",
               "physxMeshCollision:approximation", "physics:approximation",
               "physxSDFMeshCollision:sdfResolution"):
        a = prim.GetAttribute(an)
        e[an] = None if not (a and a.IsValid()) else str(a.Get())
    try:
        mb = UsdShade.MaterialBindingAPI(prim)
        e["bound_physics_material"] = str(mb.GetDirectBinding("physics").GetMaterialPath())
        e["all_collection_bindings"] = [str(b.GetMaterialPath())
                                        for b in mb.GetCollectionBindings("physics")]
    except Exception as ex:
        e["bound_physics_material"] = f"<{type(ex).__name__}>"
    pads[p] = e
    print(f"\n  {p}")
    for k, v in e.items():
        print(f"    {k:<42} = {v}")
OUT["pads"] = pads

hdr("PHYSICS MATERIALS PRESENT ON THE STAGE")
mats = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.MaterialAPI) and not any(
            "PhysicsMaterialAPI" in str(s) for s in prim.GetAppliedSchemas()):
        continue
    e = {}
    for a in prim.GetAttributes():
        an = a.GetName()
        if an.startswith("physics:") or an.startswith("physxMaterial:"):
            try:
                e[an] = str(a.Get())
            except Exception:
                pass
    mats[str(prim.GetPath())] = e
    print(f"  {prim.GetPath()}\n    {e}")
OUT["materials"] = mats

hdr("BODY MASSES / INERTIAS ON THE GRIPPER LINKS")
print("  PhysX turns a mimic joint's naturalFrequency into an absolute constraint stiffness using\n"
      "  the articulation's effective inertia, so identical nf on different inertias is a different\n"
      "  spring. These must match REALM's converted asset or nf is not comparable at all.")
bodies = {}
bn = list(robot.data.body_names)
masses = np_(getattr(robot.root_physx_view, "get_masses", lambda: None)())
inertias = np_(getattr(robot.root_physx_view, "get_inertias", lambda: None)())
for i, n in enumerate(bn):
    if not any(k in n for k in ("finger", "knuckle", "base_link")):
        continue
    m = None if masses is None else float(masses[0, i])
    inr = None if inertias is None else [float(x) for x in np.asarray(inertias[0, i]).reshape(-1)]
    bodies[n] = dict(mass=m, inertia=inr)
    diag = None if inr is None else [inr[0], inr[4], inr[8]] if len(inr) == 9 else inr
    print(f"  {n:<30} mass={m}  inertia_diag={diag}")
OUT["bodies_runtime"] = bodies

os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
with open(args_cli.out, "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True, default=str)
print(f"\nwrote {args_cli.out}")
print("WRAPDIFF_ROBOLAB_OK")
simulation_app.close()
