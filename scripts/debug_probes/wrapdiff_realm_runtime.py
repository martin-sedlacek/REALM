"""Read back what the REALM / OmniGibson stack ACTUALLY runs -- the mirror of
wrapdiff_robolab_runtime.py, printing the SAME fields in the SAME order so the two logs diff.

Everything is a read-back off the live scene after play, never a restatement of a config, because
the whole question is what the wrapper leaves behind rather than what it was asked for.

Answers, on the REALM side:
  * PhysicsScene: enableGPUDynamics / broadphaseType / solverType / timeStepsPerSecond and the
    min/max solver iteration window. OmniGibson defaults gm.USE_GPU_DYNAMICS=False, which selects
    the CPU articulation solver and MBP broadphase; Isaac Lab runs device=cuda:0, so the RoboLab
    side reads enableGPUDynamics=True / broadphaseType=GPU. That is the largest scene-level
    difference left once dt, solver type and position-iteration count are shown to match.
  * the ROBOT articulation's solver iteration counts, against that window
  * per gripper joint: gains, effort limit, velocity limit, armature, friction, limits, and the live
    mimic attributes read by literal token name
  * the two PAD links' collision geoms: approximation, contact/rest offset, and the applied physics
    material through BOTH geom_prim.get_applied_physics_material() and a direct USD binding read --
    the former returning nothing is not by itself evidence of an unbound material
  * every physics material on the stage, and the gripper links' masses / inertias

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/wrapdiff_realm_runtime.py

Isaac exits 139 at teardown; grep for WRAPDIFF_REALM_OK, never the exit code.
"""
import argparse
import json
import os

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=220)

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze/wrapdiff_realm_runtime.json")
args = ap.parse_args()

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

OUT = {}


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def jget(j, attr):
    try:
        v = getattr(j, attr)
        if isinstance(v, bool) or v is None:
            return v
        return float(_np(v))
    except Exception:
        return None


print(f"[wrapdiff-realm] robot={args.robot} task={args.task_cfg}", flush=True)
set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=True, rendering_mode="rt", robot=args.robot,
)
obs, _ = env.reset()
robot = env.robot
stage = og.sim.stage

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

pc = og.sim._physics_context
ctx = {}
for name in ("is_gpu_dynamics_enabled", "get_broadphase_type", "get_solver_type",
             "is_ccd_enabled", "get_bounce_threshold", "get_friction_offset_threshold",
             "get_friction_correlation_distance", "get_physics_dt", "get_gravity",
             "is_stablization_enabled", "get_gpu_max_rigid_contact_count",
             "get_invert_collision_group_filter"):
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
ctx["og_physics_dt"] = str(og.sim.get_physics_dt())
ctx["og_rendering_dt"] = str(og.sim.get_rendering_dt())
ctx["og_sim_step_dt"] = str(getattr(og.sim, "_sim_step_dt", None))
ctx["gm_USE_GPU_DYNAMICS"] = str(og.gm.USE_GPU_DYNAMICS)
ctx["gm_ENABLE_CCD"] = str(og.gm.ENABLE_CCD)
ctx["gm_DEFAULT_PHYSICS_FREQ"] = str(og.gm.DEFAULT_PHYSICS_FREQ)
ctx["gm_DEFAULT_RENDERING_FREQ"] = str(og.gm.DEFAULT_RENDERING_FREQ)
print("\n  OmniGibson dt / macro read-back (control decimation = sim_step_dt / physics_dt):")
for k in ("og_physics_dt", "og_rendering_dt", "og_sim_step_dt", "gm_USE_GPU_DYNAMICS",
          "gm_ENABLE_CCD", "gm_DEFAULT_PHYSICS_FREQ", "gm_DEFAULT_RENDERING_FREQ"):
    print(f"    {k:<42} = {ctx[k]}")
OUT["physics_context"] = ctx

hdr("ARTICULATION SOLVER ITERATION COUNTS -- requested vs what the scene window allows")
art = dict(solverPositionIterationCount=str(robot.solver_position_iteration_count),
           solverVelocityIterationCount=str(robot.solver_velocity_iteration_count))
art_prim = robot.prim
for nm, getter in (("enabledSelfCollisions", "GetEnabledSelfCollisionsAttr"),
                   ("sleepThreshold", "GetSleepThresholdAttr"),
                   ("stabilizationThreshold", "GetStabilizationThresholdAttr")):
    try:
        api = lazy.pxr.PhysxSchema.PhysxArticulationAPI(art_prim)
        art[nm] = str(getattr(api, getter)().Get())
    except Exception as e:
        art[nm] = f"<{type(e).__name__}>"
for k, v in art.items():
    print(f"  articulation {k:<34} = {v}")
print(f"\n  scene window: min_position={scene_attrs.get('physxScene:minPositionIterationCount')} "
      f"max_position={scene_attrs.get('physxScene:maxPositionIterationCount')}  "
      f"min_velocity={scene_attrs.get('physxScene:minVelocityIterationCount')} "
      f"max_velocity={scene_attrs.get('physxScene:maxVelocityIterationCount')}")
print("  physxScene:maxPositionIterationCount is documented as overriding actors that request MORE,\n"
      "  so the effective count is min(request, scene_max) on BOTH stacks -- compare those.")
OUT["articulation"] = art

hdr("GRIPPER JOINTS AS THE SIMULATION HAS THEM")
q_all = _np(robot.get_joint_positions())
joint_names = [None] * len(q_all)
for n, j in robot.joints.items():
    idxs = list(j.dof_indices)
    if len(idxs) == 1:
        joint_names[idxs[0]] = n
arm = list(robot.arm_joint_names[robot.default_arm])
joints = {}
print(f"  {'joint':<34} {'stiff':>14} {'damp':>14} {'effort':>12} {'maxvel':>10} "
      f"{'armature':>9} {'friction':>9}  limits")
for i, n in enumerate(joint_names):
    if n is None:
        continue
    j = robot.joints[n]
    row = dict(stiffness=jget(j, "stiffness"), damping=jget(j, "damping"),
               effort_limit=jget(j, "max_effort"), velocity_limit=jget(j, "max_velocity"),
               armature=jget(j, "armature"), friction=jget(j, "friction"),
               pos_limits=[jget(j, "lower_limit"), jget(j, "upper_limit")],
               is_mimic=jget(j, "is_mimic_joint"), driven=jget(j, "driven"),
               control_type=str(getattr(j, "control_type", None)), is_arm=n in arm)
    joints[n] = row
    print(f"  {n:<34} {row['stiffness']!s:>14.14} {row['damping']!s:>14.14} "
          f"{row['effort_limit']!s:>12.12} {row['velocity_limit']!s:>10.10} "
          f"{row['armature']!s:>9.9} {row['friction']!s:>9.9}  {row['pos_limits']}")
OUT["joints_runtime"] = joints

hdr("GRIPPER JOINT PRIMS -- live mimic attributes and drive block")
jp = {}
for n in joint_names:
    if n is None or n in arm or n == "rootJoint":
        continue
    prim = robot.joints[n].prim
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
    jp[n] = d
    print(f"\n  {n}  ({d['type']})")
    print(f"    schemas: {[s for s in d['schemas'] if 'Mimic' in s or 'Drive' in s]}")
    for k, v in d["attrs"].items():
        if k.startswith("physics:local"):
            continue
        print(f"    {k:<52} = {v}")
    for k, v in d["mimic"].items():
        print(f"    REL {k:<48} -> {v}")
OUT["joint_prims_live"] = jp

hdr("PAD COLLISION GEOMS -- offsets, approximation, and the APPLIED PHYSICS MATERIAL")
print("  get_applied_physics_material() returning None is NOT by itself an unbound material: the\n"
      "  RoboLab side reads an empty direct binding on these same geoms too, and both assets fall\n"
      "  back to the scene default material. Both readings are printed so they can be compared.")
FL = list(robot.finger_link_names[robot.default_arm])
pads = {}
for ln in FL:
    lk = robot.links[ln]
    for gname, geom in lk.collision_meshes.items():
        prim = geom.prim
        e = dict(link=ln, geom=gname)
        for an in ("physxCollision:contactOffset", "physxCollision:restOffset",
                   "physxMeshCollision:approximation", "physics:approximation",
                   "physxSDFMeshCollision:sdfResolution"):
            a = prim.GetAttribute(an)
            e[an] = None if not (a and a.IsValid()) else str(a.Get())
        try:
            m = geom.get_applied_physics_material()
            e["get_applied_physics_material"] = None if m is None else str(m.prim_path)
        except Exception as ex:
            e["get_applied_physics_material"] = f"<{type(ex).__name__}: {ex}>"
        try:
            mb = lazy.pxr.UsdShade.MaterialBindingAPI(prim)
            e["usd_direct_binding"] = str(mb.GetDirectBinding("physics").GetMaterialPath())
            e["usd_collection_bindings"] = [str(b.GetMaterialPath())
                                            for b in mb.GetCollectionBindings("physics")]
        except Exception as ex:
            e["usd_direct_binding"] = f"<{type(ex).__name__}>"
        pads[str(prim.GetPath())] = e
        print(f"\n  {prim.GetPath()}")
        for k, v in e.items():
            print(f"    {k:<42} = {v}")
OUT["pads"] = pads

hdr("PHYSICS MATERIALS PRESENT ON THE STAGE (robot subtree + scene default)")
mats = {}
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not any(("PhysicsMaterialAPI" in str(s)) for s in prim.GetAppliedSchemas()):
        continue
    if not (p.startswith(robot.prim_path) or "physicsScene" in p or "defaultMaterial" in p):
        continue
    e = {}
    for a in prim.GetAttributes():
        an = a.GetName()
        if an.startswith("physics:") or an.startswith("physxMaterial:"):
            try:
                e[an] = str(a.Get())
            except Exception:
                pass
    mats[p] = e
    print(f"  {p}\n    {e}")
OUT["materials"] = mats

hdr("BODY MASSES / INERTIAS ON THE GRIPPER LINKS")
print("  PhysX turns a mimic joint's naturalFrequency into an absolute constraint stiffness using\n"
      "  the articulation's effective inertia, so nf=1000 on different inertias is a different\n"
      "  spring. These must match the RoboLab asset or nf is not comparable between the stacks.")
bodies = {}
for ln, lk in robot.links.items():
    if not any(k in ln for k in ("finger", "knuckle", "base_link")):
        continue
    try:
        e = dict(mass=float(lk.mass), density=float(lk.density),
                 inertia=[float(x) for x in np.asarray(_np(lk.inertia)).reshape(-1)],
                 com=[float(x) for x in np.asarray(_np(lk.center_of_mass[0])).reshape(-1)])
    except Exception as ex:
        e = dict(error=f"{type(ex).__name__}: {ex}")
    bodies[ln] = e
    print(f"  {ln:<32} mass={e.get('mass')} inertia={e.get('inertia')}")
OUT["bodies_runtime"] = bodies

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True, default=str)
print(f"\nwrote {args.out}")
print("WRAPDIFF_REALM_OK")
