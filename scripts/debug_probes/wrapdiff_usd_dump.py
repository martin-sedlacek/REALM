"""Dump every physics-relevant authored attribute of a Robotiq 2F-85 gripper USD, as JSON.

Purpose: the USD-level half of the RoboLab-vs-REALM wrapper diff. Both assets are read in ONE
process with the SAME pxr build, so any difference reported is a difference in the FILE rather than
in the reader:

    ./rr python -u scripts/debug_probes/wrapdiff_usd_dump.py A.usd B.usd --out /logs/.../wrapdiff_usd

Writes <out>_<basename>.json per input. Covers, for every joint / body / collision geom:
  * joint type, body0/body1, axis, limits, and every physics:*/physxJoint:*/physxLimit:* attribute
  * DriveAPI per axis token (type, stiffness, damping, maxForce, target) -- present or absent
  * PhysxMimicJointAPI instances and every authored field INCLUDING naturalFrequency /
    dampingRatio, read via GetAttribute() by literal name. Those two are absent from the registered
    PhysxMimicJointAPI schema in both containers, but omni.physx reads them as custom attributes by
    token, so they must be read by name and not through the schema wrapper.
  * PhysxCollisionAPI contactOffset / restOffset and physxMeshCollision:approximation -- the pad
    contact geometry, which REALM rewrites at runtime in env_dynamic.update_robot_physics()
  * mass, density, diagonal inertia, centre of mass, principal axes. These matter because PhysX
    turns a mimic joint's naturalFrequency into an absolute constraint stiffness using the
    articulation's effective inertia, so identical nf on different inertias is different stiffness.
  * physics material bindings and every physxMaterial:* attribute (friction / restitution /
    compliantContactStiffness)
  * articulation root APIs and their authored solver iteration counts

Isaac exits 139 at teardown; grep for WRAPDIFF_USD_DUMP_OK, never the exit code.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("usd", nargs="+")
ap.add_argument("--out", default=None, help="written per input as <out>_<basename>.json")
ap.add_argument("--subtree", default=None,
                help="only dump prims under this path (default: whole stage)")
args = ap.parse_args()

# pxr is only importable after Kit has bootstrapped, so launch OmniGibson first -- exactly what
# scripts/convert_robolab_gripper_usd.py does. `import isaacsim` alone is NOT enough: the pip
# isaacsim keeps pxr in an extscache dir that only the Kit bootstrap puts on sys.path, and a bare
# `from pxr import Usd` there fails with ModuleNotFoundError.
import omnigibson as og  # noqa: E402

og.launch()

import omnigibson.lazy as lazy  # noqa: E402

Usd, UsdGeom, UsdPhysics, UsdShade, Sdf = (
    lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics, lazy.pxr.UsdShade, lazy.pxr.Sdf
)


def val(v):
    """JSON-safe conversion of a USD attribute value."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Sdf.AssetPath):
        return v.path
    try:                                     # Gf vectors / quats and Vt arrays all iterate
        return [val(x) for x in v]
    except TypeError:
        return str(v)


def attrs(prim, prefixes=None):
    """Every AUTHORED attribute on prim, optionally filtered to the given name prefixes."""
    out = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if prefixes and not any(n.startswith(p) for p in prefixes):
            continue
        if not a.HasAuthoredValue():
            continue
        out[n] = val(a.Get())
    return out


def rels(prim):
    out = {}
    for r in prim.GetRelationships():
        t = r.GetTargets()
        if t:
            out[r.GetName()] = [str(x) for x in t]
    return out


def dump(path_usd):
    stage = Usd.Stage.Open(path_usd)
    assert stage is not None, f"could not open {path_usd}"
    report = dict(usd=os.path.abspath(path_usd),
                  default_prim=(str(stage.GetDefaultPrim().GetPath())
                                if stage.GetDefaultPrim() else None),
                  meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                  up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                  joints={}, bodies={}, collisions={}, materials={}, artroots={}, scenes={})

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if args.subtree and not path.startswith(args.subtree):
            continue
        schemas = [str(s) for s in prim.GetAppliedSchemas()]
        tname = str(prim.GetTypeName())

        if tname == "PhysicsScene":
            report["scenes"][path] = dict(schemas=schemas, attrs=attrs(prim))

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            report["artroots"][path] = dict(
                schemas=schemas, attrs=attrs(prim, ["physics:", "physxArticulation:"]))

        if prim.IsA(UsdPhysics.Joint):
            j = dict(type=tname, schemas=schemas, rels=rels(prim),
                     attrs=attrs(prim, ["physics:", "physxJoint:", "physxLimit:", "drive:",
                                        "physxMimicJoint:", "physxTendon", "state:"]))
            j["drives"] = {}
            for ax in ("angular", "linear", "transX", "transY", "transZ",
                       "rotX", "rotY", "rotZ", "distance"):
                d = UsdPhysics.DriveAPI(prim, ax)
                if not d:
                    continue
                j["drives"][ax] = dict(
                    type=val(d.GetTypeAttr().Get()),
                    stiffness=val(d.GetStiffnessAttr().Get()),
                    damping=val(d.GetDampingAttr().Get()),
                    maxForce=val(d.GetMaxForceAttr().Get()),
                    targetPosition=val(d.GetTargetPositionAttr().Get()),
                    targetVelocity=val(d.GetTargetVelocityAttr().Get()))
            # Mimic: read by literal name. The instance token is NOT physics:axis -- the four inner
            # joints use rotX and right_outer_knuckle_joint uses rotZ on this asset.
            j["mimic"] = {}
            for s in schemas:
                if "MimicJoint" not in s:
                    continue
                inst = s.split(":", 1)[1] if ":" in s else ""
                m = {}
                for field in ("gearing", "offset", "referenceJoint", "referenceJointAxis",
                              "naturalFrequency", "dampingRatio"):
                    nm = f"physxMimicJoint:{inst}:{field}" if inst else f"physxMimicJoint:{field}"
                    a = prim.GetAttribute(nm)
                    if a and a.IsValid():
                        m[field] = dict(value=val(a.Get()), authored=bool(a.HasAuthoredValue()))
                    r = prim.GetRelationship(nm)
                    if r and r.IsValid() and r.GetTargets():
                        m[field] = dict(targets=[str(x) for x in r.GetTargets()], authored=True)
                j["mimic"][s] = m
            report["joints"][path] = j

        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            report["bodies"][path] = dict(
                type=tname, schemas=schemas, attrs=attrs(prim, ["physics:", "physxRigidBody:"]))

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            c = dict(type=tname, schemas=schemas,
                     attrs=attrs(prim, ["physics:", "physxCollision:", "physxSDFMeshCollision:",
                                        "physxMeshCollision:", "physxConvexHullCollision:",
                                        "physxConvexDecompositionCollision:",
                                        "physxTriangleMeshCollision:"]))
            try:
                bound = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics") \
                    .GetMaterialPath()
                c["physics_material"] = str(bound) if bound else None
            except Exception:
                c["physics_material"] = None
            if tname == "Mesh":
                pts = prim.GetAttribute("points").Get()
                c["n_points"] = 0 if pts is None else len(pts)
            report["collisions"][path] = c

        if prim.HasAPI(UsdPhysics.MaterialAPI) or any("PhysicsMaterialAPI" in s for s in schemas):
            report["materials"][path] = dict(
                schemas=schemas, attrs=attrs(prim, ["physics:", "physxMaterial:"]))

    print(f"\n{path_usd}\n  joints={len(report['joints'])} bodies={len(report['bodies'])} "
          f"collisions={len(report['collisions'])} materials={len(report['materials'])} "
          f"artroots={len(report['artroots'])} scenes={len(report['scenes'])}", flush=True)

    # Gripper-only summary, printed so a failed json write still leaves the answer in the log.
    GRIP = ("finger_joint", "inner_finger", "inner_knuckle", "outer_knuckle", "outer_finger",
            "base_link")
    print("  --- gripper joints ---")
    for p, j in sorted(report["joints"].items()):
        if not any(g in p for g in GRIP):
            continue
        mim = {k.split(":")[-1]: {f: (v.get("value", v.get("targets")))
                                  for f, v in fields.items()}
               for k, fields in j["mimic"].items()}
        drv = {ax: (d["stiffness"], d["damping"], d["maxForce"]) for ax, d in j["drives"].items()}
        print(f"   {p.split('/')[-1]:<30} {j['type']:<22} drive(k,d,maxF)={drv} mimic={mim}")
        lim = {k: v for k, v in j["attrs"].items() if "ower" in k or "pper" in k or "axis" in k}
        print(f"      limits/axis={lim}")
    print("  --- gripper bodies: mass / inertia (scales the mimic spring at fixed nf) ---")
    for p, b in sorted(report["bodies"].items()):
        if not any(g in p for g in GRIP):
            continue
        a = b["attrs"]
        print(f"   {p.split('/')[-1]:<30} mass={a.get('physics:mass')} "
              f"density={a.get('physics:density')} "
              f"diagInertia={a.get('physics:diagonalInertia')} "
              f"com={a.get('physics:centerOfMass')}")
    print("  --- pad collision geoms: offsets + approximation + material ---")
    for p, c in sorted(report["collisions"].items()):
        if "inner_finger" not in p:
            continue
        a = c["attrs"]
        print(f"   {p:<70} approx={a.get('physxMeshCollision:approximation')} "
              f"contactOff={a.get('physxCollision:contactOffset')} "
              f"restOff={a.get('physxCollision:restOffset')} mat={c['physics_material']} "
              f"npts={c.get('n_points')}")
    print("  --- physics materials ---")
    for p, mt in sorted(report["materials"].items()):
        print(f"   {p:<70} {mt['attrs']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        dst = f"{args.out}_{os.path.basename(path_usd).replace('.usd', '')}.json"
        with open(dst, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print(f"  wrote {dst}", flush=True)
    return report


for u in args.usd:
    dump(u)
print("\nWRAPDIFF_USD_DUMP_OK")
