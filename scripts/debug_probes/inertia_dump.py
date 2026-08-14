"""Static USD dump of gripper link mass/inertia and joint frames, for A/B diffing.

Runs on stock `pxr` (pip `usd-core`) -- no Kit, no GPU. PhysX-specific attributes are read by
raw name rather than through PhysxSchema accessors, which stock USD does not register; the
attributes themselves are plain authored properties in the crate file and read back fine.

Usage:  python inertia_dump.py <stage.usd> <out.json>
"""

import json
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

GRIPPER_LINKS = [
    "base_link",
    "left_outer_knuckle", "right_outer_knuckle",
    "left_outer_finger", "right_outer_finger",
    "left_inner_finger", "right_inner_finger",
    "left_inner_knuckle", "right_inner_knuckle",
]


def vec(v):
    """Gf vector/quat -> plain list, so it survives JSON.

    Tokens come back as `str`, which is iterable, so strings and bools are handled before the
    sequence branch or a token like "convexHull" degenerates into a list of characters.
    """
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    if isinstance(v, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
        i = v.GetImaginary()
        return [float(v.GetReal()), float(i[0]), float(i[1]), float(i[2])]
    try:
        return [float(x) for x in v]
    except TypeError:
        return float(v)


def attr(prim, name):
    a = prim.GetAttribute(name)
    if not a or not a.HasAuthoredValue():
        return None
    return vec(a.Get())


def mat(m):
    return [[float(m[i][j]) for j in range(4)] for i in range(4)] if m is not None else None


def find_links(stage):
    """Map link name -> prim for every RigidBodyAPI prim whose name we care about."""
    out = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if prim.GetName() in GRIPPER_LINKS and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            out.setdefault(prim.GetName(), prim)
    return out


def dump_link(prim, cache):
    """Everything that feeds the articulation's effective inertia for this body."""
    world = cache.GetLocalToWorldTransform(prim)
    d = {
        "path": prim.GetPath().pathString,
        "type": prim.GetTypeName(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        # --- mass properties ---
        "mass": attr(prim, "physics:mass"),
        "density": attr(prim, "physics:density"),
        "centerOfMass": attr(prim, "physics:centerOfMass"),
        "diagonalInertia": attr(prim, "physics:diagonalInertia"),
        "principalAxes": attr(prim, "physics:principalAxes"),
        "rigidBodyEnabled": attr(prim, "physics:rigidBodyEnabled"),
        "kinematicEnabled": attr(prim, "physics:kinematicEnabled"),
        # --- PhysX per-body solver overrides ---
        "physx_solverPositionIterationCount": attr(prim, "physxRigidBody:solverPositionIterationCount"),
        "physx_solverVelocityIterationCount": attr(prim, "physxRigidBody:solverVelocityIterationCount"),
        "physx_maxDepenetrationVelocity": attr(prim, "physxRigidBody:maxDepenetrationVelocity"),
        "physx_sleepThreshold": attr(prim, "physxRigidBody:sleepThreshold"),
        "physx_stabilizationThreshold": attr(prim, "physxRigidBody:stabilizationThreshold"),
        "physx_linearDamping": attr(prim, "physxRigidBody:linearDamping"),
        "physx_angularDamping": attr(prim, "physxRigidBody:angularDamping"),
        "physx_maxLinearVelocity": attr(prim, "physxRigidBody:maxLinearVelocity"),
        "physx_maxAngularVelocity": attr(prim, "physxRigidBody:maxAngularVelocity"),
        "physx_enableCCD": attr(prim, "physxRigidBody:enableCCD"),
        "physx_disableGravity": attr(prim, "physxRigidBody:disableGravity"),
        "physx_retainAccelerations": attr(prim, "physxRigidBody:retainAccelerations"),
        # --- pose ---
        "local_translate": attr(prim, "xformOp:translate"),
        "local_orient": attr(prim, "xformOp:orient"),
        "local_scale": attr(prim, "xformOp:scale"),
        "xformOpOrder": [str(x) for x in (prim.GetAttribute("xformOpOrder").Get() or [])],
        "world_matrix": mat(world),
        "world_translate": vec(world.ExtractTranslation()),
        "world_quat": vec(Gf.Quatf(world.ExtractRotationQuat())),
    }
    # --- collision geometry under this link ---
    cols = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not (p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.MeshCollisionAPI)):
            continue
        cols.append({
            "path": p.GetPath().pathString,
            "type": p.GetTypeName(),
            "approximation": attr(p, "physics:approximation"),
            "physx_approximation": attr(p, "physxConvexDecompositionCollision:hullVertexLimit"),
            "physxMeshCollision_approximation": attr(p, "physxMeshCollision:approximation"),
            "collisionEnabled": attr(p, "physics:collisionEnabled"),
            "contactOffset": attr(p, "physxCollision:contactOffset"),
            "restOffset": attr(p, "physxCollision:restOffset"),
            "world_translate": vec(cache.GetLocalToWorldTransform(p).ExtractTranslation()),
        })
    d["collisions"] = cols
    return d


JOINT_ATTRS = [
    "physics:axis", "physics:localPos0", "physics:localRot0", "physics:localPos1",
    "physics:localRot1", "physics:lowerLimit", "physics:upperLimit",
    "physics:jointEnabled", "physics:excludeFromArticulation", "physics:breakForce",
    "physics:breakTorque", "physics:collisionEnabled",
    "drive:angular:physics:type", "drive:angular:physics:stiffness",
    "drive:angular:physics:damping", "drive:angular:physics:maxForce",
    "drive:angular:physics:targetPosition", "drive:angular:physics:targetVelocity",
    "physxJoint:maxJointVelocity", "physxJoint:jointFriction",
    "physxJoint:armature", "physxJoint:enableProjection",
    "physxLimit:angular:stiffness", "physxLimit:angular:damping",
    "physxLimit:rotX:stiffness", "physxLimit:rotX:damping",
]

MIMIC_INSTANCES = ["rotX", "rotY", "rotZ", "transX", "transY", "transZ", "angular", "linear"]
MIMIC_SUFFIXES = ["referenceJoint", "referenceJointAxis", "gearing", "offset",
                  "naturalFrequency", "dampingRatio"]


def dump_joint(prim):
    d = {
        "path": prim.GetPath().pathString,
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
    }
    for rel_name in ("physics:body0", "physics:body1"):
        rel = prim.GetRelationship(rel_name)
        d[rel_name] = [t.pathString for t in rel.GetTargets()] if rel else None
    for a in JOINT_ATTRS:
        d[a] = attr(prim, a)
    mim = {}
    for inst in MIMIC_INSTANCES:
        for suf in MIMIC_SUFFIXES:
            name = f"physxMimicJoint:{inst}:{suf}"
            a = prim.GetAttribute(name)
            if a and a.HasAuthoredValue():
                v = a.Get()
                mim[name] = vec(v) if not isinstance(v, str) else v
        rel = prim.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
        if rel and rel.GetTargets():
            mim[f"physxMimicJoint:{inst}:referenceJoint"] = [t.pathString for t in rel.GetTargets()]
        a = prim.GetAttribute(f"physxMimicJoint:{inst}:referenceJointAxis")
        if a and a.HasAuthoredValue():
            mim[f"physxMimicJoint:{inst}:referenceJointAxis"] = str(a.Get())
    d["mimic"] = mim
    # anything else PhysX-ish we did not enumerate above
    d["other_authored"] = sorted(
        a.GetName() for a in prim.GetAttributes()
        if a.HasAuthoredValue() and a.GetName() not in JOINT_ATTRS
        and not a.GetName().startswith(("physxMimicJoint:", "xformOp"))
    )
    return d


def main(src, out):
    stage = Usd.Stage.Open(src)
    dp = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    result = {
        "stage": src,
        "default_prim": dp.GetPath().pathString,
        "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
        "kgPerUnit": UsdPhysics.GetStageKilogramsPerUnit(stage),
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
    }

    # --- articulation roots ---
    roots = []
    for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append({
                "path": p.GetPath().pathString,
                "articulationEnabled": attr(p, "physics:articulationEnabled"),
                "physx_solverPositionIterationCount": attr(p, "physxArticulation:solverPositionIterationCount"),
                "physx_solverVelocityIterationCount": attr(p, "physxArticulation:solverVelocityIterationCount"),
                "physx_sleepThreshold": attr(p, "physxArticulation:sleepThreshold"),
                "physx_stabilizationThreshold": attr(p, "physxArticulation:stabilizationThreshold"),
                "physx_enabledSelfCollisions": attr(p, "physxArticulation:enabledSelfCollisions"),
                "physx_articulationEnabled": attr(p, "physxArticulation:articulationEnabled"),
            })
    result["articulation_roots"] = roots

    # --- kinematic tree: direct children of the default prim ---
    result["direct_children"] = [
        {"name": p.GetName(), "type": p.GetTypeName(),
         "is_rigid_body": bool(p.HasAPI(UsdPhysics.RigidBodyAPI))}
        for p in dp.GetChildren()
    ]

    # --- every rigid body anywhere, with its full path (tree shape) ---
    result["all_rigid_bodies"] = sorted(
        p.GetPath().pathString for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    )

    # --- links ---
    links = find_links(stage)
    result["links"] = {n: dump_link(p, cache) for n, p in sorted(links.items())}
    result["links_missing"] = [n for n in GRIPPER_LINKS if n not in links]

    # --- joints (all of them, so the arm block can be checked byte-for-byte too) ---
    joints = {}
    for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies()):
        if "Joint" in p.GetTypeName():
            joints[p.GetName()] = dump_joint(p)
    result["joints"] = joints

    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {out}: {len(result['links'])} links, {len(joints)} joints, "
          f"{len(result['all_rigid_bodies'])} rigid bodies, missing={result['links_missing']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
