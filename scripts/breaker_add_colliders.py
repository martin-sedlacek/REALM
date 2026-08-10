"""Add physics to a visual-only OmniGibson asset WITHOUT touching the visible
meshes. For each visual Mesh under the link it creates a hidden `<name>_collider`
twin that carries the collision APIs (purpose=guide, invisible), and ensures the
link has RigidBodyAPI + MassAPI.

OmniGibson buckets a mesh as collision (hidden) iff it has CollisionAPI, else
visual (rendered). So the visible meshes must stay collision-free; the colliders
are the hidden twins.

Run inside the OmniGibson docker:
    python /app/scripts/breaker_add_colliders.py [/app/path/to/asset.usd]
"""
import sys

import omnigibson as og
og.launch()
import omnigibson.lazy as lazy

Usd = lazy.pxr.Usd
UsdGeom = lazy.pxr.UsdGeom
UsdPhysics = lazy.pxr.UsdPhysics
PhysxSchema = lazy.pxr.PhysxSchema

USD = sys.argv[1] if len(sys.argv) > 1 else "/app/custom_assets/breaker/breaker_box.usd"
APPROX = "convexHull"   # exact + cheap for boxy shapes
MASS = 0.5

COLLISION_APIS = [UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI]
for n in ("PhysxCollisionAPI", "PhysxConvexHullCollisionAPI",
          "PhysxConvexDecompositionCollisionAPI", "PhysxTriangleMeshCollisionAPI",
          "PhysxSDFMeshCollisionAPI"):
    cls = getattr(PhysxSchema, n, None)
    if cls is not None:
        COLLISION_APIS.append(cls)


def find_link(stage):
    root = stage.GetDefaultPrim()
    link = stage.GetPrimAtPath(root.GetPath().AppendChild("base_link"))
    if link.IsValid():
        return link
    for c in root.GetChildren():
        if c.IsA(UsdGeom.Xform):
            return c
    return root


def copy_xform_ops(src, dst):
    dxf = UsdGeom.Xformable(dst)
    dxf.ClearXformOpOrder()
    for op in UsdGeom.Xformable(src).GetOrderedXformOps():
        new = dxf.AddXformOp(op.GetOpType(), op.GetPrecision())
        v = op.Get()
        if v is not None:
            new.Set(v)


def copy_geom(src, dst):
    for attr in ("points", "faceVertexCounts", "faceVertexIndices", "extent",
                 "subdivisionScheme"):
        a = src.GetAttribute(attr)
        if a.IsValid() and a.HasValue():
            dst.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName()).Set(a.Get())


s = Usd.Stage.Open(USD)
link = find_link(s)
print("link:", link.GetPath())

# Rigid body + mass on the link (OG anchors it via fixed_base=True in the config).
UsdPhysics.RigidBodyAPI.Apply(link)
UsdPhysics.MassAPI.Apply(link).CreateMassAttr(MASS)
# Make sure the ROOT is not also a rigid body (nested bodies -> PhysX bails).
root = s.GetDefaultPrim()
if root != link:
    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
        if root.HasAPI(api):
            root.RemoveAPI(api)
            print("stripped", api.__name__, "from root", root.GetPath())

# Collect visual meshes (no collision, not already a collider), up to 2 levels
# deep under the link -- matching how OG discovers meshes.
candidates = []
for c in link.GetChildren():
    candidates.append(c)
    for gc in c.GetChildren():
        candidates.append(gc)

visual_meshes = [
    m for m in candidates
    if m.IsA(UsdGeom.Mesh)
    and not m.GetName().endswith("_collider")
    and not any(m.HasAPI(api) for api in COLLISION_APIS)
]
print(f"found {len(visual_meshes)} visual meshes")

for mesh in visual_meshes:
    coll_path = mesh.GetPath().GetParentPath().AppendChild(mesh.GetName() + "_collider")
    if s.GetPrimAtPath(coll_path).IsValid():
        s.RemovePrim(coll_path)
    collider = UsdGeom.Mesh.Define(s, coll_path)
    copy_geom(UsdGeom.Mesh(mesh), collider)
    copy_xform_ops(mesh, collider.GetPrim())
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr().Set(APPROX)
    if APPROX == "convexHull":
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(collider.GetPrim())
    UsdGeom.Imageable(collider.GetPrim()).CreatePurposeAttr().Set("guide")
    UsdGeom.Imageable(collider.GetPrim()).CreateVisibilityAttr().Set("invisible")
    print("  + collider twin:", coll_path)

s.GetRootLayer().Save()
print("saved:", USD)
og.shutdown()
