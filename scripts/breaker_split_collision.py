"""OmniGibson classifies a mesh as COLLISION (hidden, purpose=guide) if it has
UsdPhysics.CollisionAPI, otherwise VISUAL (rendered). A mesh cannot be both.

breaker_box.usd currently has CollisionAPI on every visual mesh, so OG treats
them all as (hidden) collision meshes and the object renders nothing.

This script, for each Mesh under base_link that has a CollisionAPI:
  1. creates a `<name>_collider` sibling that is a geometry copy, carrying the
     collision APIs and hidden (purpose=guide),
  2. strips the collision APIs off the original so it becomes a visible visual
     mesh (purpose=default).

Run inside the OmniGibson docker:
    python /app/scripts/breaker_split_collision.py
"""
import omnigibson as og
og.launch()
import omnigibson.lazy as lazy

Usd = lazy.pxr.Usd
UsdGeom = lazy.pxr.UsdGeom
UsdPhysics = lazy.pxr.UsdPhysics
PhysxSchema = lazy.pxr.PhysxSchema

USD = "/app/custom_assets/breaker/breaker_box.usd"
APPROX = "convexHull"  # cubes -> convex hull is exact and cheap

COLLISION_APIS = [
    UsdPhysics.CollisionAPI,
    UsdPhysics.MeshCollisionAPI,
]
for n in (
    "PhysxCollisionAPI",
    "PhysxConvexHullCollisionAPI",
    "PhysxConvexDecompositionCollisionAPI",
    "PhysxTriangleMeshCollisionAPI",
    "PhysxSDFMeshCollisionAPI",
):
    cls = getattr(PhysxSchema, n, None)
    if cls is not None:
        COLLISION_APIS.append(cls)


def copy_xform_ops(src, dst):
    dst_xf = UsdGeom.Xformable(dst)
    dst_xf.ClearXformOpOrder()
    for op in UsdGeom.Xformable(src).GetOrderedXformOps():
        new_op = dst_xf.AddXformOp(op.GetOpType(), op.GetPrecision())
        v = op.Get()
        if v is not None:
            new_op.Set(v)


def copy_geom(src_mesh, dst_mesh):
    for attr in ("points", "faceVertexCounts", "faceVertexIndices", "extent",
                 "subdivisionScheme"):
        a = src_mesh.GetAttribute(attr)
        if a.IsValid() and a.HasValue():
            dst_mesh.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName()).Set(a.Get())


s = Usd.Stage.Open(USD)
link = s.GetPrimAtPath("/breaker_box/base_link")

meshes = [c for c in link.GetChildren()
          if c.IsA(UsdGeom.Mesh) and not c.GetName().endswith("_collider")]

for mesh in meshes:
    has_coll = any(mesh.HasAPI(api) for api in COLLISION_APIS)
    if not has_coll:
        print("skip (already visual-only):", mesh.GetPath())
        continue

    # 1. Build the collider twin.
    collider_path = mesh.GetPath().GetParentPath().AppendChild(mesh.GetName() + "_collider")
    if s.GetPrimAtPath(collider_path).IsValid():
        s.RemovePrim(collider_path)
    collider = UsdGeom.Mesh.Define(s, collider_path)
    copy_geom(UsdGeom.Mesh(mesh), collider)
    copy_xform_ops(mesh, collider.GetPrim())
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    mca = UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
    mca.CreateApproximationAttr().Set(APPROX)
    if APPROX == "convexHull":
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(collider.GetPrim())
    UsdGeom.Imageable(collider.GetPrim()).CreatePurposeAttr().Set("guide")
    UsdGeom.Imageable(collider.GetPrim()).CreateVisibilityAttr().Set("invisible")
    print("created collider:", collider_path)

    # 2. Strip collision off the original so it renders as a visual mesh.
    for api in COLLISION_APIS:
        if mesh.HasAPI(api):
            mesh.RemoveAPI(api)
    ce = mesh.GetAttribute("physics:collisionEnabled")
    if ce.IsValid() and ce.HasAuthoredValue():
        mesh.RemoveProperty("physics:collisionEnabled")
    UsdGeom.Imageable(mesh).CreatePurposeAttr().Set("default")
    vis = UsdGeom.Imageable(mesh).GetVisibilityAttr()
    if vis.IsValid() and vis.HasAuthoredValue() and vis.Get() == "invisible":
        vis.Set("inherited")
    print("stripped collision -> visual:", mesh.GetPath())

s.GetRootLayer().Save()
print("saved.")
og.shutdown()
