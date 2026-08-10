"""
Patch a custom USD asset with the physics APIs OmniGibson expects on a single
rigid-body object: RigidBodyAPI + MassAPI on the link Xform, and CollisionAPI +
MeshCollisionAPI on every Mesh under it.

Usage (run inside the OmniGibson container; we boot the simulator briefly so
omnigibson.lazy.pxr resolves):

    # Inspect the prim tree to figure out the structure first:
    python scripts/fix_usd_physics.py inspect /app/custom_assets/bottle.usd

    # Patch in the APIs. Default link name is "base_link"; pass a different
    # name as the last arg if the link Xform inside your USD is called
    # something else:
    python scripts/fix_usd_physics.py fix /app/custom_assets/bottle.usd
    python scripts/fix_usd_physics.py fix /app/custom_assets/bottle.usd Mesh
"""

import sys

import numpy as np

# Launching the simulator is what wires up pxr inside the Isaac Sim runtime;
# without it, lazy.pxr cannot resolve. We launch headless and never call play.
import omnigibson as og
og.launch()  # noqa: E402  # must happen before importing lazy
import omnigibson.lazy as lazy  # noqa: E402


def inspect(usd_path):
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    print(f"root: {root.GetPath()}  ({root.GetTypeName()})")
    for prim in stage.Traverse():
        depth = len(prim.GetPath().pathString.strip("/").split("/")) - 1
        schemas = prim.GetAppliedSchemas()
        api_suffix = f"  [APIs: {', '.join(schemas)}]" if schemas else ""
        xform_parts = []
        if prim.IsA(lazy.pxr.UsdGeom.Xformable):
            xformable = lazy.pxr.UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                try:
                    val = op.Get()
                except Exception:
                    val = "<unreadable>"
                xform_parts.append(f"{op.GetOpName()}={val}")
        xform_suffix = f"  {{xform: {'; '.join(xform_parts)}}}" if xform_parts else ""

        mesh_suffix = ""
        if prim.IsA(lazy.pxr.UsdGeom.Mesh):
            pts_attr = prim.GetAttribute("points")
            if pts_attr.IsValid() and pts_attr.HasValue():
                pts = pts_attr.Get()
                if pts and len(pts) > 0:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    zs = [p[2] for p in pts]
                    bb_min = (min(xs), min(ys), min(zs))
                    bb_max = (max(xs), max(ys), max(zs))
                    size = (bb_max[0] - bb_min[0], bb_max[1] - bb_min[1], bb_max[2] - bb_min[2])
                    mesh_suffix = f"  {{n_pts={len(pts)}; bbox_min={bb_min}; bbox_max={bb_max}; size={size}}}"

        # Visibility / purpose / activation diagnostics for any Imageable prim.
        vis_parts = []
        if prim.IsA(lazy.pxr.UsdGeom.Imageable):
            imageable = lazy.pxr.UsdGeom.Imageable(prim)
            vis_attr = imageable.GetVisibilityAttr()
            if vis_attr.IsValid() and vis_attr.HasAuthoredValue():
                vis_parts.append(f"visibility={vis_attr.Get()}")
            purpose_attr = imageable.GetPurposeAttr()
            if purpose_attr.IsValid() and purpose_attr.HasAuthoredValue():
                vis_parts.append(f"purpose={purpose_attr.Get()}")
        if not prim.IsActive():
            vis_parts.append("ACTIVE=False")
        rb_enabled = prim.GetAttribute("physics:rigidBodyEnabled")
        if rb_enabled.IsValid() and rb_enabled.HasAuthoredValue():
            vis_parts.append(f"rigidBodyEnabled={rb_enabled.Get()}")
        coll_enabled = prim.GetAttribute("physics:collisionEnabled")
        if coll_enabled.IsValid() and coll_enabled.HasAuthoredValue():
            vis_parts.append(f"collisionEnabled={coll_enabled.Get()}")
        mass_attr = prim.GetAttribute("physics:mass")
        if mass_attr.IsValid() and mass_attr.HasAuthoredValue():
            vis_parts.append(f"mass={mass_attr.Get()}")
        com_attr = prim.GetAttribute("physics:centerOfMass")
        if com_attr.IsValid() and com_attr.HasAuthoredValue():
            vis_parts.append(f"centerOfMass={com_attr.Get()}")
        vis_suffix = f"  [{'; '.join(vis_parts)}]" if vis_parts else ""

        print(f"{'  ' * depth}{prim.GetPath()}  ({prim.GetTypeName()}){api_suffix}{xform_suffix}{mesh_suffix}{vis_suffix}")


def fix(usd_path, link_name="base_link", mass=0.5, collision_approximation="convexHull"):
    stage = lazy.pxr.Usd.Stage.Open(usd_path)

    # Find the link prim by name; fall back to the first Xform child of the root.
    root_xform = stage.GetDefaultPrim()
    link = stage.GetPrimAtPath(root_xform.GetPath().AppendChild(link_name))
    if not link.IsValid():
        for child in root_xform.GetChildren():
            if child.IsA(lazy.pxr.UsdGeom.Xform):
                link = child
                break
    if not link.IsValid():
        print(f"  could not find an Xform link prim under {root_xform.GetPath()}; restructure the USD first.")
        return
    print(f"  link prim: {link.GetPath()}")

    # PhysX rejects nested rigid bodies. OG expects the object root to be a
    # plain Xform with NO rigid-body API, and the link Xform underneath to
    # carry the API. Strip any rigid-body / mass APIs from the root.
    for api_cls in (
        lazy.pxr.UsdPhysics.RigidBodyAPI,
        lazy.pxr.UsdPhysics.MassAPI,
    ):
        if root_xform.HasAPI(api_cls):
            root_xform.RemoveAPI(api_cls)
            print(f"  stripped {api_cls.__name__} from {root_xform.GetPath()}")
    # PhysxRigidBodyAPI is the PhysX extension version of the same schema; also
    # clear it if present (set by some Isaac Sim exports).
    physx_rb = getattr(lazy.pxr, "PhysxSchema", None)
    if physx_rb is not None and root_xform.HasAPI(physx_rb.PhysxRigidBodyAPI):
        root_xform.RemoveAPI(physx_rb.PhysxRigidBodyAPI)
        print(f"  stripped PhysxRigidBodyAPI from {root_xform.GetPath()}")

    lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(link)
    mass_api = lazy.pxr.UsdPhysics.MassAPI.Apply(link)
    mass_api.CreateMassAttr(mass)

    for child in link.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Mesh):
            print(f"    collision mesh: {child.GetPath()}")
            lazy.pxr.UsdPhysics.CollisionAPI.Apply(child)
            mca = lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(child)
            mca.CreateApproximationAttr().Set(collision_approximation)

    stage.GetRootLayer().Save()
    print("  saved.")


def bake(usd_path):
    """Bake the world transform of each Mesh under the link into its points/
    normals, then clear every xform op so the chain is all identity. After
    this OG can position the rigid body cleanly and the mesh rides along.

    Uses ComputeLocalToWorldTransform so the composition order is handled by
    USD itself (it's row-vector convention; manually multiplying the wrong
    way silently produces garbage positions)."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root; nothing to bake.")
        return

    meshes = [c for c in link.GetChildren() if c.IsA(lazy.pxr.UsdGeom.Mesh)]
    if not meshes:
        print("  no Mesh prims found under the link; nothing to bake.")
        return

    time_code = lazy.pxr.Usd.TimeCode.Default()
    for mesh in meshes:
        # Per-mesh local-to-world; this composes mesh_local @ link_local @ root_local correctly.
        world_xform = lazy.pxr.UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time_code)
        print(f"  baking transforms into {mesh.GetPath()}")

        # Transform vertex points (full affine).
        pts_attr = mesh.GetAttribute("points")
        if pts_attr.IsValid() and pts_attr.HasValue():
            pts = pts_attr.Get()
            new_pts = [world_xform.Transform(p) for p in pts]
            pts_attr.Set(new_pts)

        # Transform normals (rotation only, then renormalize).
        nrm_attr = mesh.GetAttribute("normals")
        if nrm_attr.IsValid() and nrm_attr.HasValue():
            nrms = nrm_attr.Get()
            new_nrms = []
            for n in nrms:
                t = world_xform.TransformDir(n)
                length = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]) ** 0.5
                if length > 1e-12:
                    t = type(n)(t[0] / length, t[1] / length, t[2] / length)
                new_nrms.append(t)
            nrm_attr.Set(new_nrms)

        # Refresh the local extent (AABB) so renderers/culling don't skip the mesh.
        boundable = lazy.pxr.UsdGeom.Boundable(mesh)
        ext_attr = boundable.GetExtentAttr()
        new_pts_list = mesh.GetAttribute("points").Get()
        if new_pts_list:
            xs = [p[0] for p in new_pts_list]
            ys = [p[1] for p in new_pts_list]
            zs = [p[2] for p in new_pts_list]
            extent = lazy.pxr.Vt.Vec3fArray([
                lazy.pxr.Gf.Vec3f(min(xs), min(ys), min(zs)),
                lazy.pxr.Gf.Vec3f(max(xs), max(ys), max(zs)),
            ])
            ext_attr.Set(extent)

    # Clear xform ops on root, link, and every mesh.
    for prim in (root, link, *meshes):
        lazy.pxr.UsdGeom.Xformable(prim).ClearXformOpOrder()

    stage.GetRootLayer().Save()
    print("  baked + saved.")


def scale_mesh(usd_path, sx, sy, sz):
    """Multiply every mesh vertex (and extent) under the link by (sx, sy, sz).
    Intended for use AFTER bake, when all xforms are identity."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    meshes = [c for c in link.GetChildren() if c.IsA(lazy.pxr.UsdGeom.Mesh)]
    for mesh in meshes:
        pts_attr = mesh.GetAttribute("points")
        if pts_attr.IsValid() and pts_attr.HasValue():
            pts = pts_attr.Get()
            new_pts = [type(p)(p[0] * sx, p[1] * sy, p[2] * sz) for p in pts]
            pts_attr.Set(new_pts)

            xs = [p[0] for p in new_pts]
            ys = [p[1] for p in new_pts]
            zs = [p[2] for p in new_pts]
            extent = lazy.pxr.Vt.Vec3fArray([
                lazy.pxr.Gf.Vec3f(min(xs), min(ys), min(zs)),
                lazy.pxr.Gf.Vec3f(max(xs), max(ys), max(zs)),
            ])
            lazy.pxr.UsdGeom.Boundable(mesh).GetExtentAttr().Set(extent)
            print(f"  scaled {mesh.GetPath()} by ({sx}, {sy}, {sz})")

    stage.GetRootLayer().Save()
    print("  saved.")


def _vertex_cluster_decimate(verts: np.ndarray, tris: np.ndarray, target_faces: int):
    """Pure-numpy vertex-clustering decimator. Lower quality than quadric edge
    collapse, but no external deps and plenty good for a closed collision
    shell — which is all we need."""
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent = bbox_max - bbox_min
    extent = np.where(extent < 1e-6, 1e-6, extent)

    # Iterate voxel resolution until we land near @target_faces. Start from a
    # heuristic guess then double/halve until close.
    def cluster(voxel_size):
        cell = np.floor((verts - bbox_min) / voxel_size).astype(np.int64)
        # Pack the 3-int cell key into a single int64 for fast unique.
        c_min, c_max = cell.min(axis=0), cell.max(axis=0)
        dims = (c_max - c_min + 1)
        key = (
            (cell[:, 0] - c_min[0])
            + (cell[:, 1] - c_min[1]) * dims[0]
            + (cell[:, 2] - c_min[2]) * dims[0] * dims[1]
        )
        # Map each cell -> new vertex (centroid of contained vertices).
        unique_keys, inv = np.unique(key, return_inverse=True)
        n_new = len(unique_keys)
        new_verts = np.zeros((n_new, 3), dtype=np.float32)
        counts = np.zeros(n_new, dtype=np.int64)
        for axis in range(3):
            np.add.at(new_verts[:, axis], inv, verts[:, axis])
        np.add.at(counts, inv, 1)
        new_verts = new_verts / counts[:, None]

        # Remap triangles and drop degenerate / duplicate ones.
        new_tris_raw = inv[tris]
        a, b, c = new_tris_raw[:, 0], new_tris_raw[:, 1], new_tris_raw[:, 2]
        mask = (a != b) & (b != c) & (a != c)
        new_tris = new_tris_raw[mask]
        sorted_faces = np.sort(new_tris, axis=1)
        _, uniq_idx = np.unique(sorted_faces, axis=0, return_index=True)
        return new_verts, new_tris[uniq_idx]

    # Initial guess for voxel size.
    surface_area = 2 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
    voxel_size = float(np.sqrt(surface_area / max(target_faces, 1)))
    if voxel_size <= 0:
        voxel_size = float(extent.max()) / 30.0

    best_verts, best_tris = cluster(voxel_size)
    for _ in range(8):
        n = len(best_tris)
        ratio = n / max(target_faces, 1)
        if 0.7 <= ratio <= 1.5:
            break
        # Too many tris -> bigger voxel; too few -> smaller voxel.
        voxel_size *= ratio ** 0.5
        best_verts, best_tris = cluster(voxel_size)
    return best_verts, best_tris


def decimate_for_collision(usd_path, target_faces=5000, approximation="convexDecomposition"):
    """Add a low-poly *collision-only* sibling mesh next to the existing visual
    mesh, decimated to ~@target_faces triangles via vertex clustering, with
    @approximation as the collision approximation. The original mesh keeps all
    its detail but has its collision APIs stripped so it becomes visual-only.
    """
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    candidate_apis_to_strip = [
        lazy.pxr.UsdPhysics.CollisionAPI,
        lazy.pxr.UsdPhysics.MeshCollisionAPI,
    ]
    physx = lazy.pxr.PhysxSchema
    for n in (
        "PhysxCollisionAPI",
        "PhysxConvexHullCollisionAPI",
        "PhysxConvexDecompositionCollisionAPI",
        "PhysxTriangleMeshCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
        "PhysxTriangleMeshSimplificationCollisionAPI",
    ):
        cls = getattr(physx, n, None)
        if cls is not None:
            candidate_apis_to_strip.append(cls)

    all_meshes = [c for c in link.GetChildren() if c.IsA(lazy.pxr.UsdGeom.Mesh)]
    if not all_meshes:
        print("  no Mesh found under the link.")
        return

    # ---- Cleanup: remove any stale "*_collider" sibling prims left behind by
    # previous runs of this mode. We always rebuild the collision in place on
    # the user's authored collider mesh now.
    stale_paths = [m.GetPath() for m in all_meshes if m.GetName().endswith("_collider")]
    for sp in stale_paths:
        stage.RemovePrim(sp)
        print(f"  removed stale collider prim {sp}")
    if stale_paths:
        # Re-read children after removal.
        all_meshes = [c for c in link.GetChildren() if c.IsA(lazy.pxr.UsdGeom.Mesh)]

    # A "collider candidate" mesh: has any physics collision API applied, OR
    # purpose is guide/proxy, OR name ends in _collider. Everything else
    # (default-purpose, no collision API) is treated as visual-only and left
    # alone. This matches the user's authoring convention: visual mesh stays
    # untouched, collider mesh gets decimated.
    physx = lazy.pxr.PhysxSchema
    collision_api_classes = [
        lazy.pxr.UsdPhysics.CollisionAPI,
        lazy.pxr.UsdPhysics.MeshCollisionAPI,
    ]
    for n in (
        "PhysxCollisionAPI",
        "PhysxConvexHullCollisionAPI",
        "PhysxConvexDecompositionCollisionAPI",
        "PhysxTriangleMeshCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
        "PhysxTriangleMeshSimplificationCollisionAPI",
    ):
        cls = getattr(physx, n, None)
        if cls is not None:
            collision_api_classes.append(cls)

    def has_explicit_collider_marker(m):
        # The user's "intended collider" marker: purpose=guide/proxy or _collider suffix.
        if m.GetName().endswith("_collider"):
            return True
        purpose_attr = lazy.pxr.UsdGeom.Imageable(m).GetPurposeAttr()
        if purpose_attr.HasAuthoredValue() and purpose_attr.Get() in ("guide", "proxy"):
            return True
        return False

    def has_any_collision_api(m):
        for api_cls in collision_api_classes:
            if m.HasAPI(api_cls):
                return True
        return False

    # If any mesh has an explicit collider marker, those are the colliders.
    # Every other mesh with collision APIs is a misclassified visual → strip
    # its collision APIs. If no mesh has explicit markers, fall back to the
    # old behavior (any mesh with collision APIs is a candidate).
    explicit_colliders = [m for m in all_meshes if has_explicit_collider_marker(m)]
    if explicit_colliders:
        collider_meshes = explicit_colliders
        for m in all_meshes:
            if m in explicit_colliders:
                continue
            if has_any_collision_api(m):
                for api_cls in collision_api_classes:
                    if m.HasAPI(api_cls):
                        m.RemoveAPI(api_cls)
                print(f"  stripped collision APIs from non-collider mesh {m.GetPath()}")
    else:
        collider_meshes = [m for m in all_meshes if has_any_collision_api(m)]

    # Also clear the stale rigidBodyEnabled attribute from the root prim if
    # it's present (some Blender USD exports leave that on the outer Xform
    # alongside the actual rigid body on the link → PhysX treats it as nested
    # rigid bodies and bails with the max_shapes AttributeError).
    rb_attr = root.GetAttribute("physics:rigidBodyEnabled")
    if rb_attr.IsValid() and rb_attr.HasAuthoredValue():
        root.RemoveProperty("physics:rigidBodyEnabled")
        print(f"  cleared rigidBodyEnabled on {root.GetPath()}")

    if not collider_meshes:
        print(
            "  no collider candidate Mesh found under the link "
            "(expected a sibling mesh with a collision API or purpose=guide). "
            "Mark one mesh as your collider first, then re-run."
        )
        return

    for collider_mesh in collider_meshes:
        # ---- Read source mesh data ----
        pts_attr = collider_mesh.GetAttribute("points")
        fvc_attr = collider_mesh.GetAttribute("faceVertexCounts")
        fvi_attr = collider_mesh.GetAttribute("faceVertexIndices")
        if not (pts_attr.HasValue() and fvc_attr.HasValue() and fvi_attr.HasValue()):
            print(f"  {collider_mesh.GetPath()}: missing points/faceVertex attrs; skipping.")
            continue
        pts = pts_attr.Get()
        fvc = list(fvc_attr.Get())
        fvi = list(fvi_attr.Get())

        # Bake the collider mesh's local transform into its vertices so we can
        # leave the prim at identity afterward (avoids transform-mismatch with
        # the visual mesh, and simplifies PhysX's view of the collision shape).
        local_xform = lazy.pxr.UsdGeom.Xformable(collider_mesh).GetLocalTransformation()

        # Triangulate any non-triangular faces (fan triangulation).
        raw_verts = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        verts_np = np.array(
            [local_xform.Transform(lazy.pxr.Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))) for v in raw_verts],
            dtype=np.float32,
        )
        tris = []
        idx = 0
        for count in fvc:
            face_idxs = fvi[idx:idx + count]
            idx += count
            for k in range(1, count - 1):
                tris.append([face_idxs[0], face_idxs[k], face_idxs[k + 1]])
        tris_np = np.array(tris, dtype=np.int64)
        src_face_count = len(tris_np)
        print(f"  {collider_mesh.GetPath()}: {len(verts_np)} verts, {src_face_count} tris (after triangulation + local-xform bake)")

        # ---- Decimate ----
        if src_face_count <= target_faces:
            print(f"    already at/below {target_faces} faces; using as-is.")
            simplified_verts = verts_np
            simplified_faces = tris_np
        else:
            simplified_verts, simplified_faces = _vertex_cluster_decimate(verts_np, tris_np, target_faces)
        print(f"    simplified to {len(simplified_verts)} verts, {len(simplified_faces)} tris")

        # ---- Update the collider mesh IN PLACE ----
        # Replace points/faceVertex attrs with the decimated geometry.
        new_points = [
            lazy.pxr.Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
            for v in simplified_verts
        ]
        new_fvc = [3] * len(simplified_faces)
        new_fvi = [int(x) for x in simplified_faces.flatten().tolist()]

        collider_geom = lazy.pxr.UsdGeom.Mesh(collider_mesh)
        collider_geom.GetPointsAttr().Set(new_points)
        collider_geom.GetFaceVertexCountsAttr().Set(new_fvc)
        collider_geom.GetFaceVertexIndicesAttr().Set(new_fvi)

        # Refresh extent.
        xs = [v[0] for v in new_points]
        ys = [v[1] for v in new_points]
        zs = [v[2] for v in new_points]
        collider_geom.GetExtentAttr().Set(
            lazy.pxr.Vt.Vec3fArray([
                lazy.pxr.Gf.Vec3f(min(xs), min(ys), min(zs)),
                lazy.pxr.Gf.Vec3f(max(xs), max(ys), max(zs)),
            ])
        )

        # Clear the local xform now that the transform is baked into the points.
        lazy.pxr.UsdGeom.Xformable(collider_mesh).ClearXformOpOrder()

        # Drop normals — they'd be wrong after decimation and aren't needed for collision.
        nrm_attr = collider_mesh.GetAttribute("normals")
        if nrm_attr.IsValid() and nrm_attr.HasValue():
            nrm_attr.Clear()

        # Hide it (just in case) and force purpose=guide so it never renders.
        lazy.pxr.UsdGeom.Imageable(collider_mesh).CreateVisibilityAttr().Set("invisible")
        lazy.pxr.UsdGeom.Imageable(collider_mesh).CreatePurposeAttr().Set("guide")

        # Re-apply collision APIs in case they were stripped previously.
        lazy.pxr.UsdPhysics.CollisionAPI.Apply(collider_mesh)
        mca = lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(collider_mesh)
        mca.CreateApproximationAttr().Set(approximation)
        if approximation == "convexHull":
            physx.PhysxConvexHullCollisionAPI.Apply(collider_mesh)
        elif approximation == "convexDecomposition":
            physx.PhysxConvexDecompositionCollisionAPI.Apply(collider_mesh)
        elif approximation == "sdf":
            physx.PhysxSDFMeshCollisionAPI.Apply(collider_mesh)
        elif approximation == "meshSimplification":
            physx.PhysxTriangleMeshSimplificationCollisionAPI.Apply(collider_mesh)
        print(f"    updated {collider_mesh.GetPath()} in place as collision-only ({approximation})")

    stage.GetRootLayer().Save()
    print("  saved.")


def set_center_of_mass(usd_path, cx, cy, cz):
    """Author physics:centerOfMass on the link Xform under the default prim.
    Coordinates are in the link's local frame (so they're affected by any
    xform ops on the link itself). For a bottle that tends to tip on contact,
    pushing the CoM down (negative on the bottle's tall axis) makes it
    bottom-heavy and more stable."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    mass_api = lazy.pxr.UsdPhysics.MassAPI.Apply(link)
    com_attr = mass_api.GetCenterOfMassAttr()
    if not com_attr or not com_attr.IsValid():
        com_attr = mass_api.CreateCenterOfMassAttr()
    com_attr.Set(lazy.pxr.Gf.Vec3f(float(cx), float(cy), float(cz)))
    print(f"  set centerOfMass on {link.GetPath()} -> ({cx}, {cy}, {cz})")

    stage.GetRootLayer().Save()
    print("  saved.")


def add_default_material(usd_path, color=(0.6, 0.6, 0.6), mat_name="DefaultSurface"):
    """Create an OmniPBR material (Isaac Sim's RTX-native shader) and bind it
    to every Mesh under the link. Also set primvars:displayColor as a fallback
    for renderers that don't pick up MDL. Isaac Sim's RTX path skips meshes
    whose bound material doesn't resolve to an OmniPBR / MDL surface."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    # Create or reuse a /<root>/materials scope and the material under it.
    materials_scope_path = root.GetPath().AppendChild("materials")
    materials_scope = stage.GetPrimAtPath(materials_scope_path)
    if not materials_scope.IsValid():
        materials_scope = lazy.pxr.UsdGeom.Scope.Define(stage, materials_scope_path).GetPrim()

    mat_path = materials_scope_path.AppendChild(mat_name)
    material = lazy.pxr.UsdShade.Material.Define(stage, mat_path)

    # OmniPBR is the Isaac Sim native MDL surface — RTX recognizes it.
    omni_shader_path = mat_path.AppendChild("OmniSurface")
    omni_shader = lazy.pxr.UsdShade.Shader.Define(stage, omni_shader_path)
    omni_shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    omni_shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    omni_shader.CreateInput("diffuse_color_constant", lazy.pxr.Sdf.ValueTypeNames.Color3f).Set(
        lazy.pxr.Gf.Vec3f(*color)
    )
    omni_shader.CreateInput("reflection_roughness_constant", lazy.pxr.Sdf.ValueTypeNames.Float).Set(0.7)
    omni_shader.CreateInput("metallic_constant", lazy.pxr.Sdf.ValueTypeNames.Float).Set(0.0)

    # Connect through the MDL "surface" terminal.
    out = omni_shader.CreateOutput("out", lazy.pxr.Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)

    # Fallback UsdPreviewSurface for non-RTX viewers.
    preview_shader_path = mat_path.AppendChild("PreviewSurface")
    preview_shader = lazy.pxr.UsdShade.Shader.Define(stage, preview_shader_path)
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    preview_shader.CreateInput("diffuseColor", lazy.pxr.Sdf.ValueTypeNames.Color3f).Set(
        lazy.pxr.Gf.Vec3f(*color)
    )
    preview_shader.CreateInput("roughness", lazy.pxr.Sdf.ValueTypeNames.Float).Set(0.7)
    preview_shader.CreateInput("metallic", lazy.pxr.Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        preview_shader.ConnectableAPI(), "surface"
    )

    # Bind to every mesh under the link, plus a hard-coded displayColor fallback.
    for mesh in link.GetChildren():
        if not mesh.IsA(lazy.pxr.UsdGeom.Mesh):
            continue
        binding_api = lazy.pxr.UsdShade.MaterialBindingAPI.Apply(mesh)
        binding_api.Bind(material)
        # displayColor as last-ditch fallback for primitive Hydra paths.
        disp_attr = lazy.pxr.UsdGeom.Mesh(mesh).CreateDisplayColorPrimvar("constant")
        disp_attr.Set([lazy.pxr.Gf.Vec3f(*color)])
        print(f"  bound {mat_path} (OmniPBR + UsdPreviewSurface + displayColor) to {mesh.GetPath()}")

    stage.GetRootLayer().Save()
    print("  saved.")


def make_double_sided(usd_path):
    """Force doubleSided=true on every Mesh under the link. Use this when the
    mesh is colliding but not rendering — usually a face-winding issue where
    the visible faces happen to be culled because they're facing 'away' from
    the camera."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    for mesh in link.GetChildren():
        if not mesh.IsA(lazy.pxr.UsdGeom.Mesh):
            continue
        ds_attr = lazy.pxr.UsdGeom.Mesh(mesh).GetDoubleSidedAttr()
        if not ds_attr:
            ds_attr = lazy.pxr.UsdGeom.Mesh(mesh).CreateDoubleSidedAttr()
        ds_attr.Set(True)
        print(f"  forced doubleSided=True on {mesh.GetPath()}")

    stage.GetRootLayer().Save()
    print("  saved.")


def clean_collision(usd_path, approximation="convexDecomposition"):
    """Strip every PhysxConvexHull/TriangleMesh/ConvexDecomposition/SDFMesh
    collision API from every Mesh under the link, then re-apply only the
    requested @approximation. Stacking multiple approximation APIs puts
    PhysX in an undefined state and on dynamic bodies usually disables the
    body silently."""
    stage = lazy.pxr.Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()

    link = None
    for child in root.GetChildren():
        if child.IsA(lazy.pxr.UsdGeom.Xform):
            link = child
            break
    if link is None:
        print("  no link Xform under root.")
        return

    physx = lazy.pxr.PhysxSchema
    candidate_apis = [
        getattr(physx, n, None)
        for n in (
            "PhysxConvexHullCollisionAPI",
            "PhysxConvexDecompositionCollisionAPI",
            "PhysxTriangleMeshCollisionAPI",
            "PhysxSDFMeshCollisionAPI",
            "PhysxTriangleMeshSimplificationCollisionAPI",
        )
    ]
    candidate_apis = [a for a in candidate_apis if a is not None]

    meshes = [c for c in link.GetChildren() if c.IsA(lazy.pxr.UsdGeom.Mesh)]

    # If any mesh has an explicit collider marker (_collider suffix or
    # purpose=guide/proxy), restrict cleanup to those. Other meshes are
    # assumed to be visual-only and any stray collision APIs on them are
    # stripped wholesale.
    def is_explicit_collider(m):
        if m.GetName().endswith("_collider"):
            return True
        purpose_attr = lazy.pxr.UsdGeom.Imageable(m).GetPurposeAttr()
        if purpose_attr.HasAuthoredValue() and purpose_attr.Get() in ("guide", "proxy"):
            return True
        return False

    explicit_colliders = [m for m in meshes if is_explicit_collider(m)]
    if explicit_colliders:
        # Strip any leftover collision APIs from non-collider visual meshes.
        for m in meshes:
            if m in explicit_colliders:
                continue
            for api_cls in candidate_apis + [lazy.pxr.UsdPhysics.CollisionAPI, lazy.pxr.UsdPhysics.MeshCollisionAPI]:
                if m.HasAPI(api_cls):
                    m.RemoveAPI(api_cls)
                    print(f"  stripped {api_cls.__name__} from visual mesh {m.GetPath()}")
        target_meshes = explicit_colliders
    else:
        target_meshes = meshes

    for mesh in target_meshes:
        for api_cls in candidate_apis:
            if mesh.HasAPI(api_cls):
                mesh.RemoveAPI(api_cls)
                print(f"  removed {api_cls.__name__} from {mesh.GetPath()}")

        # Re-apply the requested one + the approximation attr.
        mca = lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(mesh)
        mca.CreateApproximationAttr().Set(approximation)
        # The corresponding PhysX schema (auto-applied by geom_prim too).
        if approximation == "convexHull":
            physx.PhysxConvexHullCollisionAPI.Apply(mesh)
        elif approximation == "convexDecomposition":
            physx.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
        elif approximation == "sdf":
            physx.PhysxSDFMeshCollisionAPI.Apply(mesh)
        elif approximation == "meshSimplification":
            physx.PhysxTriangleMeshSimplificationCollisionAPI.Apply(mesh)
        print(f"  set {mesh.GetPath()} approximation -> {approximation}")

    stage.GetRootLayer().Save()
    print("  saved.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__.strip())
        sys.exit(1)
    mode = sys.argv[1]
    paths = sys.argv[2:]
    if mode == "inspect":
        for p in paths:
            print(p)
            inspect(p)
    elif mode == "fix":
        link_name = "base_link"
        if paths and not paths[-1].endswith(".usd"):
            link_name = paths[-1]
            paths = paths[:-1]
        for p in paths:
            print(p)
            fix(p, link_name=link_name)
    elif mode == "bake":
        for p in paths:
            print(p)
            bake(p)
    elif mode == "double_sided":
        for p in paths:
            print(p)
            make_double_sided(p)
    elif mode == "default_material":
        for p in paths:
            print(p)
            add_default_material(p)
    elif mode == "set_com":
        # Last 3 args are cx, cy, cz; the rest are usd paths.
        cx, cy, cz = float(paths[-3]), float(paths[-2]), float(paths[-1])
        paths = paths[:-3]
        for p in paths:
            print(p)
            set_center_of_mass(p, cx, cy, cz)
    elif mode == "decimate_collision":
        # Optional last arg = target_face_count.
        target_faces = 5000
        if paths and not paths[-1].endswith(".usd"):
            target_faces = int(paths[-1])
            paths = paths[:-1]
        for p in paths:
            print(p)
            decimate_for_collision(p, target_faces=target_faces)
    elif mode == "clean_collision":
        approx = "convexDecomposition"
        if paths and not paths[-1].endswith(".usd"):
            approx = paths[-1]
            paths = paths[:-1]
        for p in paths:
            print(p)
            clean_collision(p, approximation=approx)
    elif mode == "scale":
        # Last 3 args (or 1 uniform arg) are scale factors; the rest are paths.
        if len(paths) >= 4:
            sx, sy, sz = float(paths[-3]), float(paths[-2]), float(paths[-1])
            paths = paths[:-3]
        else:
            s = float(paths[-1])
            sx = sy = sz = s
            paths = paths[:-1]
        for p in paths:
            print(p)
            scale_mesh(p, sx, sy, sz)
    else:
        print(f"unknown mode: {mode!r}; use 'inspect', 'fix', 'bake', 'scale', 'clean_collision', 'double_sided', 'default_material', 'decimate_collision', or 'set_com'")

    og.shutdown()
