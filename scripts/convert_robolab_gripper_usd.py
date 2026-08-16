"""Restructure the robolab Franka + Robotiq 2F-85 USD into the layout OmniGibson expects.

The robolab asset nests the gripper's rigid bodies three levels deep:

    /panda                                  [ArticulationRoot]
      panda_link0 .. panda_link8            [RigidBody]   <- direct children, fine
      Gripper                               <Xform, translate+rotate 180deg>
        Robotiq_2F_85                       <Xform, identity>
          base_link, *_outer_knuckle, *_outer_finger, *_inner_finger, *_inner_knuckle  [RigidBody]
          Joints                            <Scope>  (finger_joint + 5 mimic joints)
          Shader

OmniGibson builds its link set from *direct Xform children* of the robot prim only
(`EntityPrim._load_links`, entity_prim.py:202), so as-is it would see ten links -- the nine panda
links plus a bogus "Gripper" -- and none of the actual gripper bodies. Downstream that breaks link
lookups by name (finger_link_names), root-link inference, and the articulation's joint accounting.

This script rewrites the asset so every rigid body is a direct child of /panda, preserving world
poses by baking the Gripper xform into each moved body's local transform. It also:

  - moves the Joints scope and Shader to /panda (Scope/Shader are not Xforms, so OmniGibson will
    not mistake them for links; joints are found by full-subtree traversal regardless)
  - deletes the asset's own /panda/rootJoint. It has body0=panda_link0 with an *empty* body1, which
    makes `_load_links` raise IndexError on `body1[0]`, and OmniGibson creates its own rootJoint at
    exactly that path for fixed-base robots anyway.

Usage:
    python scripts/convert_robolab_gripper_usd.py <src.usd> <dst.usd>
"""

import shutil
import sys

import omnigibson as og

og.launch()

import omnigibson.lazy as lazy  # noqa: E402

Usd, UsdGeom, UsdPhysics, Sdf, Gf = (
    lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics, lazy.pxr.Sdf, lazy.pxr.Gf
)
PhysxSchema = lazy.pxr.PhysxSchema

def remap_paths(stage, old_prefix, new_prefix):
    """Rewrite every relationship target / attribute connection under old_prefix.

    Sdf.BatchNamespaceEdit moves the prim specs but leaves authored paths that point *into* the
    moved subtree untouched, so joint body0/body1 and material bindings would dangle without this.
    """
    n_rel = n_conn = 0
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            new = [Sdf.Path(t.pathString.replace(old_prefix, new_prefix, 1))
                   if t.pathString.startswith(old_prefix) else t for t in targets]
            if new != list(targets):
                rel.SetTargets(new)
                n_rel += 1
        for attr in prim.GetAttributes():
            conns = attr.GetConnections()
            new = [Sdf.Path(c.pathString.replace(old_prefix, new_prefix, 1))
                   if c.pathString.startswith(old_prefix) else c for c in conns]
            if new != list(conns):
                attr.SetConnections(new)
                n_conn += 1
    print(f"remapped {n_rel} relationship(s) and {n_conn} attribute connection(s)")


def strip_mimic_drives(stage):
    """Remove DriveAPI from mimic joints.

    A mimic joint's motion is enforced by its PhysX mimic constraint, not by a motor, and the ones
    in this asset carry vestigial drives with stiffness=0, damping=0. OmniGibson asserts that every
    DOF *not* claimed by a controller is undriven
    (robot.py:658, "All unused joints not mapped to any controller should not have DriveAPI"),
    and only `finger_joint` is claimed here -- deliberately, since commanding the followers directly
    would fight the linkage. Their drives contribute nothing, so they are removed.
    """
    Drive = UsdPhysics.DriveAPI
    stripped = []
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if "Joint" not in prim.GetTypeName():
            continue
        is_mimic = any("Mimic" in schema for schema in prim.GetAppliedSchemas())
        if not is_mimic or not prim.HasAPI(Drive):
            continue
        for inst in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
            if prim.HasAPI(Drive, inst):
                prim.RemoveAPI(Drive, inst)
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith("drive:"):
                prim.RemoveProperty(attr.GetName())
        stripped.append(prim.GetName())
    print(f"stripped DriveAPI from {len(stripped)} mimic joint(s): {stripped}")


def add_visual_copies(stage):
    """Give every link a *visual* copy of its geometry.

    OmniGibson splits a link's geoms into visual and collision sets in RigidPrim.update_meshes():
    a geom counts as collision if it, or any ancestor inside the link, carries UsdPhysics.CollisionAPI
    or PhysxCollisionAPI -- and collision meshes are then hidden with `purpose = "guide"`. Assets
    authored for OmniGibson keep separate `visuals`/`collisions` subtrees, but this one reuses a
    single mesh per link for both, so *everything* is classified as collision and the robot renders
    completely invisible (physics and cameras still work, which makes it a confusing failure).

    For each link, every direct child subtree containing a collision API is duplicated alongside the
    original and stripped of its collision APIs. Copying at the same level preserves the local
    transform, so the copy lands on the original's world pose with no transform maths.

    TWO KNOWN GAPS, both measured 2026-08-17 -- read before "fixing" either.

    1. The type filter below is `("Xform", "Mesh")`, so a collider authored as a **primitive gprim**
       (`Cylinder`, `Cube`, `Sphere`, `Cone`) is skipped and gets no visual copy. The robolab asset
       has exactly one: `/panda/panda_link8/gripper_adapter`, the pad between the flange and the
       gripper. Widening the tuple is not the fix on its own -- see 2.
    2. **A visual copy placed on the EEF link would not render even so.** `Robot._load_controllers`
       (`robot.py:1255`) calls `self._links[self.eef_link_names[arm]].visible = False`, and
       `panda_link8` IS this robot's eef link, so OmniGibson authors `visibility = invisible` on the
       link prim. USD visibility prunes: no descendant can override an invisible ancestor, and
       `purpose` is orthogonal to it. Measured -- `purpose = "render"` on such a copy moves 605 of
       359,637 pixels against a 390-pixel noise floor, i.e. nothing.

    So a render-only copy of an EEF-link collider has to be parented to a *visible* link that is
    rigidly joined to it. `scripts/fix_link8_adapter_visual.py` does that for this one prim, moving
    it to `/panda/base_link` (the gripper base, joined by the fixed `panda_hand_joint`).
    """
    layer = stage.GetRootLayer()
    Collision, MeshCollision = UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI
    PhysxCollision = PhysxSchema.PhysxCollisionAPI

    def has_collision(prim):
        return any(p.HasAPI(Collision) or p.HasAPI(PhysxCollision) for p in Usd.PrimRange(prim))

    todo = []
    for link in stage.GetDefaultPrim().GetChildren():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        for child in link.GetChildren():
            if child.GetTypeName() in ("Xform", "Mesh") and has_collision(child):
                todo.append((child.GetPath().pathString,
                             f"{link.GetPath().pathString}/visual__{child.GetName()}"))

    for src, dst in todo:
        assert Sdf.CopySpec(layer, Sdf.Path(src), layer, Sdf.Path(dst)), f"CopySpec {src} -> {dst}"

    stripped = 0
    for _, dst in todo:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(dst)):
            for api, inst in ((Collision, None), (PhysxCollision, None), (MeshCollision, None)):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
                    stripped += 1
            for attr in list(prim.GetAttributes()):
                if attr.GetName().startswith(("physics:collision", "physxCollision:", "physics:approximation")):
                    prim.RemoveProperty(attr.GetName())
    print(f"duplicated {len(todo)} geom subtree(s) as visuals, stripped {stripped} collision API(s)")


def deinstance(stage):
    """Clear the `instanceable` flag on referenced prims.

    The gripper's collision geometry comes in via references to the Defeatured_2F_85_*.usd files,
    authored as instanceable. OmniGibson force-writes material inputs
    (MaterialPrim._post_load -> set_input("reflection_roughness_texture_influence")) onto every
    material it wraps, and USD refuses property authoring on an instance proxy:
      "Cannot create property spec at path .../Looks/Cod_Gray/Shader; authoring to an instance
       proxy is not allowed".
    XFormPrim._post_load also asserts outright that prims are neither instanceable nor proxies.
    Dropping instancing costs memory/draw-call sharing but is required for OmniGibson to load these.
    """
    n = 0
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            n += 1
    print(f"cleared instanceable flag on {n} prim(s)")


def ensure_scale_ops(stage):
    """Author an identity xformOp:scale on any Xformable lacking one.

    OmniGibson's XFormPrim._post_load does `th.tensor(self.get_attribute("xformOp:scale"))` on every
    prim it wraps, and only synthesises the standard ops for prims it creates itself -- a USD loaded
    from disk is assumed to have them already. The robolab gripper's collision meshes author only
    translate+orient, so loading raises "Could not infer dtype of NoneType". Appending an identity
    scale to the existing op order leaves the transform unchanged.
    """
    n = 0
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        x = UsdGeom.Xformable(prim)
        if not x:
            continue
        if any(op.GetOpName() == "xformOp:scale" for op in x.GetOrderedXformOps()):
            continue
        x.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        n += 1
    print(f"authored identity xformOp:scale on {n} prim(s) that lacked one")


def prune_dangling(stage):
    """Drop relationship targets that resolve to nothing.

    The source asset carries Isaac robot-schema bookkeeping on /panda
    (`isaac:physics:robotJoints`) still pointing at `panda_hand/panda_finger_joint1|2` -- the stock
    Franka hand this asset replaced with the Robotiq gripper, and which exists nowhere in the file
    -- plus the rootJoint we remove. They are stale in the source, not produced by this conversion.
    """
    pruned = []
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        for rel in prim.GetRelationships():
            targets = list(rel.GetTargets())
            keep = [t for t in targets if stage.GetPrimAtPath(t)]
            if len(keep) != len(targets):
                pruned += [(prim.GetPath().pathString, rel.GetName(), t.pathString)
                           for t in targets if t not in keep]
                rel.SetTargets(keep)
    print(f"pruned {len(pruned)} stale relationship target(s):")
    for x in pruned:
        print(f"    {x[0]} .{x[1]} -> {x[2]}")


def matrices_close(a, b, eps=1e-9):
    """Element-wise Matrix4d comparison (Gf.IsClose has no Matrix4d overload)."""
    return all(abs(a[i][j] - b[i][j]) <= eps for i in range(4) for j in range(4))


GROUP_PATH = "/panda/Gripper"
INNER_PATH = "/panda/Gripper/Robotiq_2F_85"
ROOT_JOINT = "/panda/rootJoint"


def main(src, dst):
    shutil.copyfile(src, dst)
    print(f"copied {src} -> {dst}")

    # --- 1. record world transforms before the move -------------------------------------------
    stage = Usd.Stage.Open(dst)
    inner = stage.GetPrimAtPath(INNER_PATH)
    assert inner, f"{INNER_PATH} not found -- asset layout differs from what this script expects"

    # Names as plain strings, and transforms as Gf value copies: both must outlive the stage,
    # since Usd.Prim handles dangle (and segfault) once the stage is released.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    movable_names = [p.GetName() for p in inner.GetChildren()]
    world = {
        p.GetName(): Gf.Matrix4d(cache.GetLocalToWorldTransform(p))
        for p in inner.GetChildren()
        if p.GetTypeName() == "Xform"
    }
    print(f"moving {len(movable_names)} prims ({len(world)} of them Xforms with poses to preserve)")

    # /panda itself must be identity for the bake below to be a straight copy of the world matrix.
    root_xf = Gf.Matrix4d(cache.GetLocalToWorldTransform(stage.GetDefaultPrim()))
    assert matrices_close(root_xf, Gf.Matrix4d(1.0)), f"/panda is not identity: {root_xf}"

    del cache
    del inner
    del stage  # drop the Usd stage before editing the layer's namespace

    # --- 2. reparent everything up to /panda ---------------------------------------------------
    layer = Sdf.Layer.FindOrOpen(dst)
    edit = Sdf.BatchNamespaceEdit()
    for name in movable_names:
        edit.Add(Sdf.NamespaceEdit.Reparent(f"{INNER_PATH}/{name}", "/panda", -1))
    assert layer.Apply(edit), "namespace reparent failed"

    # Drop the now-empty grouping xforms and the asset's own root joint.
    cleanup = Sdf.BatchNamespaceEdit()
    cleanup.Add(Sdf.NamespaceEdit.Remove(GROUP_PATH))
    if layer.GetPrimAtPath(ROOT_JOINT):
        cleanup.Add(Sdf.NamespaceEdit.Remove(ROOT_JOINT))
    assert layer.Apply(cleanup), "cleanup removal failed"
    layer.Save()
    print("reparented and removed grouping xforms + asset rootJoint")

    # --- 3. repair authored paths, then bake the old parent transform into each moved body -----
    stage = Usd.Stage.Open(dst)
    remap_paths(stage, INNER_PATH + "/", "/panda/")
    for name, m in world.items():
        prim = stage.GetPrimAtPath(f"/panda/{name}")
        assert prim, f"/panda/{name} missing after reparent"
        x = UsdGeom.Xformable(prim)
        x.ClearXformOpOrder()
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith("xformOp:"):
                prim.RemoveProperty(attr.GetName())
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        x.AddTranslateOp().Set(Gf.Vec3d(t))
        x.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
        x.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    prune_dangling(stage)
    strip_mimic_drives(stage)
    add_visual_copies(stage)
    deinstance(stage)
    ensure_scale_ops(stage)
    stage.Save()
    print("baked world transforms into moved bodies")

    # --- 4. verify -----------------------------------------------------------------------------
    stage = Usd.Stage.Open(dst)
    dp = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bad = []
    for name, m_old in world.items():
        m_new = Gf.Matrix4d(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/panda/{name}")))
        if not matrices_close(m_old, m_new, 1e-6):
            bad.append((name, m_old.ExtractTranslation(), m_new.ExtractTranslation()))
    print(f"\nworld-pose check: {len(world) - len(bad)}/{len(world)} preserved")
    for name, a, b in bad:
        print(f"  MISMATCH {name}: {a} -> {b}")

    links = [p.GetName() for p in dp.GetChildren() if p.GetTypeName() == "Xform"]
    print(f"direct Xform children (= OmniGibson links, {len(links)}): {links}")

    joints = [p for p in Usd.PrimRange(dp) if "Joint" in p.GetTypeName()]
    dangling = []
    for j in joints:
        for rel_name in ("physics:body0", "physics:body1"):
            rel = j.GetRelationship(rel_name)
            for t in (rel.GetTargets() if rel else []):
                if not stage.GetPrimAtPath(t):
                    dangling.append((j.GetName(), rel_name, t.pathString))
    print(f"joints: {len(joints)}, dangling body targets: {dangling if dangling else 'none'}")

    unbound = []
    for p in Usd.PrimRange(dp):
        for rel in p.GetRelationships():
            for t in rel.GetTargets():
                if not stage.GetPrimAtPath(t):
                    unbound.append((p.GetPath().pathString, rel.GetName(), t.pathString))
    print(f"dangling relationship targets overall: {len(unbound)}")
    for u in unbound[:5]:
        print(f"  {u}")
    assert not bad and not dangling and not unbound, "conversion produced inconsistencies -- see above"
    print("\nOK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    og.shutdown()
