"""Make the `panda_link8` gripper-adapter pad visible in OmniGibson renders, without touching physics.

The problem
-----------
`droid_robolab_v2.usd` carries the adapter pad -- the ~75 mm x 18 mm disc between the end of the arm
and the start of the Robotiq gripper -- as two coincident cylinders under `/panda/panda_link8`:

    gripper_adapter      Cylinder, PhysicsCollisionAPI + PhysicsMassAPI   <- the collider
    gripper_adapter_01   Cylinder, PhysicsMassAPI only                    <- a render-only twin

That is the standard OmniGibson trick -- `RigidPrim.update_meshes()` hides every collision geom by
setting `purpose = "guide"`, so a collider-free duplicate is what actually draws -- and OmniGibson
does classify the twin correctly, as a **visual** mesh with `purpose = "default"`.

It still does not render, and the reason is not the collider property at all:

    omnigibson/robots/robot.py:1255
        if self.is_manipulation:
            for arm in self.arm_names:
                self._links[self.eef_link_names[arm]].visible = False

    realm/robots/definitions/droid_robolab_v2/droid_robolab_v2.yaml:99
        eef_link_names: {"0": "panda_link8"}

`panda_link8` IS the eef link, so OmniGibson calls `MakeInvisible()` on the link prim itself for
every manipulation robot. USD visibility **prunes**: once an ancestor is `invisible`, no descendant
can override it, and `purpose` is orthogonal to visibility. Measured on the loaded robot, every
child of `panda_link8` -- both cylinders, the `contact_frame` Xform, and the `panda_hand_joint`
that no mesh code ever touches -- computes `visibility = invisible`, while every geom on all
sixteen other links computes visible. So authoring `purpose = "render"` on the twin, or an explicit
`visibility`, or a material, or renaming it to the `visual__` convention, or converting the
`Cylinder` to a `Mesh`, are all downstream of the prune and cannot work.

The fix
-------
Move the render-only twin onto a link that is *not* pruned, keeping its world pose exactly.
`/panda/base_link` is the Robotiq gripper base: it is joined to `panda_link8` by the fixed
`panda_hand_joint`, so a prim under it tracks the flange rigidly, and it is visible because it
draws the gripper. The twin is reparented there as `visual__gripper_adapter` -- the same naming the
converter uses for every other render-only copy in this asset -- with a local transform of
`T_twin_world . T_base_world^-1`.

Physics cannot move, and that is checked rather than asserted:

  * the twin has no collision API of any kind, so it is not a collider before or after;
  * `RigidPrim.update_meshes()` accumulates volume and centre of mass from **collision** meshes
    only, and `RigidPrim.aabb` from `collision_boundary_points_world`, so a visual-only prim
    contributes to neither;
  * its `PhysicsMassAPI` authors no attributes at all (an empty schema application), and
    `panda_link8` authors its own link-level `physics:mass`, which PhysX uses in preference to any
    shape-derived value. The API is dropped anyway so the moved prim is purely visual, matching
    every other `visual__` copy in the file.

Usage:
    python scripts/fix_link8_adapter_visual.py realm/robots/panda_robotiq/droid_robolab_v2.usd
    python scripts/fix_link8_adapter_visual.py <src.usd> --out <dst.usd>     # leave the source alone

Runs on plain `usd-core` -- no GPU, no container, no OmniGibson -- and falls back to
`omnigibson.lazy.pxr` when run inside the image.
"""

import argparse
import hashlib
import shutil
import sys

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
except ImportError:  # inside the REALM image, pxr is only reachable through a launched Kit app
    import omnigibson as og

    og.launch()
    import omnigibson.lazy as lazy

    Usd, UsdGeom, UsdPhysics, Sdf, Gf = (
        lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics, lazy.pxr.Sdf, lazy.pxr.Gf)

SRC_PATH = "/panda/panda_link8/gripper_adapter_01"
DST_PARENT = "/panda/base_link"
DST_NAME = "visual__gripper_adapter"
DST_PATH = f"{DST_PARENT}/{DST_NAME}"

COLLISION_APIS = ("PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI")
#: Every property whose value decides how the body behaves. The audit below compares these across
#: the whole file, not just the prims this script touches -- a namespace edit can silently retarget
#: a relationship several links away.
PHYSICS_PREFIXES = ("physics:", "physx", "drive:", "limit:", "state:")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(path):
    """Everything that must not change: schemas, authored properties, and world transforms.

    Keyed by prim path so the two sides can be diffed key-wise, and taken with the stage closed
    afterwards so no Usd handle outlives it (they dangle, and dangling handles segfault).
    """
    stage = Usd.Stage.Open(path)
    dp = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims, props, rels, xf = {}, {}, {}, {}
    for p in Usd.PrimRange(dp):
        sp = str(p.GetPath())
        prims[sp] = (str(p.GetTypeName()), tuple(sorted(p.GetAppliedSchemas())))
        for a in p.GetAttributes():
            if a.HasAuthoredValue():
                props[(sp, a.GetName())] = repr(a.Get())
        for r in p.GetRelationships():
            t = r.GetTargets()
            if t:
                rels[(sp, r.GetName())] = tuple(str(x) for x in t)
        if UsdGeom.Xformable(p):
            m = cache.GetLocalToWorldTransform(p)
            xf[sp] = tuple(round(m[i][j], 12) for i in range(4) for j in range(4))
    del cache, dp, stage
    return {"prims": prims, "props": props, "rels": rels, "xf": xf}


def decompose(m):
    """Row-vector Matrix4d -> (translate, quat, scale) for xformOpOrder [translate, orient, scale].

    USD composes those ops as p' = p . S . R . T, so the upper 3x3's rows are the rotation's rows
    scaled per-axis. Normalising them before extracting the quaternion matters: ExtractRotationQuat
    on a matrix that still carries a 0.018 z-scale does not return a unit rotation.
    """
    rows = [Gf.Vec3d(m[i][0], m[i][1], m[i][2]) for i in range(3)]
    scale = [r.GetLength() for r in rows]
    assert all(s > 1e-12 for s in scale), f"degenerate scale {scale}"
    n = [rows[i] / scale[i] for i in range(3)]
    # Build the rotation by CONSTRUCTING a Matrix3d, never by assigning into `rot[i][j]`: the
    # Python binding hands back a copy of the row, so element assignment writes to a temporary and
    # is silently lost -- which yields an identity rotation and a pose error of exactly one scale
    # factor. The round-trip assert below is what caught that.
    m3 = Gf.Matrix3d(n[0][0], n[0][1], n[0][2],
                     n[1][0], n[1][1], n[1][2],
                     n[2][0], n[2][1], n[2][2])
    t, q = m.ExtractTranslation(), Gf.Matrix4d(m3, Gf.Vec3d(0, 0, 0)).ExtractRotationQuat()

    recomposed = (Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))
                  * Gf.Matrix4d().SetRotate(q)
                  * Gf.Matrix4d().SetTranslate(t))
    err = max(abs(recomposed[i][j] - m[i][j]) for i in range(4) for j in range(4))
    assert err < 1e-9, f"decomposition does not round-trip: max |delta| = {err:.3e}"
    return t, q, scale


def physics_delta(before, after):
    """Physics-relevant differences only, as (kind, key, old, new) tuples."""
    out = []
    for sp, (t, schemas) in before["prims"].items():
        if sp not in after["prims"]:
            out.append(("prim_removed", sp, f"{t} {schemas}", None))
        elif after["prims"][sp] != (t, schemas):
            out.append(("prim_changed", sp, f"{t} {schemas}", str(after["prims"][sp])))
    for sp, v in after["prims"].items():
        if sp not in before["prims"]:
            out.append(("prim_added", sp, None, str(v)))
    for k, v in before["props"].items():
        if not k[1].startswith(PHYSICS_PREFIXES):
            continue
        if k not in after["props"]:
            out.append(("prop_removed", f"{k[0]} .{k[1]}", v, None))
        elif after["props"][k] != v:
            out.append(("prop_changed", f"{k[0]} .{k[1]}", v, after["props"][k]))
    for k, v in after["props"].items():
        if k[1].startswith(PHYSICS_PREFIXES) and k not in before["props"]:
            out.append(("prop_added", f"{k[0]} .{k[1]}", None, v))
    for k, v in before["rels"].items():
        if after.get("rels", {}).get(k) != v:
            out.append(("rel_changed", f"{k[0]} .{k[1]}", str(v), str(after["rels"].get(k))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the robolab v2 USD")
    ap.add_argument("--out", default=None, help="write here instead of editing src in place")
    ap.add_argument("--backup", default=None, help="copy src here before editing in place")
    args = ap.parse_args()

    dst = args.out or args.src
    print(f"src {args.src}  md5 {md5(args.src)}")
    if args.backup:
        shutil.copyfile(args.src, args.backup)
        print(f"backup {args.backup}  md5 {md5(args.backup)}")
    if dst != args.src:
        shutil.copyfile(args.src, dst)
        print(f"copied -> {dst}")

    before = snapshot(args.src)

    # --- 1. is there anything to do? ---------------------------------------------------------
    stage = Usd.Stage.Open(dst)
    if stage.GetPrimAtPath(DST_PATH):
        print(f"{DST_PATH} already exists -- nothing to do")
        del stage
        return 0
    src_prim = stage.GetPrimAtPath(SRC_PATH)
    assert src_prim, (f"{SRC_PATH} not found. This asset does not carry the render-only twin, so "
                      f"there is nothing to reparent -- see the module docstring.")
    assert not any(a in src_prim.GetAppliedSchemas() for a in COLLISION_APIS), (
        f"{SRC_PATH} carries a collision API {src_prim.GetAppliedSchemas()} -- it is a COLLIDER, "
        f"not the render-only twin. Moving it would change physics. Refusing.")
    base = stage.GetPrimAtPath(DST_PARENT)
    assert base and base.HasAPI(UsdPhysics.RigidBodyAPI), f"{DST_PARENT} is not a rigid body"

    # World poses are captured now, as value copies, because they are what the reparent has to
    # preserve and because Usd handles do not survive the stage being dropped.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    m_twin = Gf.Matrix4d(cache.GetLocalToWorldTransform(src_prim))
    m_base = Gf.Matrix4d(cache.GetLocalToWorldTransform(base))
    m_local = m_twin * m_base.GetInverse()
    print(f"twin world T {tuple(round(v, 9) for v in m_twin.ExtractTranslation())}")
    print(f"base world T {tuple(round(v, 9) for v in m_base.ExtractTranslation())}")
    del cache, src_prim, base, stage      # drop the stage before editing the layer's namespace

    # --- 2. reparent + rename -----------------------------------------------------------------
    layer = Sdf.Layer.FindOrOpen(dst)
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(Sdf.NamespaceEdit.ReparentAndRename(SRC_PATH, DST_PARENT, DST_NAME, -1))
    assert layer.Apply(edit), f"reparent {SRC_PATH} -> {DST_PATH} failed"
    layer.Save()
    print(f"reparented {SRC_PATH} -> {DST_PATH}")

    # --- 3. rebake the local transform, drop the inert mass API --------------------------------
    stage = Usd.Stage.Open(dst)
    prim = stage.GetPrimAtPath(DST_PATH)
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    for a in list(prim.GetAttributes()):
        if a.GetName().startswith("xformOp:"):
            prim.RemoveProperty(a.GetName())
    t, q, s = decompose(m_local)
    x.AddTranslateOp().Set(Gf.Vec3d(t))
    x.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    x.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in s]))
    print(f"local T {tuple(round(v, 9) for v in t)} scale {tuple(round(v, 9) for v in s)}")

    # An empty PhysicsMassAPI on a render-only prim authors nothing and is read by nothing, but it
    # is the one thing on this prim that could ever be mistaken for a physics contribution. Every
    # other visual__ copy in this asset carries no physics API at all; match them.
    if prim.HasAPI(UsdPhysics.MassAPI):
        prim.RemoveAPI(UsdPhysics.MassAPI)
        print("removed PhysicsMassAPI from the moved prim")
    stage.Save()
    del prim, x, stage

    # --- 4. verify ------------------------------------------------------------------------------
    after = snapshot(dst)
    ok = True

    m_new = Gf.Matrix4d(*after["xf"][DST_PATH])
    pose_err = max(abs(m_new[i][j] - m_twin[i][j]) for i in range(4) for j in range(4))
    print(f"\nworld-pose preserved: max |delta| = {pose_err:.3e}")
    ok &= pose_err < 1e-9

    moved = {SRC_PATH, DST_PATH}
    unmoved_bad = [k for k, v in before["xf"].items()
                   if k not in moved and (k not in after["xf"] or after["xf"][k] != v)]
    print(f"every other prim's world transform unchanged: {len(before['xf']) - 1 - len(unmoved_bad)}"
          f"/{len(before['xf']) - 1}")
    ok &= not unmoved_bad
    for k in unmoved_bad[:5]:
        print(f"  MOVED {k}")

    delta = physics_delta(before, after)
    expected = {("prim_removed", SRC_PATH), ("prim_added", DST_PATH)}
    unexpected = [d for d in delta if (d[0], d[1]) not in expected]
    print(f"\nphysics-relevant differences: {len(delta)} total, {len(unexpected)} unexpected")
    for d in delta:
        mark = "  " if (d[0], d[1]) in expected else "!!"
        print(f"{mark} {d[0]:14s} {d[1]}\n       {d[2]}\n    -> {d[3]}")
    ok &= not unexpected

    stage = Usd.Stage.Open(dst)
    l8 = [c.GetName() for c in stage.GetPrimAtPath("/panda/panda_link8").GetChildren()]
    newp = stage.GetPrimAtPath(DST_PATH)
    print(f"\npanda_link8 children now: {l8}")
    print(f"{DST_PATH} schemas: {list(newp.GetAppliedSchemas())}")
    ok &= not any(a in newp.GetAppliedSchemas() for a in COLLISION_APIS)
    ok &= "gripper_adapter" in l8 and "gripper_adapter_01" not in l8
    dangling = [(str(p.GetPath()), r.GetName(), str(t))
                for p in Usd.PrimRange(stage.GetDefaultPrim())
                for r in p.GetRelationships() for t in r.GetTargets()
                if not stage.GetPrimAtPath(t)]
    print(f"dangling relationship targets: {dangling if dangling else 'none'}")
    ok &= not dangling
    del newp, stage

    print(f"\ndst {dst}  md5 {md5(dst)}")
    print("\nOK" if ok else "\nFAILED -- see the marked lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
