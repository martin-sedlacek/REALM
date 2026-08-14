"""Build an "xform-flattened" variant of the robolab Robotiq 2F-85 asset.

Why
---
`RigidPrim.update_meshes()` (omnigibson/prims/rigid_prim.py) derives each link's centre of mass by
composing every collision geom's pose with

    local_pos, local_orn = mesh.get_position_orientation(frame="parent")

whose own comment says "transform ... to the LINK's local frame". `frame="parent"` is documented in
`xform_prim.py` as "get position relative to the object **parent**" -- the IMMEDIATE parent prim.
That equals the link frame only when the geom is a direct child of the link.

In this asset it never is. Every collision API sits on an intermediate `Defeatured_*_01` **Xform**
with the `Mesh` beneath it, and `GEOM_TYPES = {Sphere, Cube, Cone, Cylinder, Mesh}` excludes Xform,
so the geom OmniGibson wraps is the Mesh and the Xform -> link step is silently dropped. That Xform
carries the left/right MIRROR (90 deg on the left, 180 deg on the right), so both pads come out with
an IDENTICAL computed CoM including the sign of y -- impossible for a mirrored pair -- landing
128.347 mm from the true centroid. PhysX derives all mass properties from the collision shapes
(neither this USD nor RoboLab's authors any), so that displacement enters each pad's inertia about
its own pivot as m*d^2 = 1.57e-4 kg m^2 against a true ~1.9e-6: a 77x inflation. A PhysX mimic joint
realises stiffness k ~ omega^2 * I, so at the authored `naturalFrequency 1000` the fingertips are
~77x too stiff and will not curl under load.

What this changes
-----------------
Nothing about the physics, the joints, the mass, or where any surface is. Purely the LEVEL at which
the collision transform is authored:

    BEFORE                                          AFTER
    link                                            link
     +- Defeatured_*_01  [Xform, CollisionAPI]        +- Defeatured_*_01  [Xform, CollisionAPI]
        translate = t                                   translate = (0,0,0)      <- identity
        orient    = q                                   orient    = identity     <- identity
        scale     = (1,1,1)                             scale     = (1,1,1)
        +- Defeatured_*  [Mesh]                         +- Defeatured_*  [Mesh]
           scale = (1,1,1)                                 translate = t         <- moved down
                                                           orient    = q         <- moved down
                                                           scale = (1,1,1)

`t` and `q` are transplanted VERBATIM -- the same double3 / quatd values, no arithmetic, no matrix
decomposition -- which is possible because in this asset every such Mesh carries *only* an
`xformOp:scale` of exactly (1,1,1) and every `_01` Xform's scale is exactly (1,1,1) too. Both
preconditions are asserted, so a future asset that violates them fails loudly instead of being
silently re-derived through a lossy decomposition.

The composed world transform of every collision surface is therefore unchanged -- verified below by
transforming every mesh POINT to world space and comparing against the source, not just by comparing
the matrices. Contact, rendering and the collision hulls are untouched.

The intermediate Xform is left in place (rather than the Mesh being reparented under the link) so
that every prim path, every material binding into `_01/Looks/...`, every GeomSubset and the
CollisionAPI/MeshCollisionAPI placement stay exactly where they were. With `_01` at identity the
dropped Xform -> link step IS the identity, so OmniGibson's `frame="parent"` composition returns the
link-frame pose by construction and the bug cannot fire. **This needs no OmniGibson patch and works
against a completely stock loader** -- that is the point of the route.

The ARM IS NOT TOUCHED: only the nine gripper links are visited, and the writer re-opens what it
wrote and compares every authored attribute on `panda_joint1..7` and on every `panda_link*` prim
against the source (grep XFLAT_ARM_IDENTICAL).

Usage
-----
Inside the container (authoritative -- writes a crate file with the runtime's own USD version):

    ./scripts/clara/interactive/rr python -u /app/scripts/make_xflat_gripper_usd.py

On the host with `usd-core` from PyPI, for a dry run / re-verification of a file built elsewhere:

    <venv>/bin/python scripts/make_xflat_gripper_usd.py --dst /tmp/x.usd
    <venv>/bin/python scripts/make_xflat_gripper_usd.py --verify-only --dst <built.usd>
"""

import argparse
import math
import os
import shutil
import sys

DEFAULT_SRC = "/app/realm/robots/panda_robotiq/droid_robolab_v2.usd"
DEFAULT_DST = "/app/realm/robots/panda_robotiq/droid_robolab_xflat.usd"

# The nine gripper links, and ONLY these. base_link here is the 2F-85's own base (fixed to
# panda_link8), not the arm's panda_link0.
GRIPPER_LINKS = (
    "base_link",
    "left_outer_knuckle", "left_outer_finger", "left_inner_finger", "left_inner_knuckle",
    "right_outer_knuckle", "right_outer_finger", "right_inner_finger", "right_inner_knuckle",
)
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
# omnigibson.utils.constants.GEOM_TYPES -- the prim types RigidPrim.update_meshes() will wrap.
GEOM_TYPES = {"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=DEFAULT_SRC)
ap.add_argument("--dst", default=DEFAULT_DST)
ap.add_argument("--verify-only", action="store_true",
                help="do not write; just re-verify an existing --dst against --src")
ap.add_argument("--tol-world", type=float, default=1e-12,
                help="max allowed world-space displacement of any collision mesh POINT (metres)")
args = ap.parse_args()

# pxr is importable directly from a `usd-core` venv on the host; inside the REALM container it only
# exists once a Kit app is running, which og.launch() provides.
try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: F401
    _og = None
except ImportError:
    import omnigibson as og
    og.launch()
    import omnigibson.lazy as lazy
    Gf, Usd, UsdGeom, UsdPhysics = lazy.pxr.Gf, lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics
    _og = og


def collision_geoms(link):
    """Every prim under @link that RigidPrim.update_meshes() will treat as a COLLISION geom.

    Mirrors `_find_geom_prims`: descend the subtree, latch `is_collision` the moment a prim carrying
    UsdPhysics.CollisionAPI is seen, and keep the descendants whose type is in GEOM_TYPES.
    """
    out = []

    def rec(prim, is_collision):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            is_collision = True
        if prim.GetPrimTypeInfo().GetTypeName() in GEOM_TYPES and is_collision:
            out.append(prim)
        for child in prim.GetChildren():
            rec(child, is_collision)

    rec(link, False)
    return out


def op_map(prim):
    """{op name: op} for @prim's ordered xform ops."""
    return {o.GetOpName(): o for o in UsdGeom.Xformable(prim).GetOrderedXformOps()}


def is_identity_scale(v):
    return v is not None and tuple(float(x) for x in v) == (1.0, 1.0, 1.0)


def flatten_geom(link, geom):
    """Push the link -> geom.parent chain down onto @geom and identity out the intermediates.

    Returns a dict describing what moved. Asserts the preconditions that make the transplant EXACT.
    """
    chain = []                                   # intermediate prims, link-side first
    p = geom.GetParent()
    while p != link:
        assert p.IsValid() and p.GetPath() != p.GetParent().GetPath(), f"{geom.GetPath()} is not under {link.GetPath()}"
        chain.append(p)
        p = p.GetParent()
    chain.reverse()

    if not chain:
        return dict(geom=str(geom.GetPath()), moved=False, why="already a direct child of the link")

    # --- preconditions for a verbatim (arithmetic-free) transplant ------------------------------
    assert len(chain) == 1, (
        f"{geom.GetPath()}: {len(chain)} intermediate prims; the verbatim transplant only handles "
        f"one. Composing several would need a matrix decomposition -- refusing to do it silently.")
    xf = chain[0]
    xops, gops = op_map(xf), op_map(geom)
    assert set(xops) == {"xformOp:translate", "xformOp:orient", "xformOp:scale"}, \
        f"{xf.GetPath()}: unexpected xform ops {sorted(xops)}"
    assert set(gops) <= {"xformOp:scale"}, \
        f"{geom.GetPath()}: already carries {sorted(set(gops) - {'xformOp:scale'})}; the transplant " \
        f"would have to COMPOSE rather than assign, which is not arithmetic-free"
    assert is_identity_scale(xops["xformOp:scale"].Get()), \
        f"{xf.GetPath()}: scale {xops['xformOp:scale'].Get()} != (1,1,1); a non-unit intermediate " \
        f"scale does not commute past the geom's own ops"
    assert not UsdGeom.Xformable(geom).GetResetXformStack() and not UsdGeom.Xformable(xf).GetResetXformStack(), \
        "resetXformStack is set; the composition is not what this script assumes"

    t, q = xops["xformOp:translate"].Get(), xops["xformOp:orient"].Get()

    # --- move them down, VERBATIM ---------------------------------------------------------------
    gx = UsdGeom.Xformable(geom)
    gx.ClearXformOpOrder()
    t_op = gx.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(t)
    q_op = gx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    q_op.Set(q)
    if "xformOp:scale" in gops:                              # keep the geom's own scale, last
        s_op = gx.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
        s_op.Set(gops["xformOp:scale"].Get())

    # --- and identity out the intermediate ------------------------------------------------------
    xops["xformOp:translate"].Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xops["xformOp:orient"].Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))

    return dict(geom=str(geom.GetPath()), xform=str(xf.GetPath()), moved=True,
                translate=tuple(float(v) for v in t),
                orient=(float(q.GetReal()),) + tuple(float(v) for v in q.GetImaginary()))


# ------------------------------------------------------------------------------------------------
# verification
# ------------------------------------------------------------------------------------------------
def attr_snapshot(prim):
    """Every AUTHORED attribute on @prim, as {name: repr}. Same shape as make_curlgrip's."""
    return {a.GetName(): str(a.Get()) for a in prim.GetAttributes() if a.HasAuthoredValue()}


def subtree_snapshot(prim):
    return {str(p.GetPath()): attr_snapshot(p) for p in Usd.PrimRange(prim)}


def world_points(prim, cache):
    """@prim's `points` transformed to world space, or None if it has none."""
    pts = prim.GetAttribute("points")
    if not pts.IsValid() or pts.Get() is None:
        return None
    M = cache.GetLocalToWorldTransform(prim)
    return [M.Transform(p) for p in pts.Get()]


def mat_delta(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


def verify(src_path, dst_path, tol_world):
    src, dst = Usd.Stage.Open(src_path), Usd.Stage.Open(dst_path)
    sc, dc = UsdGeom.XformCache(), UsdGeom.XformCache()
    ok = True

    # 1. GEOMETRIC NEUTRALITY. Every collision surface must land in an identical WORLD pose. The
    #    matrices are compared, and then so is every individual mesh point -- a matrix compare alone
    #    would miss an op-order mistake that happens to cancel in the decomposition.
    print("\n[1] GEOMETRIC NEUTRALITY -- collision geometry in world space")
    worst_mat, worst_pt, npts, ngeom = 0.0, 0.0, 0, 0
    for lname in GRIPPER_LINKS:
        slink, dlink = src.GetPrimAtPath("/panda/" + lname), dst.GetPrimAtPath("/panda/" + lname)
        sgeoms, dgeoms = collision_geoms(slink), collision_geoms(dlink)
        assert [g.GetPath() for g in sgeoms] == [g.GetPath() for g in dgeoms], \
            f"{lname}: the set of collision geoms changed"
        for sg, dg in zip(sgeoms, dgeoms):
            ngeom += 1
            dm = mat_delta(sc.GetLocalToWorldTransform(sg), dc.GetLocalToWorldTransform(dg))
            sp, dp = world_points(sg, sc), world_points(dg, dc)
            pm = 0.0
            if sp is not None and dp is not None:
                assert len(sp) == len(dp), f"{sg.GetPath()}: point count changed"
                npts += len(sp)
                pm = max((a - b).GetLength() for a, b in zip(sp, dp))
            worst_mat, worst_pt = max(worst_mat, dm), max(worst_pt, pm)
            flag = "OK" if pm <= tol_world else "*** MOVED ***"
            print(f"    {lname:<20} {sg.GetName():<40} dM={dm:.3e}  max|dP|={pm:.3e} m  {flag}")
            ok &= pm <= tol_world
    print(f"    {ngeom} collision geoms, {npts} points compared. "
          f"worst matrix delta {worst_mat:.3e}, worst point displacement {worst_pt:.3e} m "
          f"(tolerance {tol_world:.0e})")

    # 2. THE BUG IS NOW UNREACHABLE. For every collision geom, what update_meshes() reads
    #    (frame="parent") must equal what it MEANT to read (the link frame).
    print("\n[2] DROPPED STEP -- parent-relative vs link-relative, per collision geom")
    worst_before, worst_after = 0.0, 0.0
    for stage, cache, label in ((src, sc, "src"), (dst, dc, "dst")):
        for lname in GRIPPER_LINKS:
            link = stage.GetPrimAtPath("/panda/" + lname)
            Minv = cache.GetLocalToWorldTransform(link).GetInverse()
            for g in collision_geoms(link):
                Mparent = cache.GetLocalTransformation(g)[0]
                Mlink = cache.GetLocalToWorldTransform(g) * Minv
                drop = Mparent.GetInverse() * Mlink
                dt = drop.ExtractTranslation().GetLength()
                dq = drop.ExtractRotationQuat()
                ang = 2.0 * math.degrees(math.acos(min(1.0, abs(dq.GetReal()))))
                if label == "src":
                    worst_before = max(worst_before, dt)
                    print(f"    src {lname:<20} {g.GetName():<40} |t|={dt * 1000:9.4f} mm  ang={ang:8.4f} deg")
                else:
                    worst_after = max(worst_after, dt)
                    print(f"    dst {lname:<20} {g.GetName():<40} |t|={dt * 1000:9.4f} mm  ang={ang:8.4f} deg"
                          + ("" if (dt < 1e-12 and ang < 1e-6) else "   *** STILL DROPPED ***"))
                    ok &= (dt < 1e-12 and ang < 1e-6)
    print(f"    worst dropped translation  BEFORE {worst_before * 1000:.4f} mm  ->  AFTER {worst_after * 1000:.3e} mm")

    # 3. THE ARM MUST BE UNTOUCHED. Every authored attribute on the seven arm joints and on the
    #    whole subtree of every panda_link* prim.
    print("\n[3] ARM PHYSICS -- every authored attribute vs the source")
    arm_ok, n_attr = True, 0
    sj = {p.GetName(): p for p in Usd.PrimRange(src.GetDefaultPrim()) if "Joint" in p.GetTypeName()}
    dj = {p.GetName(): p for p in Usd.PrimRange(dst.GetDefaultPrim()) if "Joint" in p.GetTypeName()}
    for name in ARM_JOINTS:
        assert name in sj and name in dj, f"{name} missing from one of the stages"
        a, b = attr_snapshot(sj[name]), attr_snapshot(dj[name])
        n_attr += len(a)
        if a != b:
            arm_ok = False
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    print(f"      *** {name}.{k}: src={a.get(k)!r} dst={b.get(k)!r}")
        print(f"    {name:<16} {len(a):>3} authored attributes  {'IDENTICAL' if a == b else '*** DIFFERS ***'}")
    n_links = 0
    for sp in src.GetDefaultPrim().GetChildren():
        if not sp.GetName().startswith("panda_link"):
            continue
        n_links += 1
        dp = dst.GetPrimAtPath(sp.GetPath())
        sa, da = subtree_snapshot(sp), subtree_snapshot(dp)
        n_attr += sum(len(v) for v in sa.values())
        if sa != da:
            arm_ok = False
            for k in sorted(set(sa) | set(da)):
                if sa.get(k) != da.get(k):
                    print(f"      *** {k} DIFFERS")
    print(f"    {n_links} panda_link* subtrees compared: {'all identical' if arm_ok else '*** SOME DIFFER ***'}")
    print(f"    {n_attr} arm attributes compared in total")
    ok &= arm_ok

    # 4. NOTHING ELSE MOVED. Whole-stage sweep: every authored attribute must match except the
    #    xformOps on exactly the prims this script is allowed to touch.
    print("\n[4] WHOLE-STAGE SWEEP -- what else changed")
    allowed = set()
    for lname in GRIPPER_LINKS:
        for g in collision_geoms(dst.GetPrimAtPath("/panda/" + lname)):
            allowed.add(str(g.GetPath()))
            allowed.add(str(g.GetParent().GetPath()))
    changed, unexpected = [], []
    sall, dall = subtree_snapshot(src.GetDefaultPrim()), subtree_snapshot(dst.GetDefaultPrim())
    assert set(sall) == set(dall), f"prim set changed: {sorted(set(sall) ^ set(dall))[:10]}"
    for path in sorted(sall):
        if sall[path] == dall[path]:
            continue
        keys = sorted(set(sall[path]) | set(dall[path]))
        diffkeys = [k for k in keys if sall[path].get(k) != dall[path].get(k)]
        changed.append((path, diffkeys))
        if path not in allowed or any(not k.startswith("xformOp") for k in diffkeys):
            unexpected.append((path, diffkeys))
    for path, keys in changed:
        mark = "*** UNEXPECTED ***" if (path, keys) in unexpected else ""
        print(f"    {path}  {keys} {mark}")
    print(f"    {len(sall)} prims compared; {len(changed)} changed, {len(unexpected)} unexpected")
    ok &= not unexpected

    print(f"\nXFLAT_ARM_{'IDENTICAL' if arm_ok else 'CHANGED'}")
    print(f"XFLAT_USD_{'OK' if ok else 'FAIL'} {dst_path}")
    return ok


def main():
    assert os.path.isfile(args.src), f"no source asset at {args.src}"
    assert os.path.abspath(args.src) != os.path.abspath(args.dst), \
        "refusing to edit the shipped asset in place -- pass a different --dst"

    if not args.verify_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
        shutil.copyfile(args.src, args.dst)
        print(f"copied {args.src}\n    -> {args.dst}")

        stage = Usd.Stage.Open(args.dst)
        total = 0
        for lname in GRIPPER_LINKS:
            link = stage.GetPrimAtPath("/panda/" + lname)
            assert link.IsValid(), f"/panda/{lname} not found -- is this the right source asset?"
            geoms = collision_geoms(link)
            assert geoms, f"/panda/{lname} has no collision geom"
            print(f"\n[{lname}] {len(geoms)} collision geom(s)")
            for g in geoms:
                info = flatten_geom(link, g)
                total += bool(info["moved"])
                if info["moved"]:
                    print(f"    {info['xform']}")
                    print(f"        -> {g.GetName()}  translate={info['translate']}  orient={info['orient']}")
                else:
                    print(f"    {g.GetName()}: {info['why']}")
        print(f"\nflattened {total} collision geom(s) across {len(GRIPPER_LINKS)} gripper links")
        stage.Save()

    ok = verify(args.src, args.dst, args.tol_world)
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    if _og is not None:
        _og.shutdown()
    sys.exit(rc)
