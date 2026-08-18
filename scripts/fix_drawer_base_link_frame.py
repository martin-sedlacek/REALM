"""Neutralise the `base_link` frame of the impact_drawer cabinet so OG 3.9.1 places it facing the camera.

The problem
-----------
`custom_assets/impact_drawer/usd/cabinet.usd` authors a rotation on its ROOT LINK:

    /cabinet/base_link   xformOpOrder = [xformOp:transform]
                         upper 3x3 ~= diag(-1, +1, -1)      <- a 180 deg turn about the asset's
                                                               own up axis (the layer is upAxis=Y)

Nothing is wrong with that on its own -- it is how the body mesh, which was modelled facing the
other way, is brought into line with the five drawer links, which are its SIBLINGS under `/cabinet`
and so do not inherit it. The asset is internally consistent and renders correctly on OG 1.1.1.

It stops being harmless because the two OmniGibson versions place an articulation by different
prims, and the difference is exactly this transform. Writing `L` for `base_link`'s local matrix and
`C` for the pose the task config asks for:

    OG 1.1.1   `EntityPrim.set_position_orientation` stopped branch puts the ENTITY PRIM on `C`
               (`OG-lite/omnigibson/prims/entity_prim.py:1028-1029`), so the geometry -- all of it,
               body and drawers -- is drawn at `C . (asset as authored)`.

    OG 3.9.1   OG-lite_og391 commit `7c59ed5` puts the ROOT LINK on `C` instead, by dividing `L`
               out of the entity prim first (`entity_prim.py:1039-1049`):
                   position, orientation = T.mat2pose(target @ T.pose_inv(root_local))
               so the geometry is drawn at `C . L^-1 . (asset as authored)`.

`L`'s rotation is a 180 deg turn, and the config's own quaternion is ~Rx(90) -- it maps the asset's
+Y onto world +Z -- so `L^-1` lands as a **pure 180 deg yaw about world Z**: the cabinet stays
upright, stays at the config quaternion by every numeric check, keeps a drawer joint that travels
its full 0.300 m, and shows the camera its back. Measured on the live stage before this fix, task 8:
the three handle cylinders sat at world y = -0.953 with the cabinet centre at y = -1.280 and the
policy camera (`external_sensor0`) at y = -2.453, i.e. the fronts pointed +y and the camera was at
-y, `front . (cam - centre) = -0.82`.

`7c59ed5` is not the thing to revert: without it the asset does not load on stock 3.9.1 at all --
`_set_xform_properties`' round-trip assert fires, because it read the root link and wrote the entity
prim (the failure is quoted in `realm/misc/material_prim_preset_og391.patch`). Root-link placement is
also what both versions already do once the sim is playing.

The fix
-------
Take the arbitrary frame choice out of the asset instead, by re-parameterising the cabinet so that
`base_link`'s local transform is the identity and nothing else moves:

    1. `/cabinet/base_link`                              local := identity
    2. `/cabinet/base_link/Geometry_01/Object_Geometry_02`  local := local . L
       (the single Xformable prim whose nearest Xformable ancestor is `base_link`; the `Geometry_01`
        between them is a Scope and carries no transform)
    3. the five `drawer_joint_0k`, all of which have `body0 = /cabinet/base_link`:
       `physics:localPos0` / `physics:localRot0` := (that frame) . L
       `localPos1` / `localRot1` are expressed in the DRAWER link's frame, which this does not
       touch, so they are left exactly as authored.

Read as a change of coordinates this is a no-op: **every prim's transform relative to `/cabinet`,
and every joint anchor relative to both of its bodies, is bit-for-bit what it was.** That is the
invariant, and `verify()` asserts it over every Xformable prim and all ten joint frames rather than
trusting the algebra above. Two consequences follow from it:

  * On OG 1.1.1, where the entity prim is the placement target, the edited asset renders
    **identically** to the original -- `C . I . (L . mesh) == C . (L . mesh)`. The edit cannot
    regress the 1.1.1 tree, which is why it is the right place to absorb the difference.
  * On OG 3.9.1, `root_local` is now the identity, so `C . L^-1` collapses to `C` and the cabinet
    is drawn exactly where 1.1.1 draws it. `angle_to_config_quat_deg` stays 0.0000, because the
    root link is still what gets placed on `C`.

The drawer joints keep their `physics:axis = X`, their `[0, 0.3]` limits and their drive untouched;
the axis is expressed in a joint frame that moves rigidly with `base_link`, and the invariant above
is what guarantees the joint is the same constraint afterwards.

Idempotent: re-running on an already-fixed file is a no-op (`L` is already the identity), and it
says so rather than composing the rotation a second time.

    python scripts/fix_drawer_base_link_frame.py                    # in place, with a .orig backup
    python scripts/fix_drawer_base_link_frame.py --check            # report only, write nothing
    python scripts/fix_drawer_base_link_frame.py --in X --out Y
"""
import argparse
import os
import shutil
import sys

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

BASE_LINK = "/cabinet/base_link"
CABINET = "/cabinet"
TOL = 1e-9
# The float32 floor: localRot0/1 and the mesh orient are authored Quatf, so a rewritten frame
# cannot round-trip tighter than this. 1e-6 is still a hundredth of a micron on a unit-scale frame.
TOL_F32 = 1e-6


# --------------------------------------------------------------------------------------------------
# transform helpers
#
# USD composes with ROW vectors: a point is transformed as `p' = p * M`, and a prim's
# local-to-world is `M_local * M_parent_local_to_world`. Every composition below is written in that
# order, and `verify()` is what actually proves the order is right.
# --------------------------------------------------------------------------------------------------
def local_matrix(prim):
    return UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())


def matrix_from_pos_quat(pos, quat):
    """4x4 for a joint frame given as (localPos, localRot)."""
    q = Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(*[float(c) for c in quat.GetImaginary()]))
    m = Gf.Matrix4d()
    m.SetRotate(q)
    m.SetTranslateOnly(Gf.Vec3d(*[float(c) for c in pos]))
    return m


def decompose_trs(m):
    """(translate, quat, scale) from a row-vector 4x4 whose upper 3x3 is diag(scale) * R.

    That layout is what `xformOpOrder = [translate, orient, scale]` produces, verified against the
    asset's own meshes: row i of the upper 3x3 has norm scale[i].

    Uses `Gf.Transform` rather than unpacking the rows by hand -- an earlier version built the
    rotation with `r[i][j] = ...`, which silently did nothing, because indexing a `Gf.Matrix4d` row
    returns a COPY. That produced an identity quaternion for a 180 deg rotation and would have
    written a wrong asset. The round-trip check at the end is what makes that class of mistake
    impossible to ship: it recomposes the parts and requires the product to be the input.
    """
    tr = Gf.Transform(m)
    t, q, s = tr.GetTranslation(), tr.GetRotation().GetQuat(), tr.GetScale()

    check = Gf.Matrix4d(1.0)
    check.SetScale(s)
    check *= Gf.Matrix4d(1.0).SetRotate(q)
    check *= Gf.Matrix4d(1.0).SetTranslate(t)
    worst = max(abs(check[i][j] - m[i][j]) for i in range(4) for j in range(4))
    if worst > 1e-9:
        raise ValueError(f"TRS decomposition did not round-trip (max abs delta {worst:.3e})\n"
                         f"  input  {m}\n  recomposed {check}")
    return t, q, s


def rel_to_cabinet(stage, prim):
    """A prim's transform relative to `/cabinet`, i.e. with the entity prim's own op divided out.

    This is the quantity that must not change: it is the cabinet's shape, independent of wherever
    the runtime later decides to put the object.
    """
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = cache.GetLocalToWorldTransform(prim)
    c = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CABINET))
    return m * c.GetInverse()


def snapshot(stage):
    """Everything the edit must preserve: every Xformable's pose relative to /cabinet, and both
    frames of every joint expressed the same way."""
    snap = {}
    for prim in sorted(stage.Traverse(), key=lambda p: str(p.GetPath())):
        if prim.IsA(UsdGeom.Xformable):
            snap[f"xform:{prim.GetPath()}"] = rel_to_cabinet(stage, prim)
    for prim in sorted(stage.Traverse(), key=lambda p: str(p.GetPath())):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        for idx, (rel, pos_at, rot_at) in enumerate((
            (j.GetBody0Rel(), j.GetLocalPos0Attr(), j.GetLocalRot0Attr()),
            (j.GetBody1Rel(), j.GetLocalPos1Attr(), j.GetLocalRot1Attr()),
        )):
            targets = rel.GetTargets()
            if not targets:
                continue
            body = stage.GetPrimAtPath(targets[0])
            frame = matrix_from_pos_quat(pos_at.Get(), rot_at.Get())
            # The anchor in cabinet space: joint frame, then the body it is attached to.
            snap[f"joint{idx}:{prim.GetPath()}"] = frame * rel_to_cabinet(stage, body)
    return snap


def compare(before, after, rewritten):
    """Every entry must agree, to a tolerance set by how it is authored. Returns (offenders, worst).

    Two exemptions, both deliberate and neither of them slack:

      * `xform:/cabinet/base_link` is the ONE thing the fix is allowed to move -- that is the fix.
        `main()` checks it separately and exactly (it must become the identity), so dropping it here
        does not leave it unchecked.
      * anything the fix rewrote gets TOL_F32 rather than TOL, because `localRot0/1` and the mesh's
        `xformOp:orient` are authored `Quatf`. Writing a double-precision result back through a
        float32 attribute costs ~1e-8 on a unit-scale frame, so a 1e-9 gate would reject a frame
        that is rigid to a hundredth of a micron. Untouched prims are still held to TOL.
    """
    bad, worst_overall = [], 0.0
    for k in sorted(set(before) | set(after)):
        if k == f"xform:{BASE_LINK}":
            continue
        if k not in before or k not in after:
            bad.append((k, "present in only one snapshot"))
            continue
        b, a = before[k], after[k]
        worst = max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))
        worst_overall = max(worst_overall, worst)
        tol = TOL_F32 if k in rewritten else TOL
        if worst > tol:
            bad.append((k, f"max abs delta {worst:.3e} > tol {tol:g}"))
    return bad, worst_overall


# --------------------------------------------------------------------------------------------------
def apply_fix(stage, verbose=True):
    """Returns (changed, L, rewritten). `changed` is False when base_link is already the identity.

    `rewritten` is the set of snapshot keys this function re-authored; compare() holds those to the
    float32 tolerance and everything else to TOL.
    """
    rewritten = set()
    base = stage.GetPrimAtPath(BASE_LINK)
    if not base.IsValid():
        raise SystemExit(f"no prim at {BASE_LINK}")
    L = local_matrix(base)

    if Gf.IsClose(L, Gf.Matrix4d(1.0), 1e-12):
        if verbose:
            print(f"  {BASE_LINK} local transform is already the identity -- nothing to do")
        return False, L, rewritten

    if verbose:
        print(f"  L = base_link local transform:")
        for i in range(4):
            print(f"      {tuple(round(L[i][j], 9) for j in range(4))}")
        _, lq, ls = decompose_trs(L)
        print(f"      as TRS: t={tuple(round(L[3][j], 9) for j in range(3))} "
              f"quat(w,xyz)=({lq.GetReal():.9f}, {tuple(round(c, 9) for c in lq.GetImaginary())}) "
              f"scale={tuple(round(c, 9) for c in ls)}")

    # ---- 2. the one prim that must absorb L -----------------------------------------------------
    # Xformable descendants of base_link whose nearest Xformable ancestor IS base_link. Computed
    # rather than hardcoded so a re-exported asset with a different mesh layout is handled or,
    # failing that, loudly refused.
    absorbers = []
    for prim in Usd.PrimRange(base):
        if prim == base or not prim.IsA(UsdGeom.Xformable):
            continue
        anc = prim.GetParent()
        while anc and anc != base and not anc.IsA(UsdGeom.Xformable):
            anc = anc.GetParent()
        if anc == base:
            absorbers.append(prim)
    if verbose:
        print(f"  {len(absorbers)} prim(s) absorb L:")
    for prim in absorbers:
        x = UsdGeom.Xformable(prim)
        m_new = local_matrix(prim) * L
        t, q, s = decompose_trs(m_new)
        # Rewrite through the existing ops so the op order and precisions are preserved.
        ops = {op.GetOpName().split(":")[-1]: op for op in x.GetOrderedXformOps()}
        missing = {"translate", "orient", "scale"} - set(ops)
        if missing:
            raise SystemExit(f"{prim.GetPath()}: expected translate/orient/scale ops, missing {missing}")
        _set_op(ops["translate"], t)
        _set_op(ops["orient"], q)
        _set_op(ops["scale"], s)
        rewritten.add(f"xform:{prim.GetPath()}")
        if verbose:
            print(f"      {prim.GetPath()}")
            print(f"        translate -> {tuple(round(c, 9) for c in t)}")
            print(f"        orient    -> (w={q.GetReal():.9f}, xyz={tuple(round(c, 9) for c in q.GetImaginary())})")
            print(f"        scale     -> {tuple(round(c, 9) for c in s)}")

    # ---- 3. joint frames expressed in base_link's frame -----------------------------------------
    n_joint_frames = 0
    for prim in sorted(stage.Traverse(), key=lambda p: str(p.GetPath())):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        for idx, (rel, pos_at, rot_at) in enumerate((
            (j.GetBody0Rel(), j.GetLocalPos0Attr(), j.GetLocalRot0Attr()),
            (j.GetBody1Rel(), j.GetLocalPos1Attr(), j.GetLocalRot1Attr()),
        )):
            targets = rel.GetTargets()
            if not targets or str(targets[0]) != BASE_LINK:
                continue
            frame_new = matrix_from_pos_quat(pos_at.Get(), rot_at.Get()) * L
            t, q, s = decompose_trs(frame_new)
            # 1e-6, not 1e-9: localRot0/1 are authored as Quatf, so a round-tripped joint frame
            # carries float32 noise and a 1e-9 gate rejects a perfectly rigid frame.
            if not Gf.IsClose(s, Gf.Vec3d(1, 1, 1), 1e-6):
                raise SystemExit(f"{prim.GetPath()} local{idx}: joint frame picked up scale {s}")
            _set_op_attr(pos_at, t)
            _set_op_attr(rot_at, q)
            rewritten.add(f"joint{idx}:{prim.GetPath()}")
            n_joint_frames += 1
            if verbose:
                print(f"      {prim.GetPath()} local{idx} (body0={BASE_LINK})")
                print(f"        localPos{idx} -> {tuple(round(c, 9) for c in t)}")
                print(f"        localRot{idx} -> (w={q.GetReal():.9f}, xyz={tuple(round(c, 9) for c in q.GetImaginary())})")
    if verbose:
        print(f"  {n_joint_frames} joint frame(s) rewritten")

    # ---- 1. base_link itself --------------------------------------------------------------------
    # Done LAST: rel_to_cabinet() above reads through base_link, so zeroing it first would corrupt
    # every composition in this function.
    x = UsdGeom.Xformable(base)
    ops = {op.GetOpName().split(":")[-1]: op for op in x.GetOrderedXformOps()}
    if "transform" in ops:
        ops["transform"].Set(Gf.Matrix4d(1.0))
    else:
        raise SystemExit(f"{BASE_LINK}: expected a single xformOp:transform, got {list(ops)}")
    if verbose:
        print(f"  {BASE_LINK} xformOp:transform -> identity")
    return True, L, rewritten


def _set_op(op, value):
    """Write @value through an xformOp, matching the op's authored precision."""
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            op.Set(Gf.Quatf(float(value.GetReal()), Gf.Vec3f(*[float(c) for c in value.GetImaginary()])))
        else:
            op.Set(Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*[float(c) for c in value.GetImaginary()])))
    else:
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            op.Set(Gf.Vec3f(*[float(c) for c in value]))
        else:
            op.Set(Gf.Vec3d(*[float(c) for c in value]))


def _set_op_attr(attr, value):
    """Write a joint localPos/localRot, matching the attribute's authored type."""
    tn = attr.GetTypeName()
    if isinstance(value, Gf.Quatd) or isinstance(value, Gf.Quatf):
        if tn == Sdf.ValueTypeNames.Quatf:
            attr.Set(Gf.Quatf(float(value.GetReal()), Gf.Vec3f(*[float(c) for c in value.GetImaginary()])))
        else:
            attr.Set(Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*[float(c) for c in value.GetImaginary()])))
    else:
        if tn == Sdf.ValueTypeNames.Point3f or tn == Sdf.ValueTypeNames.Float3:
            attr.Set(Gf.Vec3f(*[float(c) for c in value]))
        else:
            attr.Set(Gf.Vec3d(*[float(c) for c in value]))


def _angle_deg(m):
    """Rotation angle of a rigid 4x4, in degrees, for the log line."""
    return float(Gf.Transform(m).GetRotation().GetAngle())


def main():
    ap = argparse.ArgumentParser()
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "custom_assets/impact_drawer/usd/cabinet.usd")
    ap.add_argument("--in", dest="src", default=default)
    ap.add_argument("--out", dest="dst", default=None, help="default: in place")
    ap.add_argument("--check", action="store_true", help="report and verify, write nothing")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()
    dst = a.dst or a.src

    print(f"asset: {a.src}")
    stage = Usd.Stage.Open(a.src)
    before = snapshot(stage)
    print(f"snapshot: {len(before)} invariants recorded "
          f"({sum(1 for k in before if k.startswith('xform:'))} xformable, "
          f"{sum(1 for k in before if k.startswith('joint'))} joint frames)")

    changed, L, rewritten = apply_fix(stage, verbose=True)
    if not changed:
        return 0

    after = snapshot(stage)
    bad, worst = compare(before, after, rewritten)
    print(f"\nverify: {len(after)} invariants re-checked "
          f"(tol {TOL:g}, {TOL_F32:g} for the {len(rewritten)} rewritten); worst delta {worst:.3e}")
    if bad:
        print(f"FAILED -- {len(bad)} invariant(s) moved:")
        for k, why in bad[:20]:
            print(f"    {k}: {why}")
        print("Nothing written.")
        return 1
    print("  PASSED -- every prim pose and joint anchor relative to /cabinet is unchanged")

    # The one intended change, checked exactly rather than exempted and forgotten.
    base_local = local_matrix(stage.GetPrimAtPath(BASE_LINK))
    if not Gf.IsClose(base_local, Gf.Matrix4d(1.0), 1e-12):
        print(f"FAILED -- {BASE_LINK} local is not the identity:\n{base_local}")
        return 1
    print(f"  {BASE_LINK} local is exactly the identity (was a {_angle_deg(L):.3f} deg rotation)")

    if a.check:
        print("\n--check: not writing.")
        return 0

    if dst == a.src and not a.no_backup:
        backup = a.src + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(a.src, backup)
            print(f"\nbackup: {backup}")
    stage.GetRootLayer().Export(dst)
    print(f"wrote: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
