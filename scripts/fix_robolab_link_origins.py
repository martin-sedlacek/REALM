"""Move robolab gripper body ORIGINS onto their own geometry, matching droid.usd's convention.

Why
---
REALM's evaluation measures from link origins, not from geometry. `check_reach_condition`
(env_base.py:206) gates on ||block - finger_link.get_position_orientation()|| < 0.1 m, and
`recompute_task_progression` breaks at the first unmet stage -- so a finger whose origin is not at
its pad freezes the whole rubric before GRASP/LIFT/PLACE are ever evaluated.

droid.usd puts each finger body's origin within ~1 cm of its pad centroid. The robolab asset (and
the vendor file it is converted from -- this is not a conversion artefact) clusters every gripper
body origin within ~2 cm of the mount, ~0.134 m away from the pads. The gripper geometry itself is
correct: both assets reach ~0.165 m beyond panda_link8, matching a real Robotiq 2F-85.

Fixing the asset rather than the metric keeps every historical REALM number comparable.

What this changes, and what it must not
---------------------------------------
Only the body FRAME moves. World geometry, joint frames and mass properties are all preserved:

  * body xformOp:translate  += R_body * d      (d = geometry centroid in body coords)
  * every child prim        -= d               (keeps meshes world-fixed)
  * every joint localPos on this body's side -= d   (keeps the articulation world-fixed)

Mass properties need no compensation here because these bodies do not author physics:centerOfMass /
diagonalInertia -- PhysX computes them from the collision shapes, whose world poses are unchanged.
The script refuses to run if that assumption ever stops holding.

Usage
-----
    python scripts/fix_robolab_link_origins.py <src.usd> <dst.usd> [--links a,b] [--dry-run]
"""

import argparse
import sys

from pxr import Usd, UsdGeom, UsdPhysics, Gf

DEFAULT_LINKS = ("left_inner_finger", "right_inner_finger")

# Authored mass attributes would be expressed in the body frame and would need the same -= d
# compensation. None of the robolab gripper bodies author them; bail out rather than silently
# changing the dynamics if that changes.
MASS_ATTRS = ("physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")


def body_prim(stage, name):
    for p in stage.Traverse():
        if p.GetName() == name and p.GetParent().GetName() == "panda":
            return p
    return None


def centroid_in_body(stage, prim):
    """Geometry centroid of this body, expressed in the body's own frame."""
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    rng = bbc.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    centre_world = (rng.GetMin() + rng.GetMax()) * 0.5
    return xc.GetLocalToWorldTransform(prim).GetInverse().Transform(centre_world)


def shift_body(stage, name, dry_run=False):
    prim = body_prim(stage, name)
    if prim is None:
        print(f"  {name}: NOT FOUND under /panda -- skipped")
        return False

    for a in MASS_ATTRS:
        at = prim.GetAttribute(a)
        if at and at.IsValid() and at.HasAuthoredValue():
            sys.exit(f"ERROR: {name} authors {a}; it is expressed in the body frame and would need "
                     f"compensating. Refusing to run rather than silently altering the dynamics.")

    d = centroid_in_body(stage, prim)
    if d is None:
        print(f"  {name}: empty bounding box -- skipped")
        return False

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m_local = xc.GetLocalToWorldTransform(prim)  # /panda sits at identity, so this is the body local
    new_origin = m_local.Transform(d)

    t_attr = prim.GetAttribute("xformOp:translate")
    old_t = t_attr.Get()
    print(f"  {name}: centroid offset in body frame = "
          f"({d[0]:+.5f}, {d[1]:+.5f}, {d[2]:+.5f})  |d|={d.GetLength():.5f} m")
    print(f"      origin {tuple(round(v, 5) for v in old_t)} -> {tuple(round(v, 5) for v in new_origin)}")

    n_children = n_joints = 0
    if not dry_run:
        t_attr.Set(Gf.Vec3d(new_origin) if isinstance(old_t, Gf.Vec3d) else Gf.Vec3f(new_origin))

        # Children keep their world pose: a point at p in the old body frame is at p - d in the new.
        for c in prim.GetChildren():
            ct = c.GetAttribute("xformOp:translate")
            if ct and ct.IsValid() and ct.Get() is not None:
                cur = ct.Get()
                ct.Set(type(cur)(cur[0] - d[0], cur[1] - d[1], cur[2] - d[2]))
            else:
                xf = UsdGeom.Xformable(c)
                if xf:
                    xf.AddTranslateOp().Set(Gf.Vec3d(-d[0], -d[1], -d[2]))
            n_children += 1

        # Joint anchors on this body's side are in the body frame and shift with it.
        path = prim.GetPath()
        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            for rel, attr in ((j.GetBody0Rel(), j.GetLocalPos0Attr()),
                              (j.GetBody1Rel(), j.GetLocalPos1Attr())):
                if path in rel.GetTargets():
                    cur = attr.Get()
                    if cur is not None:
                        attr.Set(type(cur)(cur[0] - d[0], cur[1] - d[1], cur[2] - d[2]))
                        n_joints += 1
        print(f"      compensated {n_children} child prim(s), {n_joints} joint anchor(s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--links", default=",".join(DEFAULT_LINKS),
                    help="comma-separated body names to re-origin")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.src)
    if stage is None:
        sys.exit(f"could not open {args.src}")

    print(f"re-origining bodies in {args.src}")
    changed = 0
    for name in [s for s in args.links.split(",") if s]:
        changed += bool(shift_body(stage, name, dry_run=args.dry_run))

    if args.dry_run:
        print(f"\ndry run -- {changed} body/bodies would change, nothing written")
        return
    stage.GetRootLayer().Export(args.dst)
    print(f"\nwrote {args.dst}  ({changed} body/bodies re-origined)")


if __name__ == "__main__":
    main()
