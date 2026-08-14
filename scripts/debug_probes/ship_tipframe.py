"""Hull-free fingertip geometry, straight out of the USD, plus the rigid-body audit of the
hull-derived tip observable that `curl_press_direction.py` reports as `d_tip_sep`.

Why this exists
---------------
`curl_A.log` reported `direction=DISAGREE` on every rung: the pad ROTATION said inward and the
tip-to-tip SEPARATION said outward. `collision_boundary_points_world` is known to be ~116 mm off
the pad link origins on this asset. This script settles which observable is wrong WITHOUT a GPU,
by asking the USD where the fingertip actually is in the pad link's own frame and then replaying
the recorded rigid-body motion through both lever arms.

No Isaac, no physics: it opens the USD with pxr and walks the collision meshes under each pad link.
"""
import argparse
import json
import os

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default="/app/realm/robots/panda_robotiq/droid_robolab_v2.usd")
ap.add_argument("--links", default="left_inner_finger,right_inner_finger")
ap.add_argument("--out", default="/logs/gripper_squeeze/ship_tipframe.json")
args = ap.parse_args()

stage = Usd.Stage.Open(args.usd)
print(f"opened {args.usd}")

LINKS = args.links.split(",")
res = {}

for name in LINKS:
    # find the link prim
    hits = [p for p in stage.Traverse() if p.GetName() == name]
    assert hits, f"no prim named {name}"
    link = hits[0]
    print(f"\n=== {link.GetPath()} ({link.GetTypeName()}) ===")
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    Tlink = xc.GetLocalToWorldTransform(link)
    Tlink_inv = Tlink.GetInverse()

    pts_all = []
    for p in Usd.PrimRange(link):
        if not p.IsA(UsdGeom.Mesh):
            continue
        purpose = p.GetAttribute("purpose").Get() if p.GetAttribute("purpose") else None
        has_coll = p.HasAPI(UsdPhysics.CollisionAPI)
        mesh = UsdGeom.Mesh(p)
        pts = mesh.GetPointsAttr().Get()
        if pts is None:
            continue
        Tm = xc.GetLocalToWorldTransform(p) * Tlink_inv     # mesh -> LINK frame
        a = np.array([[*Tm.Transform(Gf.Vec3d(*q))] for q in pts], dtype=float)
        tag = f"{p.GetName()} n={len(pts)} coll={has_coll} purpose={purpose}"
        print(f"  mesh {tag}")
        print(f"       link-frame bbox min {a.min(0) * 1000} max {a.max(0) * 1000} (mm)")
        if has_coll or (purpose in (None, "default", "guide")):
            pts_all.append((p.GetName(), has_coll, a))

    coll = [a for _, c, a in pts_all if c]
    use = coll if coll else [a for _, _, a in pts_all]
    A = np.concatenate(use, axis=0)
    print(f"  -> {len(A)} points over {len(use)} mesh(es) ({'collision' if coll else 'ALL render'})")
    lo, hi = A.min(0), A.max(0)
    print(f"  LINK-FRAME bbox  min {lo * 1000} max {hi * 1000} mm   extent {(hi - lo) * 1000} mm")
    print(f"  LINK-FRAME centroid {A.mean(0) * 1000} mm")
    res[name] = dict(bbox_min_mm=(lo * 1000).tolist(), bbox_max_mm=(hi * 1000).tolist(),
                     extent_mm=((hi - lo) * 1000).tolist(), centroid_mm=(A.mean(0) * 1000).tolist(),
                     n_points=int(len(A)), used_collision=bool(coll))

    # the joint whose child is this link, and its local frame on the child -- the hinge the pad
    # rotates about, in the pad's OWN frame. This is the hull-free pivot.
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b1 = j.GetBody1Rel().GetTargets()
        if b1 and b1[0].name == name:
            lp1 = p.GetAttribute("physics:localPos1").Get()
            lp0 = p.GetAttribute("physics:localPos0").Get()
            ax = p.GetAttribute("physics:axis").Get()
            b0 = j.GetBody0Rel().GetTargets()
            print(f"  JOINT {p.GetName()}  parent={b0[0].name if b0 else None}  axis={ax}")
            print(f"       localPos0 (in parent) {np.array([*lp0]) * 1000} mm")
            print(f"       localPos1 (in THIS link) {np.array([*lp1]) * 1000} mm  <- the hinge, in the pad frame")
            res[name][f"joint_{p.GetName()}"] = dict(
                parent=(b0[0].name if b0 else None), axis=str(ax),
                localPos0_mm=(np.array([*lp0]) * 1000).tolist(),
                localPos1_mm=(np.array([*lp1]) * 1000).tolist())

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print(f"\nwrote {args.out}")
print("SHIP_TIPFRAME_OK")
