"""Resolve every gripper joint anchor into WORLD space on both assets, and diff there.

Why world space is the only fair comparison
-------------------------------------------
`physics:localPos1` is the anchor expressed in the CHILD BODY's frame. `fix_robolab_link_origins.py`
deliberately moves the two pad bodies' origins onto their own geometry (so REALM's reach metric,
which measures from link origins, stops failing), and compensates by subtracting the same `d` from
every joint anchor on that body's side. So `localPos1` on the two `*_inner_finger_joint`s MUST differ
between the assets by exactly `d` -- that is the compensation doing its job, not a displaced pivot.

The physical question is whether the anchor lands in the same place in the WORLD. This resolves
    world_anchor_side0 = X_world(body0) * localPos0
    world_anchor_side1 = X_world(body1) * localPos1
on both assets and diffs those. It also reports the intra-asset |side0 - side1| gap, which is the
joint's own error and should be ~0 for a correctly authored joint.

Then it reconciles the two loose numbers in circulation -- a "~100 mm anchor discrepancy" and a
"116 mm hull-vs-origin offset" -- against the actual origin move, in the panda_link8 frame the
measurements were reported in.

Runs on stock pxr (pip usd-core): no Kit, no GPU.
"""

import json
import math
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

PADS = ("left_inner_finger", "right_inner_finger")


def quat_to_m4(q):
    """(w,x,y,z) list -> Gf.Matrix4d rotation."""
    return Gf.Matrix4d().SetRotate(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))


def joints_by_path(stage):
    out = {}
    for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if "Joint" in p.GetTypeName():
            out[p.GetPath().pathString] = p
    return out


def body_of(stage, rel):
    t = rel.GetTargets() if rel else []
    return stage.GetPrimAtPath(t[0]) if t else None


def anchors(stage, cache):
    """world anchor from each side of every joint, plus the joint's own side0/side1 gap."""
    res = {}
    for path, j in sorted(joints_by_path(stage).items()):
        name = j.GetName()
        rec = {"path": path, "name": name, "type": j.GetTypeName()}
        for side in (0, 1):
            b = body_of(stage, j.GetRelationship(f"physics:body{side}"))
            lp = j.GetAttribute(f"physics:localPos{side}").Get()
            lr = j.GetAttribute(f"physics:localRot{side}").Get()
            if b is None or lp is None:
                rec[f"world{side}"] = None
                rec[f"body{side}"] = b.GetName() if b else None
                continue
            X = cache.GetLocalToWorldTransform(b)
            rec[f"body{side}"] = b.GetName()
            rec[f"local{side}"] = [float(x) for x in lp]
            rec[f"world{side}"] = [float(x) for x in X.Transform(Gf.Vec3d(*lp))]
            if lr is not None:
                # world axis: the joint's local frame rotation applied to the body's rotation,
                # then the joint's declared axis. Compare the FRAME, not just the anchor point.
                q = Gf.Quatd(float(lr.GetReal()), Gf.Vec3d(*[float(x) for x in lr.GetImaginary()]))
                Rj = Gf.Matrix4d().SetRotate(q)
                Rb = Gf.Matrix4d().SetRotate(X.ExtractRotationQuat())
                M = Rj * Rb
                ax = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0),
                      "Z": Gf.Vec3d(0, 0, 1)}[str(j.GetAttribute("physics:axis").Get() or "X")]
                rec[f"world_axis{side}"] = [float(x) for x in M.TransformDir(ax)]
        w0, w1 = rec.get("world0"), rec.get("world1")
        rec["side_gap_m"] = math.dist(w0, w1) if w0 and w1 else None
        res[name if name not in res else path] = rec
    return res


def link_world(stage, cache, name):
    for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI):
            return Gf.Matrix4d(cache.GetLocalToWorldTransform(p)), p
    return None, None


def geom_bounds(stage, prim):
    """World-space bbox of everything under this link (its geometry), and the range corners."""
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
    rng = bbc.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return {"min": [float(x) for x in mn], "max": [float(x) for x in mx],
            "mid": [float((a + b) / 2) for a, b in zip(mn, mx)],
            "size": [float(b - a) for a, b in zip(mn, mx)]}


def main(f_rl, f_rm, out_path):
    S = {"robolab": Usd.Stage.Open(f_rl), "realm": Usd.Stage.Open(f_rm)}
    C = {k: UsdGeom.XformCache(Usd.TimeCode.Default()) for k in S}
    A = {k: anchors(S[k], C[k]) for k in S}
    report = {"robolab_usd": f_rl, "realm_usd": f_rm}

    print("=" * 118)
    print("JOINT ANCHORS RESOLVED INTO WORLD SPACE")
    print("=" * 118)
    print(f"  {'joint':34s} {'|world0 A-B|':>13s} {'|world1 A-B|':>13s} "
          f"{'gapA':>10s} {'gapB':>10s}  axis dot")
    print("  " + "-" * 114)
    rows = {}
    for n in sorted(set(A["robolab"]) | set(A["realm"])):
        a, b = A["robolab"].get(n), A["realm"].get(n)
        if a is None or b is None:
            print(f"  {n:34s}   present robolab={a is not None} realm={b is not None}")
            continue
        d0 = math.dist(a["world0"], b["world0"]) if a.get("world0") and b.get("world0") else None
        d1 = math.dist(a["world1"], b["world1"]) if a.get("world1") and b.get("world1") else None
        ax = None
        if a.get("world_axis1") and b.get("world_axis1"):
            ax = sum(x * y for x, y in zip(a["world_axis1"], b["world_axis1"]))
        f = lambda v, w=13: (f"{v:{w}.3e}" if v is not None else " " * (w - 4) + "n/a")
        print(f"  {n:34s} {f(d0)} {f(d1)} {f(a['side_gap_m'], 10)} {f(b['side_gap_m'], 10)}  "
              f"{('%.9f' % ax) if ax is not None else 'n/a'}")
        rows[n] = {"world0_delta_m": d0, "world1_delta_m": d1,
                   "robolab_side_gap_m": a["side_gap_m"], "realm_side_gap_m": b["side_gap_m"],
                   "axis_dot": ax,
                   "robolab_local1": a.get("local1"), "realm_local1": b.get("local1"),
                   "robolab_world1": a.get("world1"), "realm_world1": b.get("world1")}
    report["anchors"] = rows
    worst = max((v["world1_delta_m"] for v in rows.values()
                 if v["world1_delta_m"] is not None), default=None)
    print(f"\n  worst WORLD anchor displacement across all joints: {worst:.3e} m")

    print("\n" + "=" * 118)
    print("THE PAD JOINTS: local-frame delta vs world delta vs the link-origin move")
    print("=" * 118)
    pad_rep = {}
    for jn in ("left_inner_finger_joint", "right_inner_finger_joint"):
        r = rows.get(jn)
        if not r:
            continue
        lp_a, lp_b = r["robolab_local1"], r["realm_local1"]
        dloc = math.dist(lp_a, lp_b)
        pad = jn.replace("_joint", "")
        Xa, _ = link_world(S["robolab"], C["robolab"], pad)
        Xb, _ = link_world(S["realm"], C["realm"], pad)
        dorg = math.dist([float(x) for x in Xa.ExtractTranslation()],
                         [float(x) for x in Xb.ExtractTranslation()])
        print(f"\n  {jn}")
        print(f"      localPos1 robolab  {[round(x, 9) for x in lp_a]}")
        print(f"      localPos1 realm    {[round(x, 9) for x in lp_b]}")
        print(f"      |localPos1 delta|          = {dloc * 1000:10.4f} mm   (frame-of-expression)")
        print(f"      |{pad} origin move| = {dorg * 1000:10.4f} mm   (fix_robolab_link_origins)")
        print(f"      difference between the two = {abs(dloc - dorg) * 1e6:10.4f} um")
        print(f"      |WORLD anchor delta|       = {r['world1_delta_m'] * 1e6:10.4f} um  <<< the physical quantity")
        pad_rep[jn] = {"local_delta_mm": dloc * 1000, "origin_move_mm": dorg * 1000,
                       "world_delta_um": r["world1_delta_m"] * 1e6,
                       "explained": abs(dloc - dorg) < 1e-6}

    print("\n" + "=" * 118)
    print("RECONCILING THE LOOSE NUMBERS, in the panda_link8 frame")
    print("=" * 118)
    for k in ("robolab", "realm"):
        X8, _ = link_world(S[k], C[k], "panda_link8")
        if X8 is None:
            for p in Usd.PrimRange(S[k].GetDefaultPrim()):
                if p.GetName() == "panda_link8":
                    X8 = Gf.Matrix4d(C[k].GetLocalToWorldTransform(p))
                    break
        inv8 = X8.GetInverse()
        print(f"\n  --- {k} ---")
        for pad in PADS:
            Xp, prim = link_world(S[k], C[k], pad)
            org8 = inv8.Transform(Xp.ExtractTranslation())
            gb = geom_bounds(S[k], prim)
            print(f"    {pad}")
            print(f"      link origin in link8 frame : "
                  f"[{org8[0] * 1000:9.3f}, {org8[1] * 1000:9.3f}, {org8[2] * 1000:9.3f}] mm")
            if gb:
                mid8 = inv8.Transform(Gf.Vec3d(*gb["mid"]))
                print(f"      geom bbox mid in link8     : "
                      f"[{mid8[0] * 1000:9.3f}, {mid8[1] * 1000:9.3f}, {mid8[2] * 1000:9.3f}] mm")
                d = [(mid8[i] - org8[i]) * 1000 for i in range(3)]
                print(f"      geom mid - origin          : "
                      f"[{d[0]:9.3f}, {d[1]:9.3f}, {d[2]:9.3f}] mm   |.|={math.dist([0,0,0], d):8.3f} mm")
                print(f"      geom bbox size             : "
                      f"{[round(x * 1000, 2) for x in gb['size']]} mm")
            a = A[k].get(f"{pad}_joint")
            if a and a.get("world1"):
                anc8 = inv8.Transform(Gf.Vec3d(*a["world1"]))
                print(f"      PIVOT anchor in link8      : "
                      f"[{anc8[0] * 1000:9.3f}, {anc8[1] * 1000:9.3f}, {anc8[2] * 1000:9.3f}] mm")
                if gb:
                    lev = math.dist([float(x) for x in a["world1"]], gb["mid"])
                    print(f"      lever arm pivot->geom mid  : {lev * 1000:8.3f} mm  <<< the moment arm")
        report.setdefault("link8_frame", {})[k] = "printed"

    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out_path}")
    ok = worst is not None and worst < 1e-6
    print(f"ANCHOR_WORLD_{'IDENTICAL' if ok else 'DIFFERS'} worst={worst:.3e} m")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
