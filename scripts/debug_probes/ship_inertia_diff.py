"""Diff the GRIPPER LINKS' mass properties between RoboLab's asset and REALM's converted one.

Why this, and why it is the strongest remaining candidate
--------------------------------------------------------
A PhysX mimic joint's `naturalFrequency` is a FREQUENCY, not a stiffness. The stiffness the solver
realises from it scales with the articulation's effective inertia, roughly

    k  ~  omega^2 * I

So two stacks can carry a byte-identical authored `naturalFrequency = 1000` and still show very
different constraint compliance if their link inertias differ. That would explain every negative so
far at once: matching gains, drives, force limits, mimic attributes, joint limits and solver settings
all leave the gap untouched, because none of them is the parameter that sets the constraint stiffness.

And there is a concrete reason to suspect it HERE rather than in general: **the two stacks do not load
the same file.** RoboLab spawns `robolab_franka_robotiq_2f_85_flattened.usd`, whose gripper bodies sit
under `/panda/Gripper/Robotiq_2F_85/`. REALM spawns `droid_robolab_v2.usd`, produced by
`scripts/convert_robolab_gripper_usd.py`, which FLATTENS those bodies up to `/panda/` because
OmniGibson only treats direct Xform children of the robot prim as links. Re-parenting rigid bodies is
exactly the operation that can disturb mass, centre of mass, inertia tensor and joint anchor frames.

What it reports
---------------
Per gripper link, from BOTH stages: `physics:mass`, `physics:centerOfMass`, `physics:diagonalInertia`,
`physics:principalAxes`, the link's local transform, and -- because the converter could have moved
them too -- each gripper joint's `physics:localPos0/1`, `physics:localRot0/1`, `physics:axis`, and the
collision approximation on its meshes.

Then the CONVERGENCE TEST, which is the point. If inertia is the mechanism, then the naturalFrequency
REALM would need to realise RoboLab's stiffness is

    nf_equivalent = nf_robolab * sqrt(I_robolab / I_realm)

If that lands near the nf = 100..200 that empirically produces the curl, the mechanism is confirmed
from two directions at once. If the inertias match, this candidate is dead and the answer is
elsewhere -- which is equally worth knowing, so the table is printed either way.

    python /app/scripts/debug_probes/ship_inertia_diff.py
"""

import argparse
import json
import os

import numpy as np

import omnigibson as og

ap = argparse.ArgumentParser()
ap.add_argument("--realm", default="/app/realm/robots/panda_robotiq/droid_robolab_v2.usd")
ap.add_argument("--robolab",
                default="/app/realm/robots/panda_robotiq/robolab_franka_robotiq_2f_85_flattened.usd")
ap.add_argument("--out", default="/logs/gripper_squeeze/ship_inertia_diff.json")
ap.add_argument("--nf-robolab", type=float, default=1000.0,
                help="the naturalFrequency BOTH assets author on the four inner mimic joints")
args = ap.parse_args()

og.launch()

import omnigibson.lazy as lazy  # noqa: E402

Usd, UsdGeom, UsdPhysics = lazy.pxr.Usd, lazy.pxr.UsdGeom, lazy.pxr.UsdPhysics

GRIPPER_LINKS = ("base_link", "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger",
                 "right_outer_finger", "left_inner_finger", "right_inner_finger",
                 "left_inner_knuckle", "right_inner_knuckle")
GRIPPER_JOINTS = ("finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint",
                  "right_inner_finger_joint", "left_inner_finger_knuckle_joint",
                  "right_inner_finger_knuckle_joint")
# the two pads: the links whose pivot is the one that has to rotate for the tip to curl
PAD_LINKS = ("left_inner_finger", "right_inner_finger")

MASS_ATTRS = ("physics:mass", "physics:density", "physics:centerOfMass",
              "physics:diagonalInertia", "physics:principalAxes")
JOINT_ATTRS = ("physics:localPos0", "physics:localPos1", "physics:localRot0", "physics:localRot1",
               "physics:axis", "physics:lowerLimit", "physics:upperLimit",
               "physics:jointEnabled", "physics:excludeFromArticulation")


def val(a):
    if not a or not a.IsValid():
        return None
    v = a.Get()
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return float(v) if isinstance(v, (int, float)) else str(v)


def by_name(stage, names, want_joint=False):
    """Map name -> prim, searching the whole stage so the two different hierarchies both resolve."""
    out = {}
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        n = p.GetName()
        if n not in names:
            continue
        is_joint = "Joint" in p.GetTypeName()
        if want_joint != is_joint:
            continue
        out.setdefault(n, p)
    return out


def dump(path):
    st = Usd.Stage.Open(path)
    assert st, f"could not open {path}"
    links = by_name(st, set(GRIPPER_LINKS))
    joints = by_name(st, set(GRIPPER_JOINTS), want_joint=True)
    out = dict(usd=path, links={}, joints={}, paths={})
    for n, p in links.items():
        d = {a.split(":")[-1]: val(p.GetAttribute(a)) for a in MASS_ATTRS}
        d["prim_path"] = str(p.GetPath())
        d["type"] = str(p.GetTypeName())
        d["has_rigidbody"] = bool(p.HasAPI(UsdPhysics.RigidBodyAPI))
        d["has_massapi"] = bool(p.HasAPI(UsdPhysics.MassAPI))
        # collision approximation on this link's meshes -- convex hull vs decomposition changes the
        # contact points and therefore every lever arm the curl is measured on
        approx = []
        for c in Usd.PrimRange(p):
            a = c.GetAttribute("physxMeshCollision:approximation")
            if a and a.IsValid():
                approx.append(f"{c.GetName()}={a.Get()}")
        d["collision_approx"] = approx
        out["links"][n] = d
    for n, p in joints.items():
        d = {a.split(":")[-1]: val(p.GetAttribute(a)) for a in JOINT_ATTRS}
        for rel in ("physics:body0", "physics:body1"):
            r = p.GetRelationship(rel)
            d[rel.split(":")[-1]] = [str(t) for t in r.GetTargets()] if r and r.IsValid() else None
        d["prim_path"] = str(p.GetPath())
        out["joints"][n] = d
    return out


def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v) + "]"
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def close(a, b, tol=1e-4):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) != isinstance(b, list):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a))


R = dump(args.robolab)
M = dump(args.realm)

print("\n" + "=" * 110)
print("GRIPPER LINK MASS PROPERTIES -- RoboLab vs REALM")
print("=" * 110)
print(f"  RoboLab: {args.robolab}")
print(f"  REALM:   {args.realm}")
n_same = n_diff = 0
rows = []
for n in GRIPPER_LINKS:
    r, m = R["links"].get(n), M["links"].get(n)
    if r is None or m is None:
        print(f"\n  {n:<22} *** present in RoboLab={r is not None} REALM={m is not None}")
        continue
    print(f"\n  {n}   RoboLab {r['prim_path']}\n  {'':<{len(n)}}   REALM   {m['prim_path']}")
    for f in ("mass", "density", "centerOfMass", "diagonalInertia", "principalAxes",
              "has_rigidbody", "has_massapi", "collision_approx"):
        a, b = r.get(f), m.get(f)
        same = close(a, b) if not isinstance(a, (bool, list)) or f not in ("collision_approx",) else a == b
        if f in ("has_rigidbody", "has_massapi", "collision_approx"):
            same = (a == b)
        n_same += same
        n_diff += (not same)
        rows.append(dict(link=n, field=f, robolab=a, realm=b, same=bool(same)))
        print(f"      {f:<18} {fmt(a):>44}   {fmt(b):>44}   {'OK' if same else '***'}")

print("\n" + "=" * 110)
print("GRIPPER JOINT FRAMES / LIMITS -- RoboLab vs REALM")
print("=" * 110)
for n in GRIPPER_JOINTS:
    r, m = R["joints"].get(n), M["joints"].get(n)
    if r is None or m is None:
        print(f"  {n:<34} *** RoboLab={r is not None} REALM={m is not None}")
        continue
    diffs = []
    for f in ("localPos0", "localPos1", "localRot0", "localRot1", "axis",
              "lowerLimit", "upperLimit"):
        a, b = r.get(f), m.get(f)
        same = close(a, b)
        n_same += same
        n_diff += (not same)
        rows.append(dict(link=n, field=f, robolab=a, realm=b, same=bool(same)))
        if not same:
            diffs.append(f"{f}: {fmt(a)} vs {fmt(b)}")
    print(f"  {n:<34} {'IDENTICAL' if not diffs else '*** ' + '; '.join(diffs)}")

# ---------------------------------------------------------------- the convergence test
print("\n" + "=" * 110)
print("CONVERGENCE TEST:  nf_equivalent = nf_robolab * sqrt(I_robolab / I_realm)")
print("=" * 110)
print("  A mimic naturalFrequency is a FREQUENCY: the realised stiffness goes as omega^2 * I. If the")
print("  pads' inertia differs between the stacks, the SAME authored nf realises a different")
print("  stiffness. If the nf REALM would need to match RoboLab lands near the nf = 100..200 that")
print("  empirically produces the curl, that is the mechanism, confirmed from both directions.")
conv = {}
for n in PAD_LINKS:
    r, m = R["links"].get(n), M["links"].get(n)
    if not r or not m:
        continue
    Ir, Im = r.get("diagonalInertia"), m.get("diagonalInertia")
    mr, mm = r.get("mass"), m.get("mass")
    print(f"\n  {n}")
    print(f"      mass              RoboLab {fmt(mr)}   REALM {fmt(mm)}"
          + (f"   ratio {mm / mr:.4g}x" if mr and mm else ""))
    print(f"      diagonalInertia   RoboLab {fmt(Ir)}\n                        REALM   {fmt(Im)}")
    if Ir and Im and len(Ir) == len(Im) == 3:
        rat = [(b / a if a else float('nan')) for a, b in zip(Ir, Im)]
        print(f"      I_realm / I_robolab per axis: [{', '.join(f'{x:.4g}' for x in rat)}]")
        nf_eq = [args.nf_robolab * float(np.sqrt(a / b)) if b else float('nan')
                 for a, b in zip(Ir, Im)]
        print(f"      -> nf_equivalent per axis:    [{', '.join(f'{x:.4g}' for x in nf_eq)}]")
        conv[n] = dict(I_robolab=Ir, I_realm=Im, ratio=rat, nf_equivalent=nf_eq,
                       mass_robolab=mr, mass_realm=mm)
        good = [x for x in nf_eq if np.isfinite(x)]
        if good:
            print(f"      INERTIA_NF_EQUIV {n} min={min(good):.4g} max={max(good):.4g} "
                  f"(the empirical curl rungs are nf=100..200)")
    else:
        print("      [!] diagonalInertia missing on one side -- PhysX then DERIVES the inertia from"
              " the collision geometry and density, which is itself a difference worth chasing.")
        conv[n] = dict(I_robolab=Ir, I_realm=Im, mass_robolab=mr, mass_realm=mm, derived=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(dict(robolab=R, realm=M, rows=rows, convergence=conv,
                   n_same=n_same, n_diff=n_diff), f, indent=1, default=str)
print(f"\n  {n_same} fields match, {n_diff} differ")
print(f"  wrote {args.out}")
print(f"INERTIA_DIFF_{'CLEAN' if n_diff == 0 else 'DIFFERS'} same={n_same} diff={n_diff}")
print("SHIP_INERTIA_OK")
og.shutdown()
