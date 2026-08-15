"""Diff two inertia_dump.py JSONs and print the tables the hypothesis needs.

Usage:  python inertia_diff.py <robolab.json> <realm.json> <out.json>
"""

import json
import math
import sys

GRIPPER_LINKS = [
    "base_link",
    "left_outer_knuckle", "right_outer_knuckle",
    "left_outer_finger", "right_outer_finger",
    "left_inner_finger", "right_inner_finger",
    "left_inner_knuckle", "right_inner_knuckle",
]

GRIPPER_JOINTS = [
    "finger_joint",
    "left_outer_knuckle_joint", "right_outer_knuckle_joint",
    "left_inner_knuckle_joint", "right_inner_knuckle_joint",
    "left_inner_finger_joint", "right_inner_finger_joint",
    "left_outer_finger_joint", "right_outer_finger_joint",
]

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def fmt(v, prec=9):
    if v is None:
        return "None"
    if isinstance(v, (str, bool)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x, prec) for x in v) + "]"
    return str(v)


def same(a, b, tol=1e-9):
    """Structural equality with a float tolerance, so 1e-17 noise is not called a difference."""
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
        return False
    if isinstance(a, float):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    return a == b


def relerr(a, b):
    """Max relative difference across a scalar or vector pair, as a percentage."""
    if a is None or b is None:
        return None
    if isinstance(a, (int, float)):
        a, b = [a], [b]
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        denom = max(abs(x), abs(y))
        if denom < 1e-15:
            continue
        worst = max(worst, abs(x - y) / denom)
    return 100.0 * worst


def section(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def main(f_rl, f_rm, f_out):
    RL = json.load(open(f_rl))
    RM = json.load(open(f_rm))
    report = {"robolab": f_rl, "realm": f_rm}

    # ---------------------------------------------------------------- stage-level
    section("STAGE METADATA")
    for k in ("default_prim", "metersPerUnit", "kgPerUnit", "upAxis"):
        a, b = RL.get(k), RM.get(k)
        print(f"  {k:24s}  robolab={fmt(a):28s}  realm={fmt(b):28s}  {'SAME' if same(a, b) else 'DIFF'}")
    report["stage"] = {k: {"robolab": RL.get(k), "realm": RM.get(k)}
                       for k in ("default_prim", "metersPerUnit", "kgPerUnit", "upAxis")}

    section("ARTICULATION ROOT")
    print(f"  robolab: {json.dumps(RL['articulation_roots'], sort_keys=True)}")
    print(f"  realm  : {json.dumps(RM['articulation_roots'], sort_keys=True)}")
    print(f"  -> {'SAME' if same(RL['articulation_roots'], RM['articulation_roots']) else 'DIFF'}")
    report["articulation_roots"] = {"robolab": RL["articulation_roots"], "realm": RM["articulation_roots"]}

    section("KINEMATIC TREE -- rigid-body paths")
    rl_b, rm_b = RL["all_rigid_bodies"], RM["all_rigid_bodies"]
    print(f"  robolab: {len(rl_b)} bodies, realm: {len(rm_b)} bodies")
    rl_names = sorted(p.rsplit("/", 1)[-1] for p in rl_b)
    rm_names = sorted(p.rsplit("/", 1)[-1] for p in rm_b)
    print(f"  same body NAME set: {rl_names == rm_names}")
    if rl_names != rm_names:
        print(f"    robolab only: {sorted(set(rl_names) - set(rm_names))}")
        print(f"    realm   only: {sorted(set(rm_names) - set(rl_names))}")
    print("  paths that moved:")
    rm_by_name = {p.rsplit("/", 1)[-1]: p for p in rm_b}
    for p in rl_b:
        n = p.rsplit("/", 1)[-1]
        q = rm_by_name.get(n)
        if q != p:
            print(f"    {n:24s}  {p}  ->  {q}")
    print("  direct children of the robot prim (= OmniGibson links):")
    print(f"    robolab: {[c['name'] for c in RL['direct_children']]}")
    print(f"    realm  : {[c['name'] for c in RM['direct_children']]}")
    report["tree"] = {
        "robolab_bodies": rl_b, "realm_bodies": rm_b,
        "robolab_direct_children": RL["direct_children"],
        "realm_direct_children": RM["direct_children"],
    }

    # ---------------------------------------------------------------- mass / inertia
    section("MASS / INERTIA -- the hypothesis under test")
    hdr = f"  {'link':22s} {'field':18s} {'robolab':>40s} {'realm':>40s}  verdict"
    print(hdr)
    print("  " + "-" * 106)
    mi = {}
    for ln in GRIPPER_LINKS:
        a, b = RL["links"].get(ln), RM["links"].get(ln)
        if a is None or b is None:
            print(f"  {ln:22s} MISSING robolab={a is not None} realm={b is not None}")
            continue
        entry = {}
        for field in ("mass", "density", "centerOfMass", "diagonalInertia", "principalAxes"):
            x, y = a.get(field), b.get(field)
            ok = same(x, y, 1e-7)
            re_ = relerr(x, y)
            tag = "SAME" if ok else f"DIFF ({re_:.3g}%)" if re_ is not None else "DIFF"
            print(f"  {ln:22s} {field:18s} {fmt(x):>40s} {fmt(y):>40s}  {tag}")
            entry[field] = {"robolab": x, "realm": y, "same": ok, "rel_pct": re_}
        mi[ln] = entry
        print("  " + "-" * 106)
    report["mass_inertia"] = mi

    section("MASS / INERTIA -- rollup")
    n_same = sum(1 for l in mi.values() for f in l.values() if f["same"])
    n_tot = sum(1 for l in mi.values() for _ in l.values())
    print(f"  fields identical: {n_same}/{n_tot}")
    worst = [(l, f, d["rel_pct"]) for l, e in mi.items() for f, d in e.items()
             if not d["same"] and d["rel_pct"] is not None]
    worst.sort(key=lambda x: -x[2])
    for l, f, r in worst[:15]:
        print(f"    {l:22s} {f:18s} {r:.4g}%")
    if not worst:
        print("    -> no numerical differences at all in any mass property")
    tm_a = sum(RL["links"][l]["mass"] for l in GRIPPER_LINKS if RL["links"].get(l, {}).get("mass"))
    tm_b = sum(RM["links"][l]["mass"] for l in GRIPPER_LINKS if RM["links"].get(l, {}).get("mass"))
    print(f"  total gripper mass:  robolab={tm_a:.9g} kg   realm={tm_b:.9g} kg   "
          f"ratio={tm_b / tm_a if tm_a else float('nan'):.9g}")
    report["total_mass"] = {"robolab": tm_a, "realm": tm_b}

    section("PER-BODY PHYSX SOLVER / DAMPING OVERRIDES")
    keys = [k for k in RL["links"]["base_link"] if k.startswith("physx_")]
    any_diff = False
    for ln in GRIPPER_LINKS:
        a, b = RL["links"].get(ln, {}), RM["links"].get(ln, {})
        for k in keys:
            if not same(a.get(k), b.get(k)):
                print(f"  {ln:22s} {k:44s} robolab={fmt(a.get(k))}  realm={fmt(b.get(k))}")
                any_diff = True
    print("  -> all identical" if not any_diff else "")

    section("LINK WORLD POSES (converter re-parented; poses are supposed to be preserved)")
    print(f"  {'link':22s} {'|dt| (m)':>14s} {'quat dot':>12s}   robolab world t")
    posed = {}
    for ln in GRIPPER_LINKS:
        a, b = RL["links"].get(ln), RM["links"].get(ln)
        if not a or not b:
            continue
        ta, tb = a["world_translate"], b["world_translate"]
        d = math.dist(ta, tb)
        qa, qb = a["world_quat"], b["world_quat"]
        dot = abs(sum(x * y for x, y in zip(qa, qb)))
        print(f"  {ln:22s} {d:14.3e} {dot:12.9f}   {fmt(ta, 6)}")
        posed[ln] = {"trans_delta_m": d, "quat_absdot": dot,
                     "robolab_world_t": ta, "realm_world_t": tb}
    report["world_poses"] = posed

    # ---------------------------------------------------------------- joints
    section("GRIPPER JOINT FRAMES / LIMITS / DRIVES")
    jkeys = ["type", "physics:body0", "physics:body1", "physics:axis",
             "physics:localPos0", "physics:localRot0", "physics:localPos1", "physics:localRot1",
             "physics:lowerLimit", "physics:upperLimit", "physics:jointEnabled",
             "physics:excludeFromArticulation", "physics:breakForce", "physics:breakTorque",
             "physics:collisionEnabled",
             "drive:angular:physics:type", "drive:angular:physics:stiffness",
             "drive:angular:physics:damping", "drive:angular:physics:maxForce",
             "drive:angular:physics:targetPosition", "drive:angular:physics:targetVelocity",
             "physxJoint:maxJointVelocity", "physxJoint:jointFriction", "physxJoint:armature"]
    jrep = {}
    for jn in GRIPPER_JOINTS:
        a, b = RL["joints"].get(jn), RM["joints"].get(jn)
        print(f"\n  --- {jn} ---")
        if a is None or b is None:
            print(f"      present robolab={a is not None} realm={b is not None}")
            jrep[jn] = {"present": {"robolab": a is not None, "realm": b is not None}}
            continue
        e = {}
        for k in jkeys:
            x, y = a.get(k), b.get(k)
            if k in ("physics:body0", "physics:body1") and x and y:
                # paths legitimately differ by the removed /Gripper/Robotiq_2F_85 nesting
                xs = [p.rsplit("/", 1)[-1] for p in x]
                ys = [p.rsplit("/", 1)[-1] for p in y]
                ok = xs == ys
                print(f"      {k:38s} {fmt(xs):>26s} {fmt(ys):>26s}  {'SAME(name)' if ok else 'DIFF'}")
                e[k] = {"robolab": x, "realm": y, "same_leafname": ok}
                continue
            ok = same(x, y, 1e-7)
            if x is None and y is None:
                continue
            re_ = relerr(x, y)
            tag = "SAME" if ok else (f"DIFF ({re_:.3g}%)" if re_ is not None else "DIFF")
            print(f"      {k:38s} {fmt(x):>26s} {fmt(y):>26s}  {tag}")
            e[k] = {"robolab": x, "realm": y, "same": ok, "rel_pct": re_}
        # mimic block
        ma, mb = a.get("mimic", {}), b.get("mimic", {})
        if ma or mb:
            for k in sorted(set(ma) | set(mb)):
                x, y = ma.get(k), mb.get(k)
                if k.endswith("referenceJoint") and x and y:
                    xs = [p.rsplit("/", 1)[-1] for p in x]
                    ys = [p.rsplit("/", 1)[-1] for p in y]
                    print(f"      {k:38s} {fmt(xs):>26s} {fmt(ys):>26s}  "
                          f"{'SAME(name)' if xs == ys else 'DIFF'}")
                    e[k] = {"robolab": x, "realm": y, "same_leafname": xs == ys}
                    continue
                ok = same(x, y, 1e-7)
                print(f"      {k:38s} {fmt(x):>26s} {fmt(y):>26s}  {'SAME' if ok else 'DIFF'}")
                e[k] = {"robolab": x, "realm": y, "same": ok}
        sa = set(a.get("applied_schemas", []))
        sb = set(b.get("applied_schemas", []))
        if sa != sb:
            print(f"      applied_schemas  robolab-only={sorted(sa - sb)}  realm-only={sorted(sb - sa)}")
        e["applied_schemas"] = {"robolab": sorted(sa), "realm": sorted(sb)}
        jrep[jn] = e
    report["gripper_joints"] = jrep

    section("ALL JOINTS PRESENT")
    ja, jb = set(RL["joints"]), set(RM["joints"])
    print(f"  robolab {len(ja)}: {sorted(ja)}")
    print(f"  realm   {len(jb)}: {sorted(jb)}")
    print(f"  robolab only: {sorted(ja - jb)}")
    print(f"  realm   only: {sorted(jb - ja)}")
    report["joint_sets"] = {"robolab_only": sorted(ja - jb), "realm_only": sorted(jb - ja)}

    section("ARM BLOCK -- must be byte-identical")
    arm = {}
    arm_diff = []
    for jn in ARM_JOINTS:
        a, b = RL["joints"].get(jn), RM["joints"].get(jn)
        if a is None or b is None:
            arm_diff.append((jn, "PRESENCE", a is not None, b is not None))
            continue
        for k in jkeys:
            x, y = a.get(k), b.get(k)
            if k in ("physics:body0", "physics:body1") and x and y:
                x = [p.rsplit("/", 1)[-1] for p in x]
                y = [p.rsplit("/", 1)[-1] for p in y]
            if not same(x, y, 1e-9):
                arm_diff.append((jn, k, x, y))
        arm[jn] = {k: a.get(k) for k in jkeys}
    if arm_diff:
        for jn, k, x, y in arm_diff:
            print(f"  DIFF {jn:16s} {k:38s} robolab={fmt(x)}  realm={fmt(y)}")
    else:
        print(f"  all {len(ARM_JOINTS)} panda_joint* identical across every field checked")
    report["arm_joints_diff"] = arm_diff
    report["arm_joints_robolab"] = arm

    section("COLLISION APPROXIMATION ON PAD / FINGER MESHES")
    colrep = {}
    for ln in GRIPPER_LINKS:
        a, b = RL["links"].get(ln), RM["links"].get(ln)
        if not a or not b:
            continue
        ca = {c["path"].rsplit("/", 1)[-1]: c for c in a["collisions"]}
        cb = {c["path"].rsplit("/", 1)[-1]: c for c in b["collisions"]}
        if not ca and not cb:
            continue
        print(f"\n  --- {ln} ---   robolab {len(ca)} collider(s), realm {len(cb)} collider(s)")
        for n in sorted(set(ca) | set(cb)):
            x, y = ca.get(n, {}), cb.get(n, {})
            fields = ("approximation", "physxMeshCollision_approximation", "collisionEnabled",
                      "contactOffset", "restOffset")
            bits = []
            for f in fields:
                if not same(x.get(f), y.get(f)):
                    bits.append(f"{f}: {fmt(x.get(f))} -> {fmt(y.get(f))}")
            mark = "  <<< DIFF" if bits else ""
            print(f"      {n:34s} approx={fmt(x.get('approximation')):16s}/"
                  f"{fmt(y.get('approximation')):16s} "
                  f"physxMesh={fmt(x.get('physxMeshCollision_approximation')):16s}/"
                  f"{fmt(y.get('physxMeshCollision_approximation')):16s}{mark}")
            for bl in bits:
                print(f"          {bl}")
        colrep[ln] = {"robolab": ca, "realm": cb}
    report["collisions"] = colrep

    with open(f_out, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(f"\n\nwrote {f_out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
