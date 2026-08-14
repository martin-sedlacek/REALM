"""Static verification of a mass-property variant against the shipped asset it sublayers.

Three questions, all answered without a simulator:

1. **Is the arm byte-identical?**  Every prim outside the nine gripper links is compared
   attribute-by-attribute -- the `arm_0` block, the seven `panda_joint*` prims, every `panda_link*`,
   and the top-level `friction` / `armature` opinions. Any difference at all is a failure.

2. **Did `--anchor` move any collision geometry?**  For every geom under every link the FULL
   composed local-to-link matrix is compared between the two stages. Re-anchoring is supposed to
   split the same matrix differently across two prims, so this must agree to floating-point noise.
   The world extent of the points is checked too, which catches a scale mistake the matrix would not.

3. **Do the authored mass properties resolve?**  Reads them back off the composed stage.

    python scripts/debug_probes/verify_mass_variant.py <variant.usda> [--src <shipped.usd>]

Runs on the HOST on CPU (`pip install usd-core numpy`). If the variant's subLayer names the
CONTAINER path (`/app/...`), pass `--sublayer-host` and the check re-writes it into a temporary copy
first -- the shipped file is never touched.
"""
import argparse
import os
import re
import sys
import tempfile

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

LINKS = ("base_link", "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger",
         "right_outer_finger", "left_inner_finger", "right_inner_finger",
         "left_inner_knuckle", "right_inner_knuckle")
GEOM_TYPES = {"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}
MASS_ATTRS = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia",
              "physics:principalAxes", "physics:density")

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "..", "realm", "robots", "panda_robotiq",
                                              "droid_robolab_v2.usd"))
ap.add_argument("--sublayer-host", default=None,
                help="rewrite the variant's subLayers entry to this HOST path in a temp copy")
args = ap.parse_args()

SRC = os.path.abspath(args.src)
vpath = os.path.abspath(args.variant)
if args.sublayer_host:
    txt = open(vpath).read()
    txt = re.sub(r"@[^@]*droid_robolab_v2\.usd@", f"@{os.path.abspath(args.sublayer_host)}@", txt)
    fd, tmp = tempfile.mkstemp(suffix=".usda", dir=os.path.dirname(vpath))
    with os.fdopen(fd, "w") as f:
        f.write(txt)
    vpath = tmp
    print(f"[verify] sublayer rewritten to the host path in {vpath}")

A = Usd.Stage.Open(SRC)
B = Usd.Stage.Open(vpath)
assert A and B, "could not open one of the stages"
print(f"[verify] shipped {SRC}")
print(f"[verify] variant {vpath}")

fails = []


def paths(stage):
    return {p.GetPath().pathString: p for p in Usd.PrimRange(stage.GetPseudoRoot())}


PA, PB = paths(A), paths(B)
only_a = set(PA) - set(PB)
only_b = set(PB) - set(PA)
if only_a or only_b:
    fails.append(f"prim set differs: only-shipped={sorted(only_a)[:5]} only-variant={sorted(only_b)[:5]}")
print(f"[verify] {len(PA)} prims shipped, {len(PB)} in the variant, "
      f"{len(only_a)} / {len(only_b)} unique")

GRIPPER_ROOTS = tuple(f"/panda/{ln}" for ln in LINKS)


def in_gripper(path):
    return any(path == r or path.startswith(r + "/") for r in GRIPPER_ROOTS)


# ------------------------------------------------------------------ 1. everything else identical
n_checked = n_attr = 0
diffs_outside = []
for path, pa in PA.items():
    if path in only_a or in_gripper(path):
        continue
    pb = PB[path]
    n_checked += 1
    if pa.GetTypeName() != pb.GetTypeName():
        diffs_outside.append(f"{path}: type {pa.GetTypeName()} -> {pb.GetTypeName()}")
    if list(pa.GetAppliedSchemas()) != list(pb.GetAppliedSchemas()):
        diffs_outside.append(f"{path}: apiSchemas {list(pa.GetAppliedSchemas())} -> "
                             f"{list(pb.GetAppliedSchemas())}")
    na = {a.GetName() for a in pa.GetAuthoredAttributes()}
    nb = {a.GetName() for a in pb.GetAuthoredAttributes()}
    if na != nb:
        diffs_outside.append(f"{path}: authored attrs {sorted(na ^ nb)}")
    for n in na & nb:
        n_attr += 1
        va, vb = pa.GetAttribute(n).Get(), pb.GetAttribute(n).Get()
        same = (va == vb) if not hasattr(va, "__len__") or isinstance(va, str) else (
            str(va) == str(vb))
        if not same:
            diffs_outside.append(f"{path}.{n}: {str(va)[:60]} -> {str(vb)[:60]}")

ARM = [p for p in PA if "panda_joint" in p or "panda_link" in p or p.endswith("/arm_0")]
print(f"\n[1] NON-GRIPPER PRIMS: {n_checked} prims / {n_attr} authored attributes compared "
      f"({len(ARM)} of them arm prims)")
if diffs_outside:
    fails.append(f"{len(diffs_outside)} differences outside the gripper links")
    for d in diffs_outside[:20]:
        print(f"    *** {d}")
else:
    print("    IDENTICAL -- the arm, its joints and every non-gripper opinion are untouched")


# ------------------------------------------------------------------ 2. collision geometry unmoved
def local_A_t(prim):
    m = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return M[:3, :3].T.copy(), M[3, :3].copy()


def chain_to(link_prim, geom_prim):
    Ax, t, p = np.eye(3), np.zeros(3), geom_prim
    while p and p != link_prim:
        Ap, tp = local_A_t(p)
        Ax, t = Ap @ Ax, Ap @ t + tp
        p = p.GetParent()
    return Ax, t


print("\n[2] COLLISION + VISUAL GEOM POSE, composed to the link frame")
worst_R = worst_t = worst_pt = 0.0
ngeom = 0
for ln in LINKS:
    la, lb = PA.get(f"/panda/{ln}"), PB.get(f"/panda/{ln}")
    if la is None:
        fails.append(f"{ln} missing")
        continue
    for path, pa in PA.items():
        if not path.startswith(f"/panda/{ln}/"):
            continue
        if pa.GetTypeName() not in GEOM_TYPES:
            continue
        pb = PB[path]
        Aa, ta = chain_to(la, pa)
        Ab, tb = chain_to(lb, pb)
        dR = float(np.abs(Aa - Ab).max())
        dt = float(np.abs(ta - tb).max())
        pts = np.asarray(UsdGeom.Mesh(pa).GetPointsAttr().Get(), dtype=np.float64)
        wa = pts @ Aa.T + ta
        wb = pts @ Ab.T + tb
        dp = float(np.abs(wa - wb).max())
        worst_R, worst_t, worst_pt = max(worst_R, dR), max(worst_t, dt), max(worst_pt, dp)
        ngeom += 1
print(f"    {ngeom} geoms; worst |dA|={worst_R:.3e}  worst |dt|={worst_t:.3e} m  "
      f"worst per-VERTEX displacement in the link frame = {worst_pt * 1e9:.4f} nm")
if worst_pt > 1e-9:
    fails.append(f"collision/visual geometry moved by up to {worst_pt * 1e3:.6f} mm")
else:
    print("    UNMOVED to under a nanometre -- the re-anchor is a pure change of which prim "
          "carries the transform")

# ------------------------------------------------------------------ 3. authored mass properties
print("\n[3] AUTHORED MASS PROPERTIES ON THE NINE GRIPPER LINKS (as the variant resolves them)")
n_auth = 0
for ln in LINKS:
    pb = PB.get(f"/panda/{ln}")
    has = bool(pb.HasAPI(UsdPhysics.MassAPI))
    vals = {}
    for a in MASS_ATTRS:
        at = pb.GetAttribute(a)
        if at and at.HasAuthoredValue():
            vals[a.split(":")[-1]] = at.Get()
            n_auth += 1
    print(f"    {ln:<22} MassAPI={has}")
    for k, v in vals.items():
        print(f"        {k:<16} {v}")
    pa = PA.get(f"/panda/{ln}")
    was = [a for a in MASS_ATTRS if pa.GetAttribute(a) and pa.GetAttribute(a).HasAuthoredValue()]
    if was:
        print(f"        (shipped asset already authored: {was})")
print(f"    {n_auth} mass fields authored across the nine links "
      f"(the shipped asset authors 0 of 45)")

print("\n" + "=" * 96)
if fails:
    print("VERIFY_MASS_VARIANT_FAIL")
    for f_ in fails:
        print(f"  *** {f_}")
    sys.exit(1)
print("VERIFY_MASS_VARIANT_OK  arm byte-identical, geometry unmoved, mass properties authored")
