"""Write a variant USD that AUTHORS the gripper links' mass properties, so PhysX derives nothing.

Why this exists
---------------
Neither `droid_robolab_v2.usd` nor RoboLab's `robolab_franka_robotiq_2f_85_flattened.usd` authors a
single mass property on any of the nine gripper LINKS: no `physics:mass`, `physics:density`,
`physics:centerOfMass`, `physics:diagonalInertia`, `physics:principalAxes`, and no `MassAPI` on the
link prim. PhysX therefore derives everything from the collision shapes -- and OmniGibson then
OVERWRITES the derived centre of mass with a value it computes itself in `RigidPrim.update_meshes()`
(`rigid_prim.py`), using `mesh.get_position_orientation(frame="parent")`. `frame="parent"` is the
geom's IMMEDIATE parent, which here is a `Defeatured_*_01` Xform carrying the left/right mirror, not
the link. The Xform -> link step is silently dropped, both pads get an identical CoM including the
sign of y, and each lands 128 mm from its true centroid. `m*d^2` then inflates the pad's inertia
about its own pivot 77x, and a PhysX mimic joint realises `k ~ omega^2 * I`, so the fingertips come
out ~77x too stiff at the authored `naturalFrequency = 1000` and do not curl.

This is the ASSET-SIDE route to the same fix: author the mass properties explicitly so there is
nothing left to derive. It needs no OmniGibson patch.

Two things it can write, independently or together
--------------------------------------------------
`--mass`   apply `UsdPhysics.MassAPI` to the nine gripper links and author `physics:mass`,
           `physics:centerOfMass`, `physics:diagonalInertia` and `physics:principalAxes`.

`--anchor` RE-ANCHOR each collision geom: move the intermediate `Defeatured_*_01` Xform's
           translate/orient onto the Mesh beneath it and set the Xform to identity. The composed
           link -> mesh transform, and therefore every collision shape's pose in world space, is
           bit-identical; the only thing that changes is that `frame="parent"` now returns the FULL
           link -> mesh transform, so OmniGibson's own computation lands on the right answer.
           Needed because `physics:centerOfMass` is the ONE authored field OmniGibson does not
           respect -- `update_meshes()` assigns `self.center_of_mass`, whose setter calls
           `RigidPrimView.set_coms()`, whose stopped-simulation fallback writes the USD attribute
           directly (`deprecated_utils.py`). `mass`, `diagonalInertia` and `principalAxes` survive;
           `centerOfMass` does not.

Where the numbers come from
---------------------------
`mass` and `diagonalInertia`/`principalAxes` are RoboLab's RUNTIME values, read off its live
articulation in `/logs/gripper_squeeze/wrapdiff_robolab_runtime.json` and reproduced literally in
`ROBOLAB_RUNTIME` below. They are correct by construction: RoboLab loads the same geometry through
Isaac Lab, which does not rewrite mass properties, so PhysX's own derivation survives intact. The
masses are already bit-identical to REALM's, so authoring them changes nothing; the inertia tensors
are the reference the whole investigation is calibrated against (pad inertia about the pivot
1.937e-06 against REALM's 1.496e-04).

`centerOfMass` is computed HERE from the asset's own collision triangle meshes, composed all the way
to the link frame, volume-weighted across the link's geoms -- i.e. the quantity OmniGibson intends
to compute and gets wrong. For the pads this lands at (-4.33, -+4.24, 0.00) mm, restoring the
left/right mirror, against the (-54.20, +116.34, 0.00) mm OmniGibson writes for BOTH.

RoboLab's dump carries no CoM, so the CoM cannot be lifted from it; and REALM moved the two pads'
link ORIGINS relative to RoboLab's (`scripts/fix_robolab_link_origins.py`), so a CoM would have had
to be re-expressed anyway. The inertia tensor about the CoM is origin-independent and the two link
frames share an orientation -- confirmed by the seven non-pad links, whose corrected-loader runtime
tensors agree with RoboLab's to 1.4-4.5% -- so the tensors transfer directly.

Runs on the HOST, on CPU: `pip install usd-core numpy`. No Kit, no GPU, no container.

    python scripts/debug_probes/make_mass_variant.py --mass --anchor --out ./tmp/variants
    python scripts/debug_probes/make_mass_variant.py --mass --out ./tmp/variants

Feed the result to a probe with `--variant-usd <container path>`, which monkeypatches
`Robot.usd_path` for the robolab asset only. `droid_robolab_v2.usd` is never written to and nothing
under `data/` is touched.
"""
import argparse
import os

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------------------------------------
# RoboLab's RUNTIME gripper mass properties, transcribed from
#   /logs/gripper_squeeze/wrapdiff_robolab_runtime.json  ["bodies_runtime"]
# mass in kg; inertia is the full 3x3 about the link CoM, expressed in the link frame, row-major.
# ---------------------------------------------------------------------------------------------
ROBOLAB_RUNTIME = {
    "base_link": dict(mass=0.2888006865978241, inertia=[
        1.7371861204039305e-04, -2.0263303537329193e-06, -3.0197551396576222e-08,
        -2.0263308084802702e-06, 2.1743591199163347e-04, -1.5153888890053146e-07,
        -3.0197551396576222e-08, -1.5153888890053146e-07, 2.4862775113433599e-04]),
    "left_outer_knuckle": dict(mass=0.008290168829262257, inertia=[
        1.1762049329001433e-06, 4.3998255113838706e-08, 8.8050699368225110e-10,
        4.3998255113838706e-08, 2.8993241621996043e-07, 2.3089871123762410e-10,
        8.8050699368225110e-10, 2.3089871123762410e-10, 1.2440539180660271e-06]),
    "right_outer_knuckle": dict(mass=0.008290168829262257, inertia=[
        1.1762049329001433e-06, -4.3998255113838706e-08, -8.8050699368225110e-10,
        -4.3998255113838706e-08, 2.8993241621996043e-07, 2.3089871123762410e-10,
        -8.8050699368225110e-10, 2.3089871123762410e-10, 1.2440539180660271e-06]),
    "left_outer_finger": dict(mass=0.027822598814964294, inertia=[
        2.7144869755117781e-06, -1.0459442582484800e-06, 1.2798480497622222e-08,
        -1.0459442582484800e-06, 8.4514003887888044e-06, 9.9560395483422330e-09,
        1.2798480497622222e-08, 9.9560395483422330e-09, 7.8829316356114112e-06]),
    "right_outer_finger": dict(mass=0.027822598814964294, inertia=[
        2.7144869755117781e-06, 1.0459442582484800e-06, -1.2798480497622222e-08,
        1.0459442582484800e-06, 8.4514003887888044e-06, 9.9560395483422330e-09,
        -1.2798480497622222e-08, 9.9560395483422330e-09, 7.8829316356114112e-06]),
    "left_inner_finger": dict(mass=0.009513214230537415, inertia=[
        9.4477331913367380e-07, 5.1169797643524360e-07, -1.2361139667405041e-08,
        5.1169797643524360e-07, 1.6186228322112584e-06, 7.7634929596115400e-09,
        -1.2361141443761880e-08, 7.7634894068978610e-09, 1.9366536889720010e-06]),
    "right_inner_finger": dict(mass=0.009513214230537415, inertia=[
        9.4477331913367380e-07, -5.1169809012208130e-07, 1.2361147661010818e-08,
        -5.1169803327866250e-07, 1.6186234006454470e-06, 7.7635036177525760e-09,
        1.2361147661010818e-08, 7.7634965123252190e-09, 1.9366548258403780e-06]),
    "left_inner_knuckle": dict(mass=0.025188090279698372, inertia=[
        6.1251985197303817e-06, -4.2592605495883620e-06, 1.3123537812020913e-08,
        -4.2592610043357130e-06, 7.4248641754800454e-06, 1.2801356419345211e-08,
        1.3123536923842494e-08, 1.2801356419345211e-08, 9.0441135396435857e-06]),
    "right_inner_knuckle": dict(mass=0.025188090279698372, inertia=[
        6.1251985197303817e-06, 4.2592605495883620e-06, -1.3123537812020913e-08,
        4.2592610043357130e-06, 7.4248641754800454e-06, 1.2801356419345211e-08,
        -1.3123536923842494e-08, 1.2801356419345211e-08, 9.0441135396435857e-06]),
}

LINKS = tuple(ROBOLAB_RUNTIME)
GEOM_TYPES = {"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}


# ------------------------------------------------------------------------------ linear algebra
def local_A_t(prim):
    """Local transform of @prim relative to its parent as p_parent = A @ p_local + t.

    USD's Gf.Matrix4d is row-vector: the translation is the last ROW and the linear part acts on the
    LEFT of a row vector, so the column-vector linear part is its transpose.
    """
    m = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return M[:3, :3].T.copy(), M[3, :3].copy()


def chain_to(link_prim, geom_prim):
    """Compose the full geom-local -> link-local transform, for any nesting depth."""
    A, t, p = np.eye(3), np.zeros(3), geom_prim
    while p and p != link_prim:
        Ap, tp = local_A_t(p)
        A, t = Ap @ A, Ap @ t + tp
        p = p.GetParent()
    assert p == link_prim, f"{geom_prim.GetPath()} is not under {link_prim.GetPath()}"
    return A, t


def tri_mesh(prim):
    g = UsdGeom.Mesh(prim)
    pts = np.asarray(g.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(g.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(g.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    tris, o = [], 0
    for c in counts:
        f = idx[o:o + c]
        tris.extend((f[0], f[k], f[k + 1]) for k in range(1, c - 1))
        o += c
    return pts, np.asarray(tris, dtype=np.int64)


def volume_and_centroid(pts, tris):
    """Signed-tetrahedron volume and centroid of a closed triangle mesh, in the mesh frame."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    v6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = v6.sum() / 6.0
    com = (((a + b + c) / 4.0) * (v6 / 6.0)[:, None]).sum(axis=0) / vol
    return abs(vol), com


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# ------------------------------------------------------------------------------------- the work
def collision_geoms(link_prim):
    """[(geom_prim, is_under_collision)] for every geom under @link_prim, using OmniGibson's own
    rule: a CollisionAPI anywhere on the way down marks everything beneath it as collision."""
    out = []

    def walk(prim, is_col):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or "PhysxCollisionAPI" in prim.GetAppliedSchemas():
            is_col = True
        if prim.GetTypeName() in GEOM_TYPES:
            if is_col:
                out.append(prim)
            return
        for c in prim.GetChildren():
            walk(c, is_col)

    walk(link_prim, False)
    return out


def compute(src):
    st = Usd.Stage.Open(src)
    assert st, f"could not open {src}"
    links = {}
    for p in Usd.PrimRange(st.GetPseudoRoot()):
        if p.GetName() in LINKS and "Joint" not in p.GetTypeName():
            links.setdefault(p.GetName(), p)
    missing = set(LINKS) - set(links)
    assert not missing, f"links not found in {src}: {sorted(missing)}"

    recs = {}
    for ln in LINKS:
        lp = links[ln]
        geoms = collision_geoms(lp)
        assert geoms, f"{ln} has no collision geoms"
        parts, V, C = [], 0.0, np.zeros(3)
        for g in geoms:
            A, t = chain_to(lp, g)
            scale = np.linalg.norm(A, axis=0)
            R = A / scale
            assert abs(R @ R.T - np.eye(3)).max() < 1e-9, f"{g.GetPath()} has a sheared transform"
            pts, tris = tri_mesh(g)
            vol, com = volume_and_centroid(pts * scale, tris)
            com_link = R @ com + t
            V += vol
            C += vol * com_link
            parts.append(dict(prim=g, vol=vol, com_link=com_link, R=R, t=t, scale=scale))
        C /= V

        I = np.asarray(ROBOLAB_RUNTIME[ln]["inertia"], dtype=np.float64).reshape(3, 3)
        I = 0.5 * (I + I.T)                       # symmetrise: the runtime dump is float32
        evals, evecs = np.linalg.eigh(I)
        if np.linalg.det(evecs) < 0:
            evecs[:, 0] *= -1.0
        recs[ln] = dict(mass=ROBOLAB_RUNTIME[ln]["mass"], com=C, diag=evals,
                        quat=mat_to_quat(evecs), parts=parts, volume=V,
                        recon=float(np.abs(evecs @ np.diag(evals) @ evecs.T - I).max()))
    return st, links, recs


# ------------------------------------------------------------------------------------- emitters
def f3(v):
    return "(" + ", ".join(repr(float(x)) for x in v) + ")"


def mass_body(r):
    return (f'        float physics:mass = {float(r["mass"])!r}\n'
            f'        point3f physics:centerOfMass = {f3(r["com"])}\n'
            f'        float3 physics:diagonalInertia = {f3(r["diag"])}\n'
            f'        quatf physics:principalAxes = {f3(r["quat"])}\n')


def anchor_body(r):
    """Move each collision Xform's translate/orient down onto the Mesh, leaving the Xform identity.

    The composed matrix is unchanged: the original is  T_x . R_x . S_x . S_m  and this writes
    T_x . R_x . (S_x * S_m) on the mesh with the Xform at identity -- the two scales are adjacent
    diagonals in both, so no commutation is involved and the product is the same matrix.
    """
    body = ""
    for p in r["parts"]:
        geom = p["prim"]
        xf = geom.GetParent()
        assert xf.HasAPI(UsdPhysics.CollisionAPI), \
            f"{geom.GetPath()}: expected the collision API on the immediate parent Xform"
        assert xf.GetTypeName() == "Xform", f"{xf.GetPath()} is not an Xform"
        # the transform that has to move down is exactly the parent's own local transform
        Ax, tx = local_A_t(xf)
        sx = np.linalg.norm(Ax, axis=0)
        Rx = Ax / sx
        Am, tm = local_A_t(geom)
        sm = np.linalg.norm(Am, axis=0)
        assert np.allclose(Am / sm, np.eye(3), atol=1e-12), \
            f"{geom.GetPath()} already carries a rotation; re-anchoring is not a pure move"
        assert np.allclose(tm, 0.0, atol=1e-15), \
            f"{geom.GetPath()} already carries a translation; re-anchoring is not a pure move"
        qx = mat_to_quat(Rx)
        body += (
            f'        over "{xf.GetName()}"\n        {{\n'
            f'            double3 xformOp:translate = (0, 0, 0)\n'
            f'            quatd xformOp:orient = (1, 0, 0, 0)\n'
            f'            over "{geom.GetName()}"\n            {{\n'
            f'                double3 xformOp:translate = {f3(tx)}\n'
            f'                quatd xformOp:orient = {f3(qx)}\n'
            f'                float3 xformOp:scale = {f3(sx * sm)}\n'
            f'                uniform token[] xformOpOrder = ["xformOp:translate", '
            f'"xformOp:orient", "xformOp:scale"]\n'
            f'            }}\n'
            f'        }}\n')
    return body


def link_over(ln, r, do_mass, do_anchor):
    """ONE `over` per link -- two `over "base_link"` blocks in the same layer is a parse error."""
    meta = ('    over "%s" (\n        prepend apiSchemas = ["PhysicsMassAPI"]\n    )\n' % ln
            if do_mass else '    over "%s"\n' % ln)
    body = (mass_body(r) if do_mass else "") + (anchor_body(r) if do_anchor else "")
    return f'{meta}    {{\n{body}    }}\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "realm", "robots", "panda_robotiq",
                                                  "droid_robolab_v2.usd"),
                    help="HOST path of the shipped asset, opened to compute the numbers")
    ap.add_argument("--sublayer", default="/app/realm/robots/panda_robotiq/droid_robolab_v2.usd",
                    help="path written into the variant's subLayers, i.e. as the CONTAINER sees it")
    ap.add_argument("--out", default="./tmp/variants")
    ap.add_argument("--name", default=None)
    ap.add_argument("--mass", action="store_true", help="author MassAPI on the nine gripper links")
    ap.add_argument("--anchor", action="store_true",
                    help="re-anchor the collision Xforms so frame='parent' reaches the link")
    args = ap.parse_args()
    if not (args.mass or args.anchor):
        raise SystemExit("nothing to write -- pass --mass and/or --anchor")

    st, links, recs = compute(os.path.abspath(args.src))

    print(f"source {os.path.abspath(args.src)}")
    print(f"{'link':<22} {'mass (kg)':>20}  {'centerOfMass (mm)':>34}  diagonalInertia")
    for ln in LINKS:
        r = recs[ln]
        c = r["com"] * 1000.0
        print(f"  {ln:<20} {r['mass']:>20.15g}  "
              f"[{c[0]:9.4f},{c[1]:9.4f},{c[2]:9.4f}]  "
              f"[{r['diag'][0]:.6e}, {r['diag'][1]:.6e}, {r['diag'][2]:.6e}]")
        print(f"  {'':<20} principalAxes(w,x,y,z) = "
              f"[{r['quat'][0]:.9f}, {r['quat'][1]:.9f}, {r['quat'][2]:.9f}, {r['quat'][3]:.9f}]"
              f"   |c|={np.linalg.norm(r['com']) * 1000:.4f} mm  recon={r['recon']:.2e}")
        d = np.sort(r["diag"])
        assert d[0] > 0, f"{ln}: non-positive principal inertia {r['diag']}"
        if d[0] + d[1] < d[2]:
            print(f"  {'':<20} *** WARNING: triangle inequality violated for {ln}")

    body = "".join(link_over(ln, recs[ln], args.mass, args.anchor) for ln in LINKS)

    name = args.name or ("mass_authored" if args.mass else "mass") + \
        ("_anchor" if args.anchor else "")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{name}.usda")
    text = f'''#usda 1.0
(
    """Mass-property variant of droid_robolab_v2.usd -- generated by
    scripts/debug_probes/make_mass_variant.py.  mass={args.mass} anchor={args.anchor}

    Every prim, mesh, joint and material comes from the sublayer, which is the shipped asset,
    unmodified. This layer authors ONLY:
      * MassAPI + physics:mass / centerOfMass / diagonalInertia / principalAxes on the nine
        GRIPPER links, so PhysX derives no mass property for them;
      * (with --anchor) the collision Xform -> Mesh transform split, which leaves every collision
        shape's world pose bit-identical and only changes which prim carries the transform.
    The seven panda_joint* DOFs, the arm links and the whole arm_0 block are untouched.
    """
    defaultPrim = "panda"
    subLayers = [
        @{args.sublayer}@
    ]
)

over "panda"
{{
{body}}}
'''
    with open(path, "w") as f:
        f.write(text)
    print(f"\nwrote {path}  ({len(text)} bytes)")


if __name__ == "__main__":
    main()
