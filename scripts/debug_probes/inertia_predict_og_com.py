"""Predict, statically, the centre of mass OmniGibson will WRITE onto each pad link.

The suspicion
-------------
`RigidPrim.update_meshes()` (omnigibson/prims/rigid_prim.py:245-249) does

    volume, com = get_mesh_volume_and_com(mesh.prim)
    # "We need to transform the volume and CoM from the mesh's local frame to the link's local frame"
    local_pos, local_orn = mesh.get_position_orientation(frame="parent")
    coms.append(T.quat2mat(local_orn) @ (com * mesh.scale) + local_pos)

and then assigns the volume-weighted result to `center_of_mass`, i.e. `set_coms()` on the physics
view. The comment says "to the link's local frame", but `frame="parent"` is documented in
xform_prim.py:267 as "get position relative to the object parent" -- the IMMEDIATE parent prim, one
level up.

That is only the link frame when the geom is a direct child of the link. In this asset it is not:
the collision APIs sit on an *Xform* (`Defeatured_2F_85_PAD_OPEN_*step_01`) whose child is the
`Mesh`, and GEOM_TYPES = {Sphere, Cube, Cone, Cylinder, Mesh} (utils/constants.py:100) excludes
Xform -- so the geom OmniGibson wraps is the Mesh, and its "parent" is that Xform, not the link.
The Xform -> link transform is therefore dropped.

That matters here specifically because `fix_robolab_link_origins.py` moved each pad's link origin
onto its geometry centroid, which makes the dropped Xform -> link transform large (~134 mm) on
REALM's asset and ~0 on the unfixed robolab one.

This computes both answers without a GPU:
    com_true : volume-weighted centroid over the collision meshes, correctly composed all the way
               up to the LINK frame
    com_og   : the same sum with the Xform -> link step omitted, i.e. what the code above produces

Volume and centroid come from the standard signed-tetrahedron sum over the triangulated mesh, which
is what trimesh (and hence get_mesh_volume_and_com) uses.

Runs on stock pxr (pip usd-core).
"""

import json
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

PADS = ("left_inner_finger", "right_inner_finger")


def tri_volume_com(points, counts, indices):
    """Signed-tet volume and centroid of a closed triangle soup, in the mesh's own frame."""
    P = np.asarray(points, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    tris, off = [], 0
    for c in counts:
        for k in range(1, c - 1):                      # fan-triangulate any n-gon
            tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    if not tris:
        return 0.0, np.zeros(3)
    T = np.asarray(tris, dtype=np.int64)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))    # 6*signed volume of tet (0,a,b,c)
    V = vol6.sum() / 6.0
    if abs(V) < 1e-18:
        return 0.0, P.mean(axis=0)
    cent = (a + b + c) / 4.0                            # tet centroid, apex at origin
    C = (cent * vol6[:, None]).sum(axis=0) / vol6.sum()
    return float(V), C


def m4(prim, cache):
    return Gf.Matrix4d(cache.GetLocalToWorldTransform(prim))


def rel_xform(child, ancestor, cache):
    """child's transform expressed in ancestor's frame."""
    return m4(child, cache) * m4(ancestor, cache).GetInverse()


def link_prim(stage, name):
    for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
    return None


def analyse(path, label):
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    print(f"\n{'=' * 106}\n{label}: {path}\n{'=' * 106}")
    for pad in PADS:
        link = link_prim(stage, pad)
        if link is None:
            continue
        rec = {"meshes": []}
        num_true = np.zeros(3)
        num_og = np.zeros(3)
        den = 0.0
        for holder in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            # the prim that carries the collision API -- an Xform here, not the Mesh
            if not (holder.HasAPI(UsdPhysics.CollisionAPI)
                    or holder.HasAPI(UsdPhysics.MeshCollisionAPI)):
                continue
            for mesh in Usd.PrimRange(holder, Usd.TraverseInstanceProxies()):
                if mesh.GetTypeName() != "Mesh":
                    continue
                mg = UsdGeom.Mesh(mesh)
                pts = mg.GetPointsAttr().Get()
                cts = mg.GetFaceVertexCountsAttr().Get()
                idx = mg.GetFaceVertexIndicesAttr().Get()
                if pts is None or cts is None or idx is None:
                    continue
                V, C = tri_volume_com([tuple(p) for p in pts], cts, idx)
                # correct: mesh frame -> LINK frame, composing every level
                X_ml = rel_xform(mesh, link, cache)
                com_true = X_ml.Transform(Gf.Vec3d(*C))
                # what OmniGibson computes: mesh frame -> its IMMEDIATE PARENT only
                parent = mesh.GetParent()
                X_mp = rel_xform(mesh, parent, cache)
                com_og = X_mp.Transform(Gf.Vec3d(*C))
                Vabs = abs(V)
                num_true += Vabs * np.array([float(x) for x in com_true])
                num_og += Vabs * np.array([float(x) for x in com_og])
                den += Vabs
                rec["meshes"].append({
                    "mesh": mesh.GetPath().pathString,
                    "collision_holder": holder.GetPath().pathString,
                    "holder_type": str(holder.GetTypeName()),
                    "parent_of_mesh": parent.GetPath().pathString,
                    "parent_is_link": parent == link,
                    "volume_m3": V,
                    "com_mesh_frame": [float(x) for x in C],
                    "com_link_frame": [float(x) for x in com_true],
                    "com_og_frame": [float(x) for x in com_og],
                })
        if den <= 0:
            continue
        ct, co = num_true / den, num_og / den
        rec["com_true_link_frame_mm"] = [float(x * 1000) for x in ct]
        rec["com_og_written_mm"] = [float(x * 1000) for x in co]
        rec["error_mm"] = float(np.linalg.norm(ct - co) * 1000)
        rec["total_volume_m3"] = den
        rec["mass_at_1000_kgm3"] = den * 1000.0
        print(f"\n  {pad}")
        for m in rec["meshes"]:
            print(f"      mesh {m['mesh'].rsplit('/', 1)[-1]:<42} vol={m['volume_m3']:.6e} m^3")
            print(f"        collision API on : {m['holder_type']} "
                  f"{m['collision_holder'].rsplit('/', 1)[-1]}")
            print(f"        mesh parent is the link? {m['parent_is_link']}")
        print(f"      com CORRECT (link frame) = "
              f"[{ct[0]*1000:9.4f}, {ct[1]*1000:9.4f}, {ct[2]*1000:9.4f}] mm")
        print(f"      com OMNIGIBSON WRITES    = "
              f"[{co[0]*1000:9.4f}, {co[1]*1000:9.4f}, {co[2]*1000:9.4f}] mm")
        print(f"      displacement introduced  = {rec['error_mm']:9.4f} mm")
        print(f"      total collision volume   = {den:.6e} m^3  "
              f"(-> {den * 1000:.6f} kg at the PhysX default 1000 kg/m^3)")
        out[pad] = rec
    return out


def main(f_rl, f_rm, out_path):
    res = {"robolab": analyse(f_rl, "ROBOLAB (unfixed origins)"),
           "realm": analyse(f_rm, "REALM (origins moved onto geometry)")}
    print(f"\n{'=' * 106}\nSUMMARY\n{'=' * 106}")
    print(f"  {'asset':10s} {'pad':22s} {'|com_true|':>12s} {'|com_og|':>12s} {'displacement':>14s}")
    for k in ("robolab", "realm"):
        for pad, r in res[k].items():
            t = np.linalg.norm(r["com_true_link_frame_mm"])
            o = np.linalg.norm(r["com_og_written_mm"])
            print(f"  {k:10s} {pad:22s} {t:9.4f} mm {o:9.4f} mm {r['error_mm']:11.4f} mm")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {out_path}")
    worst = max((r["error_mm"] for k in res for r in res[k].values()), default=0.0)
    print(f"OG_COM_{'DISPLACED' if worst > 1.0 else 'OK'} worst={worst:.4f} mm")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
