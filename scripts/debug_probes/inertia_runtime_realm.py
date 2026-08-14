"""Read REALM's RUNTIME gripper mass properties and close the omega^2*I convergence test.

Why a runtime probe at all
--------------------------
The static USD diff (scripts/debug_probes/inertia_dump.py + inertia_diff.py) already settled the
authored side: NEITHER asset authors physics:mass, physics:density, physics:centerOfMass,
physics:diagonalInertia or physics:principalAxes, and neither applies MassAPI. So there is nothing
to diff -- PhysX DERIVES every mass property from the collision shapes, and those agree between the
two assets to 5.4e-11 m in world space with identical approximations. `ship_inertia_diff.py`'s
convergence test therefore cannot fire; it reads authored attributes that are absent on both sides.

The realised numbers can still differ, for one concrete reason the static diff cannot see:
OmniGibson REWRITES mass properties at load, and Isaac Lab does not.
  * rigid_prim.py:120  applies UsdPhysics.MassAPI to every rigid body
  * rigid_prim.py:276-281  computes a volume-weighted centroid over the link's COLLISION MESHES,
    in the LINK frame, and assigns it to `center_of_mass` -- which calls set_coms() on the physics
    view, overriding whatever PhysX derived from the actual convex hulls / decompositions.
  * rigid_prim.py:224-228  overwrites every collider's contact/rest offset
  * rigid_prim.py:250-253  may swap an oblong mesh's approximation to boundingCube
A COM override matters because the pad's effective inertia about its pivot is I_com + m*d^2, and d
is measured from the COM to the pivot axis. Move the COM and you move the constraint's effective
inertia without touching a single authored number.

REALM's pads are the sharpest possible test of this, because `fix_robolab_link_origins.py` has
already put each pad's link ORIGIN exactly on its geometry centroid (verified statically: geom bbox
mid - origin = 0.000 mm). So if PhysX/OmniGibson agree with that centroid, `get_coms()` must come
back at ~0 in the link frame. Whatever it actually returns is the answer.

What it reports
---------------
Per gripper link: runtime mass, COM (link frame), full 3x3 inertia; and for the two pads the
effective inertia about the pivot axis, I_eff = n^T I_com n + m*d_perp^2, which is the quantity that
enters the mimic constraint's realised stiffness k ~ omega^2 * I_eff. Then

    nf_eq = nf_robolab * sqrt(I_eff_robolab / I_eff_realm)

against RoboLab's already-captured runtime numbers in wrapdiff_robolab_runtime.json. The empirical
curl ladder is nf = 100..200, so a nf_eq landing there confirms the mechanism from two directions;
a nf_eq near 1000 kills it.

    ./scripts/clara/interactive/rr python -u /app/scripts/debug_probes/inertia_runtime_realm.py

Isaac exits 139 at teardown; grep INERTIA_RT_OK / INERTIA_NF_EQ, never the exit code. JSON is
written before the summary is printed.
"""
import argparse
import json
import os
import traceback

import numpy as np

np.set_printoptions(precision=9, suppress=False, linewidth=220)

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze/inertia_runtime_realm.json")
ap.add_argument("--robolab-json", default="/logs/gripper_squeeze/wrapdiff_robolab_runtime.json")
ap.add_argument("--nf-robolab", type=float, default=1000.0)
args = ap.parse_args()

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

OUT = {"robot": args.robot}
GRIPPER_LINKS = ("base_link", "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger",
                 "right_outer_finger", "left_inner_finger", "right_inner_finger",
                 "left_inner_knuckle", "right_inner_knuckle")
PADS = ("left_inner_finger", "right_inner_finger")
PAD_JOINT = {"left_inner_finger": "left_inner_finger_joint",
             "right_inner_finger": "right_inner_finger_joint"}


def hdr(s):
    print(f"\n{'=' * 104}\n{s}\n{'=' * 104}", flush=True)


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception:
        try:
            import warp as wp
            return np.asarray(wp.to_torch(x).detach().cpu(), dtype=np.float64)
        except Exception:
            return None


def save():
    """Write before printing -- Isaac's teardown hang makes a time-limit kill routine."""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1, default=str)


print(f"[inertia-rt] robot={args.robot} task={args.task_cfg}", flush=True)
set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=True, rendering_mode="rt", robot=args.robot,
)
robot = env.robot
print(f"[inertia-rt] robot prim_path={robot.prim_path}", flush=True)
OUT["prim_path"] = str(robot.prim_path)

# ------------------------------------------------------------------ the articulation tensor view
# OG's RigidDynamicPrim exposes .mass / .center_of_mass but NOT .inertia (that is what the earlier
# wrapdiff run hit: "'RigidDynamicPrim' object has no attribute 'inertia'"). The inertia lives on
# the raw PhysX tensor view, reached here the same way Isaac Lab's root_physx_view is on the
# RoboLab side, so the two sets of numbers are the same quantity in the same convention:
# inertia about the link COM, expressed in the link frame, flattened 3x3.
view = None
for getter in (
    lambda: robot.joints["finger_joint"]._articulation_view,
    lambda: robot._articulation_view,
    lambda: robot.root_link._rigid_prim_view,
):
    try:
        v = getter()
        if v is not None:
            view = v
            print(f"[inertia-rt] articulation view via {getter.__doc__ or 'lambda'}: {type(v)}",
                  flush=True)
            break
    except Exception as e:
        print(f"[inertia-rt] view getter failed: {type(e).__name__}: {e}", flush=True)

phys = getattr(view, "_physics_view", None) if view is not None else None
print(f"[inertia-rt] view={type(view).__name__ if view is not None else None} "
      f"physics_view={type(phys).__name__ if phys is not None else None}", flush=True)

link_names, masses, inertias, coms = None, None, None, None
for src_name, src in (("physics_view", phys), ("view", view)):
    if src is None:
        continue
    try:
        masses = _np(src.get_masses()) if hasattr(src, "get_masses") else masses
        inertias = _np(src.get_inertias()) if hasattr(src, "get_inertias") else inertias
        coms = _np(src.get_coms()) if hasattr(src, "get_coms") else coms
        md = getattr(src, "_metadata", None) or getattr(src, "shared_metatype", None)
        if md is not None and hasattr(md, "link_names"):
            link_names = list(md.link_names)
        print(f"[inertia-rt] from {src_name}: masses={None if masses is None else masses.shape} "
              f"inertias={None if inertias is None else inertias.shape} "
              f"coms={None if coms is None else coms.shape} links={len(link_names or [])}",
              flush=True)
    except Exception as e:
        print(f"[inertia-rt] {src_name} readback partial: {type(e).__name__}: {e}", flush=True)

if not link_names:
    link_names = list(robot.links.keys())
    print(f"[inertia-rt] falling back to robot.links order: {link_names}", flush=True)
OUT["link_names"] = [str(x) for x in link_names]
OUT["shapes"] = {k: (None if v is None else list(v.shape))
                 for k, v in (("masses", masses), ("inertias", inertias), ("coms", coms))}
save()


def row(arr, idx):
    """Pull link idx out of a (1, nlinks, k) or (nlinks, k) tensor."""
    if arr is None:
        return None
    a = arr[0] if arr.ndim == 3 else arr
    if idx >= len(a):
        return None
    return np.asarray(a[idx], dtype=np.float64).reshape(-1)


# ------------------------------------------------------------------ per-link runtime properties
hdr("REALM RUNTIME GRIPPER LINK MASS PROPERTIES")
bodies = {}
name_to_idx = {str(n).split("/")[-1]: i for i, n in enumerate(link_names)}
for ln in GRIPPER_LINKS:
    e = {}
    i = name_to_idx.get(ln)
    e["view_index"] = i
    if i is not None:
        m = row(masses, i)
        inr = row(inertias, i)
        cm = row(coms, i)
        e["mass_view"] = float(m[0]) if m is not None and len(m) else None
        e["inertia"] = [float(x) for x in inr] if inr is not None else None
        e["com_view"] = [float(x) for x in cm[:3]] if cm is not None and len(cm) >= 3 else None
    # OmniGibson's own wrappers, which is what REALM code actually reads
    try:
        lk = robot.links[ln]
        e["mass_og"] = float(_np(lk.mass))
        e["density_og"] = float(_np(lk.density))
        e["com_og"] = [float(x) for x in _np(lk.center_of_mass).reshape(-1)[:3]]
        p, q = lk.get_position_orientation()
        e["link_world_pos"] = [float(x) for x in _np(p).reshape(-1)]
        e["link_world_quat"] = [float(x) for x in _np(q).reshape(-1)]
    except Exception as ex:
        e["og_error"] = f"{type(ex).__name__}: {ex}"
    bodies[ln] = e
    print(f"  {ln:<22} mass={e.get('mass_og')} com_link_frame={e.get('com_og')}", flush=True)
    if e.get("inertia"):
        d = e["inertia"]
        print(f"  {'':<22} inertia_diag=[{d[0]:.9e}, {d[4]:.9e}, {d[8]:.9e}]", flush=True)
OUT["bodies_runtime"] = bodies
save()

# ------------------------------------------------------------------ does OG displace the COM?
hdr("IS THE PAD COM WHERE THE GEOMETRY IS?")
print("  fix_robolab_link_origins.py put each pad's link ORIGIN exactly on its geometry centroid")
print("  (verified statically: geom bbox mid - link origin = 0.000 mm). So a COM derived from that")
print("  same geometry must read ~0 in the LINK frame. A large value means it was overridden.")
for ln in PADS:
    e = bodies.get(ln, {})
    for k in ("com_og", "com_view"):
        c = e.get(k)
        if c:
            print(f"  {ln:<22} {k:<9} = [{c[0] * 1000:9.4f}, {c[1] * 1000:9.4f}, "
                  f"{c[2] * 1000:9.4f}] mm   |.|={np.linalg.norm(c) * 1000:8.4f} mm", flush=True)

# ------------------------------------------------------------------ pivot geometry, live
hdr("PAD PIVOT ANCHOR AND AXIS, LIVE")
pivots = {}
for ln in PADS:
    jn = PAD_JOINT[ln]
    rec = {}
    try:
        jp = lazy.omni.isaac.core.utils.prims.get_prim_at_path(
            f"{robot.prim_path}/{ln}/{jn}")
        if not (jp and jp.IsValid()):
            for p in lazy.pxr.Usd.PrimRange(
                    lazy.omni.isaac.core.utils.prims.get_prim_at_path(robot.prim_path)):
                if p.GetName() == jn:
                    jp = p
                    break
        rec["prim_path"] = str(jp.GetPath())
        for a in ("physics:localPos0", "physics:localPos1", "physics:axis"):
            at = jp.GetAttribute(a)
            v = at.Get() if at else None
            rec[a] = (str(v) if isinstance(v, str) else
                      ([float(x) for x in v] if v is not None else None))
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    pivots[ln] = rec
    print(f"  {ln:<22} {rec}", flush=True)
OUT["pivots"] = pivots
save()

# ------------------------------------------------------------------ effective inertia about pivot
hdr("EFFECTIVE INERTIA ABOUT THE PAD PIVOT  --  I_eff = n' I_com n + m * d_perp^2")
print("  This, not a raw diagonal element, is what omega^2 * I means for the mimic constraint on")
print("  *_inner_finger_joint: the pad's resistance to rotating about ITS OWN pivot axis.")


def eff_inertia(mass, inertia9, com_link, anchor_link, axis_letter):
    """I about the pivot axis, all quantities in the LINK frame."""
    if mass is None or inertia9 is None or anchor_link is None:
        return None
    I = np.asarray(inertia9, dtype=np.float64).reshape(3, 3)
    n = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]),
         "Z": np.array([0, 0, 1.0])}[str(axis_letter or "X").upper()]
    n = n / np.linalg.norm(n)
    c = np.asarray(com_link if com_link is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    a = np.asarray(anchor_link, dtype=np.float64)
    r = c - a                        # COM relative to a point on the axis
    r_perp = r - np.dot(r, n) * n    # component perpendicular to the axis
    return float(n @ I @ n + mass * float(np.dot(r_perp, r_perp)))


realm_eff = {}
for ln in PADS:
    e = bodies.get(ln, {})
    pv = pivots.get(ln, {})
    Ie = eff_inertia(e.get("mass_og"), e.get("inertia"), e.get("com_og"),
                     pv.get("physics:localPos1"), pv.get("physics:axis"))
    realm_eff[ln] = Ie
    print(f"  {ln:<22} I_eff_realm = {Ie!r}", flush=True)
OUT["realm_I_eff"] = realm_eff
save()

# ------------------------------------------------------------------ the convergence test
hdr("CONVERGENCE TEST:  nf_eq = nf_robolab * sqrt(I_robolab / I_realm)")
conv = {}
try:
    RL = json.load(open(args.robolab_json))
    rl_bodies = RL.get("bodies_runtime", {})
    OUT["robolab_source"] = args.robolab_json
    for ln in PADS:
        r = rl_bodies.get(ln, {})
        m = bodies.get(ln, {})
        I_rl9, mass_rl = r.get("inertia"), r.get("mass")
        I_rm9, mass_rm = m.get("inertia"), m.get("mass_og")
        rec = {"mass_robolab": mass_rl, "mass_realm": mass_rm,
               "inertia_robolab": I_rl9, "inertia_realm": I_rm9}
        print(f"\n  {ln}")
        if mass_rl and mass_rm:
            print(f"      mass    robolab={mass_rl:.12g}  realm={mass_rm:.12g}  "
                  f"ratio={mass_rm / mass_rl:.9f}")
            rec["mass_ratio"] = mass_rm / mass_rl
        if I_rl9 and I_rm9:
            A = np.asarray(I_rl9, float).reshape(3, 3)
            B = np.asarray(I_rm9, float).reshape(3, 3)
            print(f"      I_com robolab diag = [{A[0,0]:.9e}, {A[1,1]:.9e}, {A[2,2]:.9e}]")
            print(f"      I_com realm   diag = [{B[0,0]:.9e}, {B[1,1]:.9e}, {B[2,2]:.9e}]")
            den = np.maximum(np.abs(A), np.abs(B))
            rel = np.where(den > 1e-18, np.abs(A - B) / np.where(den > 1e-18, den, 1), 0.0)
            print(f"      max |rel diff| over the full 3x3 tensor: {100 * rel.max():.6f} %")
            rec["max_rel_pct_tensor"] = float(100 * rel.max())
            # same effective-inertia reduction on RoboLab's numbers. Its pad link frame differs
            # from REALM's only by the origin move, and I_com is origin-independent, so the
            # tensor is directly comparable; the parallel-axis term uses RoboLab's own anchor.
            pv = pivots.get(ln, {})
            Ie_rl = eff_inertia(mass_rl, I_rl9, r.get("com"), pv.get("physics:localPos1"),
                                pv.get("physics:axis"))
            Ie_rm = realm_eff.get(ln)
            rec["I_eff_robolab_sameframe"] = Ie_rl
            rec["I_eff_realm"] = Ie_rm
            # axis-projected I_com is the frame-safe comparison; report nf_eq from BOTH
            n = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]),
                 "Z": np.array([0, 0, 1.0])}[str(pv.get("physics:axis") or "X").upper()]
            axA, axB = float(n @ A @ n), float(n @ B @ n)
            rec["I_axis_robolab"], rec["I_axis_realm"] = axA, axB
            if axB > 0:
                nf_axis = args.nf_robolab * float(np.sqrt(axA / axB))
                rec["nf_eq_axis"] = nf_axis
                print(f"      I about pivot axis: robolab={axA:.9e}  realm={axB:.9e}  "
                      f"ratio={axB / axA:.9f}")
                print(f"      INERTIA_NF_EQ {ln} axis-projected nf_eq = {nf_axis:.4f}")
            if Ie_rl and Ie_rm and Ie_rm > 0:
                nf_eff = args.nf_robolab * float(np.sqrt(Ie_rl / Ie_rm))
                rec["nf_eq_effective"] = nf_eff
                print(f"      I_eff (with parallel axis): robolab={Ie_rl:.9e}  realm={Ie_rm:.9e}")
                print(f"      INERTIA_NF_EQ {ln} effective   nf_eq = {nf_eff:.4f}")
        conv[ln] = rec
except Exception:
    conv["error"] = traceback.format_exc()
    print(conv["error"], flush=True)
OUT["convergence"] = conv
save()

hdr("VERDICT")
nfs = [v.get("nf_eq_axis") for v in conv.values()
       if isinstance(v, dict) and v.get("nf_eq_axis")]
if nfs:
    lo, hi = min(nfs), max(nfs)
    near = 100.0 <= lo <= 200.0 or 100.0 <= hi <= 200.0
    print(f"  nf_eq range over both pads: {lo:.3f} .. {hi:.3f}")
    print(f"  empirical curl ladder:      nf = 100 .. 200")
    print(f"  INERTIA_VERDICT {'CONVERGES' if near else 'NO_CONVERGENCE'} "
          f"nf_eq_lo={lo:.4f} nf_eq_hi={hi:.4f}")
    OUT["verdict"] = {"nf_eq_lo": lo, "nf_eq_hi": hi, "converges": bool(near)}
else:
    print("  INERTIA_VERDICT INCONCLUSIVE -- no nf_eq computed, see convergence.error")
    OUT["verdict"] = {"converges": None}
save()
print(f"\n  wrote {args.out}")
print("INERTIA_RT_OK", flush=True)
og.shutdown()
