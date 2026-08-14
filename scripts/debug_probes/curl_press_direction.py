"""Press the OPEN fingertips onto the table and measure WHICH WAY the tips rotate.

The question this exists for
---------------------------
`gripper_squeeze_compliance.py`'s mimic-`naturalFrequency` ladder reported |flex| -- a MAGNITUDE --
measured in a SQUEEZE (a 30 mm cube between the pads, loaded face-on from the inside). Neither of
those answers the actual target behaviour, which is:

    with the fingertips pressed against a SURFACE, the tips curl INWARD -- the tip rotates toward
    the closing axis, the way an underactuated 2F-85 four-bar is meant to.

So: a press load case (loaded from below, along the finger's long axis) and a SIGNED measurement.
If the tips splay OUTWARD at a rung, that rung is a failure however visible it is.

Why the jaws are OPEN and not shut
----------------------------------
`ee_press_compliance.py` pressed with the jaws SHUT. At full closure the two pads are touching each
other, so an inward curl is blocked by the opposing pad *geometrically* -- that load case cannot show
this behaviour whatever the physics says, and it is the one that produced the ~0.13 mm "both assets
are rigid" number. Here the default state is OPEN (`--states open,closed` runs both), which is also
what the real gesture is: an open hand pressed down onto a table.

Sign convention -- tied to geometry, never to a raw joint value
---------------------------------------------------------------
A joint's polarity is an asset detail (and `physics:axis` is 'Z' on all six of these joints while the
mimic instance is `rotX`), so nothing below is read off a joint sign. One right-handed frame is built
ONCE in the `panda_link8` frame from the unloaded open pose and carried with the hand:

    AXIS  the closing axis, from finger link FL[0] to FL[1] ("left -> right"; whichever link is
          which, FL[0] is at negative AXIS by construction)
    LONG  flange -> pad midpoint, orthogonalised against AXIS ("down the fingers")
    H     AXIS x LONG -- the normal of the closing plane, i.e. the direction the finger hinges rotate
          about, and the direction the close-up camera looks along

Two INDEPENDENT observables, both of which must agree for the answer to stand:

 1. TIP SEPARATION. One body-fixed material point per finger, chosen once at rest as the inboard-most
    point of the distal quarter of that finger's collision hull. `tip_sep` is their separation along
    AXIS. **Shrinking = inward.** The same measurement on the PROXIMAL quarter (`base_sep`) is the
    control: tips-in-with-base-unchanged is a curl, both-shrinking-together is the whole jaw closing,
    which is a different thing.
 2. PAD FACE NORMAL. For each pad, the body-fixed direction that coincided with the inboard closing
    axis at rest. Rotating the pad about H by psi sends AXIS -> AXIS cos psi + LONG sin psi and sends
    a point at +LONG from the hinge toward -AXIS. So for FL[0] (inboard = +AXIS) inward means psi<0,
    and for FL[1] (inboard = -AXIS) inward means psi>0; in BOTH cases the inboard normal acquires a
    NEGATIVE LONG component, i.e. the pad face tips back toward the flange. Hence one signed number,
    the same sense for both fingers:

        curl_deg = -degrees(asin(n_inboard . LONG)),   POSITIVE = INWARD

    and the pad's rigid rotation about H is reported the same way (`rot_deg`), computed from the link
    orientation rather than from the normal, plus `rot_par_deg`, the pad's rotation relative to its
    OWN PARENT LINK -- that last one is the `*_inner_finger_joint` pivot's own contribution, and its
    sign against the joint value's is what maps joint polarity to the geometry (printed, never
    assumed).

Everything is measured relative to `panda_link8`, so arm motion is removed by construction, and every
rung takes its OWN unloaded rest reference at the hover pose because softening the mimic constraint
can move the unloaded pose too.

Load case: EE control (`DroidEndEffectorController`, absolute_pose) descends a clear column over the
table -- column choice, orientation round-trip check and traverse are `ee_press_compliance.py`'s,
which is the proven code for this -- then keeps commanding a sub-surface z so the arm keeps pushing.

    ./scripts/clara/interactive/rr python -u /app/scripts/debug_probes/curl_press_direction.py \
        --tag curl_A --rungs "nf1000a=1000/0.05,nf100=100/0.05" --states open

Isaac exits 139 at teardown regardless of outcome: grep CURL_PROBE_OK, never the exit code. Results
are flushed after every (rung, state) so an allocation that expires early still leaves numbers.
"""
import argparse
import json
import os

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=220)

ap = argparse.ArgumentParser()
ap.add_argument("--load", default="tip", choices=("tip", "ee"),
                help="how the press load is applied. 'tip' (default): the arm holds reset_qpos under "
                     "JOINT control and a PINNED object is ramped UP into one fingertip, 0.5 mm per "
                     "step, until contact is detected and then past it -- no IK, no arm motion, and "
                     "the contact force is read from the contact view. 'ee': EE control descends the "
                     "hand onto the table (ee_press_compliance.py's load case). *** MEASURED "
                     "2026-08-14, job 191032: 'ee' DOES NOT PRESS on this build. The commanded z fell "
                     "117 mm while the achieved z moved 0.2 mm and panda_link8's world z moved 0.1 mm "
                     "-- the arm never left the hover pose, so nothing ever touched the table and "
                     "every deflection it reports is noise. Do not use it until EE control is fixed; "
                     "see the `verify-ee` line. ***")
ap.add_argument("--robot", default="DROID_robolab_v2",
                help="joint-control config for --load tip, an *_ee_control one for --load ee")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze")
ap.add_argument("--tag", default="curl_A", help="output filename prefix")
ap.add_argument("--rungs", default="nf1000a=1000/0.05,nf100=100/0.05",
                help="'name=nf/dr,...' -- physxMimicJoint:<inst>:naturalFrequency / dampingRatio on "
                     "the four INNER mimic joints. Authored: 1000 / 0.05. RESTATE BOTH FIELDS IN "
                     "EVERY RUNG: a '-' leaves whatever the previous rung left, it does not restore "
                     "the authored value. Repeat a rung to get the error bar.")
ap.add_argument("--mimic-joints", default=None, help="override which mimic joints nf/dr apply to")
ap.add_argument("--states", default="open",
                help="gripper states to press in: 'open', 'closed', or 'open,closed'. OPEN is the "
                     "informative one -- see the docstring.")
ap.add_argument("--pin-mass", type=float, default=200.0,
                help="--load tip: mass (kg) the pressing object is given so it is immovable")
ap.add_argument("--tip-gap", type=float, default=0.025,
                help="--load tip: m below the fingertip the object starts")
ap.add_argument("--tip-dz", type=float, default=0.0005, help="--load tip: m of object rise per step")
ap.add_argument("--tip-steps", type=int, default=90, help="--load tip: max ramp steps")
ap.add_argument("--tip-past", type=int, default=40,
                help="--load tip: keep ramping this many steps AFTER first contact")
ap.add_argument("--tip-fingers", default="both",
                help="--load tip: 'both' presses each fingertip in turn (two independent replicates "
                     "per rung), or a finger link name for just one")
ap.add_argument("--hover", type=float, default=0.030, help="m above first contact to start each press")
ap.add_argument("--press-depth", type=float, default=0.040,
                help="m of commanded overshoot past the estimated contact height")
ap.add_argument("--dz", type=float, default=0.002, help="m of commanded descent per step")
ap.add_argument("--press-steps", type=int, default=40)
ap.add_argument("--retract-steps", type=int, default=16)
ap.add_argument("--rest-steps", type=int, default=14, help="unloaded settle at the hover pose")
ap.add_argument("--hold-steps", type=int, default=15, help="one-off orientation round-trip check")
ap.add_argument("--traverse-steps", type=int, default=60)
ap.add_argument("--cam-dist", type=float, default=0.13)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--video", type=int, default=1)
args = ap.parse_args()

OUT = args.out
PFX = args.tag

import torch as th  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

try:
    from realm.environments.contact_utils import _live_impulse_matrix
except Exception:  # pragma: no cover
    _live_impulse_matrix = lambda scene_idx: None  # noqa: E731

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402
from realm.environments.constants import DROID_BASE_HEIGHT  # noqa: E402

GRIP = dict(open=-1.0, closed=+1.0)   # verified by scripts/debug_probes/verify_gripper_mapping.py
L8 = "panda_link8"
OUTER_J = "right_outer_knuckle_joint"
MIMIC_ATTRS = ("naturalFrequency", "dampingRatio", "gearing", "offset")


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


# ---------------------------------------------------------------- build
print(f"[curl] robot={args.robot} task={args.task_cfg} tag={PFX}", flush=True)
set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=False, rendering_mode="rt", robot=args.robot,
)
LOAD = args.load
if LOAD == "ee":
    assert env.ee_control, f"{args.robot} is not an EE-control config; --load ee needs one"
else:
    assert not env.ee_control, f"{args.robot} is an EE-control config; --load tip needs joint control"
obs, _ = env.reset()
obs, _, _, _, _ = env.warmup(obs)
robot = env.robot
ARM_Q = np.asarray(env.reset_qpos[:7], dtype=np.float64)

FL = list(robot.finger_link_names[robot.default_arm])
q_all = _np(robot.get_joint_positions())
joint_names = [None] * len(q_all)
for n, j in robot.joints.items():
    idxs = list(j.dof_indices)
    assert len(idxs) == 1, f"joint {n} has {len(idxs)} DOFs"
    joint_names[idxs[0]] = n
arm_joint_names = list(robot.arm_joint_names[robot.default_arm])
grip_names = [n for n in joint_names if n not in arm_joint_names]
grip_idx = np.array([joint_names.index(n) for n in grip_names])

hdr("CODE / ASSET IDENTITY")
print(f"  robot {type(robot).__name__} '{robot.name}' at {robot.prim_path}")
print(f"  n_dof={len(q_all)} arm={len(arm_joint_names)} gripper={len(grip_names)}")
print(f"  eef link     = {robot.eef_link_names[robot.default_arm]}")
print(f"  finger links = {FL}   (FL[0] is at NEGATIVE AXIS by construction, see docstring)")
print(f"  gripper joints = {grip_names}")


def jget(j, attr):
    try:
        v = getattr(j, attr)
        if isinstance(v, bool) or v is None:
            return v
        return float(_np(v))
    except Exception:
        return None


# ---- parent link of each gripper joint, read from the USD rather than guessed. The pad's rotation
# relative to ITS PARENT is the *_inner_finger_joint pivot's own contribution, as opposed to the whole
# jaw tilting, and it is also what fixes the joint-value polarity without parsing joint axes.
def joint_bodies(jname):
    prim = robot.joints[jname].prim
    out = {}
    for rel in ("physics:body0", "physics:body1"):
        r = prim.GetRelationship(rel)
        tg = [str(t) for t in r.GetTargets()] if r and r.IsValid() else []
        out[rel] = tg[0].rsplit("/", 1)[-1] if tg else None
    return out


PARENT = {}
print("\n  gripper joint topology (body0 -> body1) and mimic APIs:")
for n in grip_names:
    b = joint_bodies(n)
    schemas = [s for s in robot.joints[n].prim.GetAppliedSchemas() if "Mimic" in s]
    PARENT[n] = b["physics:body0"]
    print(f"    {n:<34} {str(b['physics:body0']):<26} -> {str(b['physics:body1']):<26} {schemas}")
PAD_JOINT = {}          # finger link -> the joint whose child it is
for ln in FL:
    cand = [n for n in grip_names if joint_bodies(n)["physics:body1"] == ln]
    PAD_JOINT[ln] = cand[0] if cand else None
    print(f"    pad link {ln} is the child of {PAD_JOINT[ln]}, whose parent is "
          f"{PARENT.get(PAD_JOINT[ln])}")

# ---------------------------------------------------------------- mimic overrides
_reg = lazy.pxr.Usd.SchemaRegistry()
_pd = _reg.FindAppliedAPIPrimDefinition("PhysxMimicJointAPI")
SCHEMA_PROPS = list(_pd.GetPropertyNames()) if _pd is not None else []
NF_IN_SCHEMA = "physxMimicJoint:__INSTANCE_NAME__:naturalFrequency" in SCHEMA_PROPS
print(f"\n  PhysxMimicJointAPI schema = {SCHEMA_PROPS}")
print(f"  naturalFrequency in schema = {NF_IN_SCHEMA} (False is expected on this build; omni.physx "
      f"reads it as a custom attribute by literal token anyway -- measured, see the probes README)")


def mimic_insts(prim):
    """PhysxMimicJointAPI instance names on @prim. NOT the joint's physics:axis -- these joints
    author axis Z and instance rotX, so it is discovered, never guessed."""
    return [s.split(":", 1)[1] for s in prim.GetAppliedSchemas()
            if s.startswith("PhysxMimicJointAPI:")]


MIMIC_JOINTS = [n for n in grip_names if mimic_insts(robot.joints[n].prim)]
INNER_MIMIC = [n for n in MIMIC_JOINTS if n != OUTER_J]
if args.mimic_joints:
    want = [x.strip() for x in args.mimic_joints.split(",") if x.strip()]
    bad = [n for n in want if n not in MIMIC_JOINTS]
    assert not bad, f"--mimic-joints names {bad}; have {MIMIC_JOINTS}"
    INNER_MIMIC = want
print(f"  mimic joints ({len(MIMIC_JOINTS)}): {MIMIC_JOINTS}")
print(f"  softened by nf/dr ({len(INNER_MIMIC)}): {INNER_MIMIC}")


def mimic_state():
    out = {}
    for n in MIMIC_JOINTS:
        prim = robot.joints[n].prim
        for inst in mimic_insts(prim):
            for a in MIMIC_ATTRS:
                at = prim.GetAttribute(f"physxMimicJoint:{inst}:{a}")
                out[f"{n}.{inst}.{a}"] = None if not at.IsValid() else at.Get()
    return out


def mimic_set(names, nf=None, dr=None):
    """Write nf/dr on @names' mimic APIs. MUST be inside og.sim.editing_usd() (simulator.py:1651)."""
    wrote = {}
    with og.sim.editing_usd():
        for n in names:
            prim = robot.joints[n].prim
            insts = mimic_insts(prim)
            assert insts, f"{n} has no PhysxMimicJointAPI"
            for inst in insts:
                for a, v in (("naturalFrequency", nf), ("dampingRatio", dr)):
                    if v is None:
                        continue
                    at = prim.GetAttribute(f"physxMimicJoint:{inst}:{a}")
                    assert at.IsValid(), f"{n} has no physxMimicJoint:{inst}:{a}"
                    at.Set(float(v))
                    wrote[f"{n}.{inst}.{a}"] = float(v)
    return wrote


MIMIC0 = mimic_state()
print("\n  mimic attributes AS AUTHORED:")
for k, v in MIMIC0.items():
    print(f"    {k:<62} = {v}")


def apply_override(nf, dr, label=""):
    wrote = mimic_set(INNER_MIMIC, nf=nf, dr=dr) if (nf is not None or dr is not None) else {}
    live = mimic_state()
    bad = [k for k, v in wrote.items()
           if live.get(k) is None or abs(live[k] - v) > 1e-6 * max(1.0, abs(v))]
    print(f"  [override {label}] wrote {wrote or '(nothing)'}")
    if wrote:
        print(f"  [override {label}] READBACK "
              + ", ".join(f"{k.split('.')[0]}.{k.split('.')[-1]}={live[k]}" for k in sorted(wrote))
              + (f"   *** MISMATCH {bad} ***" if bad else "   (all writes read back)"))
    assert not bad, f"mimic write did not stick: {bad}"
    return live


# ---------------------------------------------------------------- geometry / frames
def T8():
    p, q = robot.links[L8].get_position_orientation()
    return _np(p), Rot.from_quat(_np(q))


def link_pose(ln):
    p, q = robot.links[ln].get_position_orientation()
    return _np(p), Rot.from_quat(_np(q))


def hull_world(ln):
    return _np(robot.links[ln].collision_boundary_points_world)


p8_0, R8_0 = T8()
pl0, _ = link_pose(FL[0])
pr0, _ = link_pose(FL[1])
AXIS = R8_0.inv().apply(pr0 - pl0)
AXIS /= np.linalg.norm(AXIS)
LONG = R8_0.inv().apply((pl0 + pr0) / 2.0 - p8_0)
LONG -= LONG.dot(AXIS) * AXIS
LONG /= np.linalg.norm(LONG)
H = np.cross(AXIS, LONG)
H /= np.linalg.norm(H)
print(f"\n  panda_link8-frame basis:  AXIS (closing, FL0->FL1) = {AXIS}")
print(f"                            LONG (flange->pads)      = {LONG}")
print(f"                            H = AXIS x LONG           = {H}")
INBOARD = {FL[0]: +1.0, FL[1]: -1.0}   # sign along AXIS pointing at the OTHER finger

# Body-fixed tip / base material points, chosen ONCE here and then TRACKED. Re-taking the extreme
# every step would let the finger's own rotation change which point is being measured.
NHULL = {ln: len(hull_world(ln)) for ln in FL}
TIP_IDX, BASE_IDX, SPANS_U = {}, {}, []
for ln in FL:
    Hw = hull_world(ln)
    Hl = R8_0.inv().apply(Hw - p8_0)
    u, v = Hl @ LONG, Hl @ AXIS
    span = u.max() - u.min()
    distal = u >= u.max() - 0.25 * span
    proximal = u <= u.min() + 0.25 * span
    score = v * INBOARD[ln]                      # bigger = further inboard
    TIP_IDX[ln] = int(np.argmax(np.where(distal, score, -1e9)))
    BASE_IDX[ln] = int(np.argmax(np.where(proximal, score, -1e9)))
    SPANS_U.append(span)
    # The hull-vs-origin OFFSET, printed because it is large on this asset and it decides which
    # observables can be trusted: the pad link origins are symmetric about the flange axis (AXIS and
    # LONG both come out exactly axis-aligned), so a hull centroid that is not is a transform bug in
    # collision_boundary_points_world, not geometry. Measured 2026-08-14: ~120 mm along AXIS, the same
    # ~120 mm the squeeze probe saw between the cube's hull centre and its own pose (`hull_off`).
    o = R8_0.inv().apply(link_pose(ln)[0] - p8_0)
    print(f"  {ln}: {NHULL[ln]} hull points, long extent {span * 1000:.1f} mm; "
          f"tip pt #{TIP_IDX[ln]} at u={u[TIP_IDX[ln]] * 1000:.1f} v={v[TIP_IDX[ln]] * 1000:+.1f} mm, "
          f"base pt #{BASE_IDX[ln]} at u={u[BASE_IDX[ln]] * 1000:.1f} "
          f"v={v[BASE_IDX[ln]] * 1000:+.1f} mm")
    print(f"      link ORIGIN at u={o @ LONG * 1000:+.1f} v={o @ AXIS * 1000:+.1f} mm; hull centroid "
          f"at u={Hl.mean(0) @ LONG * 1000:+.1f} v={Hl.mean(0) @ AXIS * 1000:+.1f} mm  -> HULL-ORIGIN "
          f"OFFSET du={(Hl.mean(0) - o) @ LONG * 1000:+.1f} dv={(Hl.mean(0) - o) @ AXIS * 1000:+.1f} mm"
          f"  (a large dv means the hull-based tip/base separations are NOT trustworthy; the "
          f"origin-based and orientation-based observables are unaffected)")
FINGER_HALF = float(np.mean(SPANS_U)) / 2.0
print(f"  finger long extent {FINGER_HALF * 2000:.1f} mm -> tip taken as the pad link origin plus "
      f"{FINGER_HALF * 1000:.1f} mm along LONG")

# Body-fixed inboard face normal per pad: the direction that IS the inboard closing axis right now.
# Re-derived per rest reference below, so a rung whose unloaded pose differs gets its own.
REST = {}          # (rung, state) -> reference dict

# ---------------------------------------------------------------- the pressing object (--load tip)
# Instead of driving the hand down onto the table (which needs IK, and which does not work on this
# build -- see --load), the SURFACE is brought to the fingertip: the task object is pinned heavy,
# gravity disabled, and teleported upward 0.5 mm per step with its pose re-set every step, so contact
# cannot push it away. That is the squeeze probe's trick rotated 90 degrees: it loads the tip along
# the finger's LONG axis, which is the press load case, with the arm provably stationary.
PUSHER = None
scene_idx = robot.scene.idx
if LOAD == "tip":
    PUSHER = env.main_objects[0]
    PUSH_MASS0 = float(PUSHER.root_link.mass)
    PUSH_HOME = _np(PUSHER.get_position_orientation()[0])
    PUSH_HALF = float(_np(PUSHER.aabb_extent).max()) / 2.0
    PUSH_ROWS = RigidContactAPI.get_contact_row_indices(scene_idx, {PUSHER})
    FING_COLS = {ln: RigidContactAPI.get_contact_col_indices(scene_idx, {robot.links[ln]}) for ln in FL}
    print(f"\n  pressing object: {PUSHER.name} mass {PUSH_MASS0:.4f} kg  aabb "
          f"{_np(PUSHER.aabb_extent) * 1000} mm  half-extent {PUSH_HALF * 1000:.1f} mm")


def contact_force(M, ln):
    """|net contact force| (N) between the pressing object and finger link @ln."""
    cols = FING_COLS[ln]
    if M is None or len(PUSH_ROWS) == 0 or len(cols) == 0:
        return float("nan")
    sub = M[PUSH_ROWS][:, cols]
    return float(np.linalg.norm(_np(sub).reshape(-1, 3).sum(axis=0)))


def park_pusher():
    """1.3 m below its home, gravity off, touching nothing. This is what makes the rest reference
    genuinely UNLOADED -- the flaw that voided the first run was a reference taken in contact."""
    PUSHER.disable_gravity()
    PUSHER.set_position_orientation(th.tensor(PUSH_HOME + np.array([0.0, 0.0, -1.3]),
                                             dtype=th.float32))
    PUSHER.keep_still()


def place_under_tip(ln):
    """Park the object squarely under finger @ln's tip, one face normal along the finger's long axis.

    The tip position comes from the pad link ORIGIN (which fix_robolab_link_origins.py put on the pad
    centroid) plus half the finger's length along LONG -- deliberately NOT from
    collision_boundary_points_world, whose points sit ~120 mm off this asset's pad origins (see the
    hull note in the identity block).
    """
    p8, R8 = T8()
    a_w, l_w = R8.apply(AXIS), R8.apply(LONG)
    third = np.cross(a_w, l_w)
    quat = Rot.from_matrix(np.stack([a_w, l_w, third / np.linalg.norm(third)], axis=1)).as_quat()
    tip_w = _np(link_pose(ln)[0]) + l_w * FINGER_HALF
    c = tip_w + l_w * (PUSH_HALF + args.tip_gap)      # LONG points AWAY from the flange = downward
    PUSHER.root_link.mass = float(args.pin_mass)
    PUSHER.disable_gravity()
    PUSHER.set_position_orientation(th.tensor(c, dtype=th.float32),
                                    th.tensor(quat, dtype=th.float32))
    PUSHER.keep_still()
    return c, quat, l_w


def pad_geom():
    """Everything about the two pads, in the panda_link8 frame. No arm motion in any of it."""
    p8, R8 = T8()
    out = dict(p8=p8)
    for k, ln in enumerate(FL):
        p, R = link_pose(ln)
        Hw = hull_world(ln)
        assert len(Hw) == NHULL[ln], f"{ln} hull point count changed: {len(Hw)} != {NHULL[ln]}"
        Hl = R8.inv().apply(Hw - p8)
        out[f"pos{k}"] = R8.inv().apply(p - p8)
        out[f"rot{k}"] = (R8.inv() * R).as_quat()
        out[f"tip{k}"] = Hl[TIP_IDX[ln]]
        out[f"base{k}"] = Hl[BASE_IDX[ln]]
        out[f"low{k}"] = float(Hw[:, 2].min())            # lowest world z of this finger's hull
        out[f"oz{k}"] = float(p[2])                       # pad link ORIGIN world z (hull-free)
        # parent link, for the pivot's own contribution
        par = PARENT.get(PAD_JOINT[ln])
        if par and par in robot.links:
            pp, pR = link_pose(par)
            out[f"prot{k}"] = (R8.inv() * pR).as_quat()
    out["tip_sep"] = float((out["tip1"] - out["tip0"]) @ AXIS)
    out["base_sep"] = float((out["base1"] - out["base0"]) @ AXIS)
    out["pad_sep"] = float((out["pos1"] - out["pos0"]) @ AXIS)
    return out


def measure(tag, cmd6, grip, rung, state):
    g = pad_geom()
    q = _np(robot.get_joint_positions())
    ach = ee_pose_robot_frame()
    ee_w, _ = link_pose(L8)
    r = dict(tag=tag, rung=rung, state=state, q=q, cmd_z=float(cmd6[2]), ach_z=float(ach[2]),
             ee_world_z=float(ee_w[2]), grip=grip, **g)
    # The arm not moving is a claim, so it is measured on every step, not asserted once.
    r["arm_dev"] = float(np.abs(q[:7] - ARM_Q).max())
    if LOAD == "tip":
        M = _live_impulse_matrix(scene_idx)
        for k, ln in enumerate(FL):
            r[f"f{k}"] = contact_force(M, ln)
        r["n_contact"] = len({f for _, f in RigidContactAPI.get_contact_pairs(
            scene_idx=scene_idx, query_set={PUSHER},
            with_set={robot.links[ln] for ln in FL}, current_only=True)})
        r["touching"] = sorted({f.rsplit("/", 1)[-1] for _, f in RigidContactAPI.get_contact_pairs(
            scene_idx=scene_idx, query_set={PUSHER},
            with_set=set(robot.links.values()), current_only=True)})
        r["push_z"] = float(_np(PUSHER.get_position_orientation()[0])[2])
    r["rpy_err"] = float(np.linalg.norm(
        (Rot.from_euler("xyz", ach[3:]) * Rot.from_euler("xyz", cmd6[3:]).inv()).as_rotvec()))
    ref = REST.get((rung, state))
    r.update(signed(r, ref) if ref else {})
    return r


def signed(r, ref):
    """The SIGNED curl numbers for row @r against rest reference @ref. Positive = INWARD, for both
    fingers, by the construction in the docstring."""
    out = {}
    for k, ln in enumerate(FL):
        s_in = INBOARD[ln]
        # (a) the pad's rigid rotation relative to the flange, about H
        dR = Rot.from_quat(r[f"rot{k}"]) * Rot.from_quat(ref[f"rot{k}"]).inv()
        psi = float(dR.as_rotvec() @ H)
        # inward = psi < 0 for FL[0] (inboard +AXIS) and psi > 0 for FL[1]: see the docstring
        out[f"rot_in{k}"] = -s_in * psi
        # (b) the body-fixed inboard face normal's tilt along LONG. Independent of (a) in that it is
        # built from the link's own orientation only, with no reference-frame subtraction.
        n = Rot.from_quat(r[f"rot{k}"]).apply(ref[f"nloc{k}"])
        out[f"norm_in{k}"] = -float(np.arcsin(np.clip(n @ LONG, -1.0, 1.0)))
        out[f"norm_in0{k}"] = -float(np.arcsin(np.clip(
            Rot.from_quat(ref[f"rot{k}"]).apply(ref[f"nloc{k}"]) @ LONG, -1.0, 1.0)))
        # (c) the pad's rotation relative to its OWN PARENT link = the pad pivot's contribution
        if f"prot{k}" in r and f"prot{k}" in ref:
            dRp = ((Rot.from_quat(r[f"prot{k}"]).inv() * Rot.from_quat(r[f"rot{k}"]))
                   * (Rot.from_quat(ref[f"prot{k}"]).inv() * Rot.from_quat(ref[f"rot{k}"])).inv())
            out[f"piv_in{k}"] = -s_in * float(dRp.as_rotvec() @ H)
        # (d) how far that finger's tip moved inboard, in metres
        out[f"tip_in{k}"] = s_in * float((r[f"tip{k}"] - ref[f"tip{k}"]) @ AXIS)
        out[f"base_in{k}"] = s_in * float((r[f"base{k}"] - ref[f"base{k}"]) @ AXIS)
        # (e) the raw joint delta on this pad's own pivot joint
        jn = PAD_JOINT[ln]
        if jn:
            out[f"dq{k}"] = float(r["q"][joint_names.index(jn)] - ref["q"][joint_names.index(jn)])
    out["d_tip_sep"] = r["tip_sep"] - ref["tip_sep"]      # NEGATIVE = tips came together = inward
    out["d_base_sep"] = r["base_sep"] - ref["base_sep"]
    out["d_pad_sep"] = r["pad_sep"] - ref["pad_sep"]
    out["curl_in_deg"] = float(np.degrees(0.5 * (out["rot_in0"] + out["rot_in1"])))
    out["dz_track"] = r["cmd_z"] - r["ach_z"]
    return out


def set_rest(rung, state, rows_rest):
    """Freeze the unloaded reference for this (rung, state) from the last few hover rows."""
    r = rows_rest[-1]
    ref = dict(r)
    ref["q"] = r["q"].copy()
    for k, ln in enumerate(FL):
        # the body-fixed direction that coincides with the inboard closing axis right now
        ref[f"nloc{k}"] = Rot.from_quat(r[f"rot{k}"]).inv().apply(INBOARD[ln] * AXIS)
    REST[(rung, state)] = ref
    return ref


# ---------------------------------------------------------------- EE control plumbing
def ee_pose_robot_frame():
    if LOAD != "ee":
        return np.zeros(6)
    p, q = env.get_ee_pose()
    w = np.concatenate([_np(p), Rot.from_quat(_np(q)).as_euler("xyz")])
    return env._world2robot(np.concatenate([w, [0.0]]))[:6]


frames, rows = [], []
CAM = None
try:
    CAM = env.omnigibson_env.external_sensors["external_sensor0"]
except Exception as e:
    print(f"  [warn] no external_sensor0: {e!r}")


def aim_camera():
    """Perpendicular to the closing plane, centred on the TIP midpoint, carried with the hand.

    The wrist camera looks ALONG the fingers and hides exactly the rotation being measured, so the
    view has to be external and normal to the closing plane. Re-aimed every step: the hand descends
    ~4 cm during a press and a fixed camera would lose the tips.
    """
    if CAM is None:
        return
    p8, R8 = T8()
    g = pad_geom()
    # Aimed from the pad link ORIGINS, pushed half a finger further down LONG to the tips. NOT from
    # the hull tips: those sit ~128 mm off on this asset, which aimed the first run's camera above
    # the fingertips and framed the shot on the knuckles.
    mid = p8 + R8.apply(0.5 * (g["pos0"] + g["pos1"]) + LONG * FINGER_HALF)
    z_c = R8.apply(H)                      # a USD camera looks along -z, so sit along +H
    z_c /= np.linalg.norm(z_c)
    up = -R8.apply(LONG)                   # fingers pointing down the frame
    x_c = np.cross(up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    quat = Rot.from_matrix(np.stack([x_c, y_c, z_c], axis=1)).as_quat()
    CAM.set_position_orientation(th.tensor(mid + z_c * args.cam_dist, dtype=th.float32),
                                 th.tensor(quat, dtype=th.float32), "world")


def do_step(cmd6, grip, tag, rung="", state=""):
    global obs
    if args.video:
        aim_camera()
    action = np.concatenate([ARM_Q if LOAD == "tip" else cmd6, [grip]])
    obs, _, _, _, _ = env.step(action, n_render_iterations=1)
    r = measure(tag, cmd6, grip, rung, state)
    r["step"] = len(rows)
    rows.append(r)
    if args.video:
        ext = obs.get("external", {})
        if "external_sensor0" in ext:
            frames.append(ext["external_sensor0"]["rgb"].cpu().numpy()[..., :3].copy())
        else:
            frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
    return r


# ---------------------------------------------------------------- pick the descent column
hdr("PICKING A CLEAR COLUMN OVER THE TABLE  (ee_press_compliance.py's logic, unchanged)")
_skip = ("floor", "wall", "ceiling", "skirt", "lamp", "window", "door")
objs = []
for o in env.omnigibson_env.scene.objects:
    if o is robot or any(s in o.name.lower() for s in _skip):
        continue
    try:
        lo, hi = o.aabb
    except Exception:
        continue
    objs.append((o.name, _np(lo), _np(hi)))

base_pose0 = ee_pose_robot_frame()
FWD = env._robot2world(np.array([0.45, 0.0, base_pose0[2], *base_pose0[3:], 0.0]))[:2]


def hdist(lo, hi, px, py):
    return float(np.hypot(max(lo[0] - px, 0.0, px - hi[0]), max(lo[1] - py, 0.0, py - hi[1])))


cands = [(hdist(lo, hi, FWD[0], FWD[1]), n, lo, hi)
         for n, lo, hi in objs if 0.55 <= hi[2] <= 1.20
         and (hi[0] - lo[0]) * (hi[1] - lo[1]) > 0.15]
assert cands, "no table-like object found"
_, table_name, t_lo, t_hi = min(cands)
TABLE_TOP = float(t_hi[2])
print(f"  support surface = {table_name}  top z = {TABLE_TOP:.4f}")
others = [(n, lo, hi) for n, lo, hi in objs
          if n != table_name and hi[2] > TABLE_TOP + 0.01 and lo[2] < TABLE_TOP + 0.45]
print(f"  obstacles above it ({len(others)}): {[n for n, _, _ in others]}")


def clearance(px, py):
    best = 1e9
    for n, lo, hi in others:
        best = min(best, float(np.hypot(max(lo[0] - px, 0.0, px - hi[0]),
                                        max(lo[1] - py, 0.0, py - hi[1]))))
    return best


best = None
for rx in np.arange(0.30, 0.63, 0.02):
    for ry in np.arange(-0.35, 0.36, 0.02):
        w = env._robot2world(np.array([rx, ry, base_pose0[2], *base_pose0[3:], 0.0]))
        if not (t_lo[0] + 0.09 <= w[0] <= t_hi[0] - 0.09 and t_lo[1] + 0.09 <= w[1] <= t_hi[1] - 0.09):
            continue
        c = clearance(w[0], w[1])
        if best is None or c > best[0]:
            best = (c, rx, ry, w[:2])
assert best is not None, "no clear column over the table"
CLEAR, RX, RY, WXY = best
print(f"  chosen column: robot-frame x={RX:.3f} y={RY:.3f} world=({WXY[0]:.3f},{WXY[1]:.3f})  "
      f"clearance {CLEAR * 1000:.0f} mm")

hdr("PHASE 0: HOLD -- orientation round-trip check before any descent")
cmd = ee_pose_robot_frame()
for _ in range(args.hold_steps):
    r = do_step(cmd, GRIP["open"], "hold")
print(f"  commanded rpy={cmd[3:]}  achieved rpy={ee_pose_robot_frame()[3:]}")
print(f"  ORIENTATION residual = {r['rpy_err']:.5f} rad ({np.degrees(r['rpy_err']):.2f} deg)  "
      f"<-- a flip or an axis swap would read ~pi")
if r["rpy_err"] > 0.1:
    print("  *** ORIENTATION DOES NOT ROUND-TRIP. The hand is not where it is being commanded, so "
          "'pressed onto a surface' may not be the load case that follows. Judge the press by the "
          "measured penetration and the video, not by the commanded depth. ***")

# Contact height from the MEASURED tip geometry, not from a link origin: with the jaws open the
# distal hull points are the lowest thing on the robot, and they are what touch the table. This is a
# property of the hand's ORIENTATION only (which is held for the whole run), so it is valid before
# the traverse and the descent can be ramped into.
Z_ROBOT_OF_TABLE = TABLE_TOP - (env.robot_pos[2] + (DROID_BASE_HEIGHT if env.use_droid_with_base else 0.0))
TIP_BELOW = rows[-1]["ee_world_z"] - min(rows[-1]["low0"], rows[-1]["low1"])
Z_CONTACT = Z_ROBOT_OF_TABLE + TIP_BELOW
print(f"  table top: world {TABLE_TOP:.4f}, robot frame {Z_ROBOT_OF_TABLE:+.4f}")
print(f"  lowest finger hull point sits {TIP_BELOW * 1000:.1f} mm below panda_link8 -> commanded z "
      f"at first contact {Z_CONTACT:+.4f}; hover {args.hover * 1000:.0f} mm above it, pressing "
      f"{args.press_depth * 1000:.0f} mm past it")

hdr(f"PHASE 0b: TRAVERSE to the clear column (robot frame x={RX:.3f} y={RY:.3f}) and down to hover")
x0, y0, z0 = cmd[0], cmd[1], cmd[2]
for t in range(args.traverse_steps):
    a = (t + 1) / args.traverse_steps
    cmd[0], cmd[1] = x0 + a * (RX - x0), y0 + a * (RY - y0)
    cmd[2] = z0 + a * ((Z_CONTACT + args.hover) - z0)
    do_step(cmd, GRIP["open"], "traverse")
print(f"  arrived; cmd_z={cmd[2]:+.4f} ach_z={rows[-1]['ach_z']:+.4f} "
      f"ee_world_z={rows[-1]['ee_world_z']:.4f}  lowest finger point "
      f"{min(rows[-1]['low0'], rows[-1]['low1']):.4f} (table top {TABLE_TOP:.4f})")

# ---------------------------------------------------------------- one press cycle
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, f"{PFX}_curl.jsonl")
summary = dict(robot=args.robot, tag=PFX, task=args.task_cfg, joint_names=joint_names,
               finger_links=FL, axis=AXIS.tolist(), long=LONG.tolist(), h=H.tolist(),
               inboard={k: v for k, v in INBOARD.items()}, pad_joint=PAD_JOINT, parents=PARENT,
               mimic_authored={k: (None if v is None else float(v)) for k, v in MIMIC0.items()},
               mimic_nf_in_schema=NF_IN_SCHEMA, table=table_name, table_top=TABLE_TOP,
               tip_below_flange_mm=TIP_BELOW * 1e3, z_contact=Z_CONTACT, presses=[])
SPANS = []


def summarise(rung, state, ref, span, ln=None, which=""):
    """The verdict block for one press. Everything here is signed: + = INWARD."""
    press = [r for r in rows[span[0]:span[1]] if r["tag"].endswith("press")]
    desc = [r for r in rows[span[0]:span[1]] if r["tag"].endswith("descend")]
    last = press[-1]
    pen = TABLE_TOP - min(last["low0"], last["low1"])
    # peak of the mean tip rotation over the whole press, signed (max and min, so a sign flip shows)
    curl = np.array([r["curl_in_deg"] for r in press])
    dts = np.array([r["d_tip_sep"] for r in press])
    dps = np.array([r["d_pad_sep"] for r in press])
    hdr(f"PRESS VERDICT  rung={rung}  jaws={state}" + (f"  finger={ln} ({which})" if ln else ""))
    if LOAD == "tip":
        F = max(np.nanmax([r["f0"] for r in press]), np.nanmax([r["f1"] for r in press]))
        print(f"  load: object pinned at {args.pin_mass} kg and ramped up into the tip; peak contact "
              f"force {F:.2f} N, final ({last['f0']:.2f}, {last['f1']:.2f}) N, "
              f"{last['n_contact']} pad contacts, touching {last['touching']}")
        print(f"  the arm did not move: max |q_arm - reset_qpos| over the press = "
              f"{max(r['arm_dev'] for r in press):.2e} rad")
    else:
        print(f"  load: commanded {args.press_depth * 1000:.0f} mm past contact; tracking shortfall "
              f"{last['dz_track'] * 1000:+.2f} mm; lowest finger hull point {pen * 1000:+.2f} mm below "
              f"the table top (positive = it really is pressing in)")
    print(f"  driven joint finger_joint: rest {ref['q'][joint_names.index('finger_joint')]:+.6f} -> "
          f"loaded {last['q'][joint_names.index('finger_joint')]:+.6f} "
          f"(delta {last['q'][joint_names.index('finger_joint')] - ref['q'][joint_names.index('finger_joint')]:+.6f} rad; "
          f"near zero means the leader held and everything below is follower deviation)")
    print(f"\n  SIGNED, + = INWARD (tip toward the closing axis), - = OUTWARD (splay):")
    print(f"    {'finger':<26} {'pad rot vs flange':>18} {'face normal tilt':>18} "
          f"{'pivot vs parent':>17} {'tip moved in':>13} {'base moved in':>14} {'dq (rad)':>11}")
    for k, ln in enumerate(FL):
        print(f"    {ln:<26} {np.degrees(last[f'rot_in{k}']):>+15.3f} deg "
              f"{np.degrees(last[f'norm_in{k}'] - last[f'norm_in0{k}']):>+15.3f} deg "
              f"{np.degrees(last.get(f'piv_in{k}', float('nan'))):>+14.3f} deg "
              f"{last[f'tip_in{k}'] * 1000:>+10.3f} mm {last[f'base_in{k}'] * 1000:>+11.3f} mm "
              f"{last.get(f'dq{k}', float('nan')):>+11.6f}")
    print(f"\n  OBSERVABLE 1 (DISTANCE, hull-free): pad-origin separation change = "
          f"{last['d_pad_sep'] * 1000:+.3f} mm   (NEGATIVE = pads came together = INWARD)")
    print(f"  OBSERVABLE 2 (ROTATION, hull-free): mean pad rotation about H   = "
          f"{last['curl_in_deg']:+.3f} deg   (POSITIVE = INWARD)")
    print(f"  hull-based, treat as advisory: tip-to-tip {last['d_tip_sep'] * 1000:+.3f} mm, "
          f"base-to-base {last['d_base_sep'] * 1000:+.3f} mm  -- see the hull-origin offset in the "
          f"identity block before quoting either")
    print(f"  over the whole press: curl min {curl.min():+.3f} max {curl.max():+.3f} deg; "
          f"d_pad_sep min {dps.min() * 1000:+.3f} max {dps.max() * 1000:+.3f} mm")
    # Magnitude gate FIRST: with no deflection at all both signs are noise, and calling that
    # "INWARD" because a micron landed on the right side would be the same mistake as reading |flex|.
    tiny = abs(last["curl_in_deg"]) < 0.05 and abs(last["d_pad_sep"]) < 50e-6
    agree = (last["d_pad_sep"] < 0) == (last["curl_in_deg"] > 0)
    verdict = ("NO_DEFLECTION" if tiny else
               "INWARD" if last["curl_in_deg"] > 0 and last["d_pad_sep"] < 0 else
               "OUTWARD" if last["curl_in_deg"] < 0 and last["d_pad_sep"] > 0 else "DISAGREE")
    print(f"\n  CURL_VERDICT rung={rung} state={state} finger={which or 'both'} direction={verdict} "
          f"curl_deg={last['curl_in_deg']:+.4f} d_pad_sep_mm={last['d_pad_sep'] * 1000:+.4f} "
          f"d_tip_sep_mm={last['d_tip_sep'] * 1000:+.4f} "
          f"force_N={(max(np.nanmax([r.get('f0', np.nan) for r in press]), np.nanmax([r.get('f1', np.nan) for r in press])) if LOAD == 'tip' else float('nan')):.2f} "
          f"pen_mm={pen * 1000:+.3f} shortfall_mm={last['dz_track'] * 1000:+.3f} "
          f"observables_agree={agree}")
    if not agree:
        print("  *** THE TWO OBSERVABLES DISAGREE. Do not pick one; report both. ***")
    rec = dict(rung=rung, state=state, finger=which or "both", finger_link=ln,
               direction=verdict, observables_agree=bool(agree),
               load=LOAD,
               force_N=(float(max(np.nanmax([r.get("f0", np.nan) for r in press]),
                                  np.nanmax([r.get("f1", np.nan) for r in press])))
                        if LOAD == "tip" else None),
               n_contact_final=last.get("n_contact"), touching_final=last.get("touching"),
               arm_dev_max=float(max(r["arm_dev"] for r in press)),
               curl_in_deg=last["curl_in_deg"],
               d_tip_sep_mm=last["d_tip_sep"] * 1e3, d_base_sep_mm=last["d_base_sep"] * 1e3,
               d_pad_sep_mm=last["d_pad_sep"] * 1e3,
               curl_min_deg=float(curl.min()), curl_max_deg=float(curl.max()),
               d_tip_sep_min_mm=float(dts.min() * 1e3), d_tip_sep_max_mm=float(dts.max() * 1e3),
               penetration_mm=pen * 1e3, shortfall_mm=last["dz_track"] * 1e3,
               n_press=len(press), n_desc=len(desc),
               finger_rot_in_deg={ln: float(np.degrees(last[f"rot_in{k}"])) for k, ln in enumerate(FL)},
               finger_norm_in_deg={ln: float(np.degrees(last[f"norm_in{k}"] - last[f"norm_in0{k}"]))
                                   for k, ln in enumerate(FL)},
               finger_piv_in_deg={ln: float(np.degrees(last.get(f"piv_in{k}", float("nan"))))
                                  for k, ln in enumerate(FL)},
               finger_tip_in_mm={ln: last[f"tip_in{k}"] * 1e3 for k, ln in enumerate(FL)},
               finger_base_in_mm={ln: last[f"base_in{k}"] * 1e3 for k, ln in enumerate(FL)},
               dq={PAD_JOINT[ln]: last.get(f"dq{k}") for k, ln in enumerate(FL)},
               dq_all={joint_names[i]: float(last["q"][i] - ref["q"][i]) for i in grip_idx},
               mimic_live={k: (None if v is None else float(v)) for k, v in mimic_state().items()},
               span=list(span))
    summary["presses"].append(rec)
    with open(JSONL, "a") as f:                       # flushed per press, not at the end
        f.write(json.dumps(rec, default=float) + "\n")
    with open(os.path.join(OUT, f"{PFX}_curl.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return rec


def annotate(im, i):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return im
    r = rows[i]
    img = Image.fromarray(im)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(17, im.shape[0] // 30))
    except TypeError:
        font = ImageFont.load_default()
    txt = (f"{PFX}  rung {r['rung'] or '-'}  jaws {r['state'] or '-'}  {r['tag']}  step {i}\n"
           f"pad-pad {r.get('d_pad_sep', float('nan')) * 1000:+7.2f} mm   "
           f"curl {r.get('curl_in_deg', float('nan')):+6.2f} deg  (+ = INWARD)   "
           f"F {r.get('f0', float('nan')):.1f}/{r.get('f1', float('nan')):.1f} N")
    d.rectangle([0, 0, im.shape[1], int(im.shape[0] * 0.13)], fill=(0, 0, 0))
    d.multiline_text((10, 6), txt, fill=(255, 255, 80), font=font)
    return np.asarray(img)


def write_clip(path, sel, crop):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    ims = []
    for i in sel:
        fr = frames[i]
        if fr.shape[0] < 16:
            continue
        if crop:
            h, w = fr.shape[:2]
            ch, cw = h // 4, w // 4
            fr = fr[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
        ims.append(annotate(np.ascontiguousarray(fr), i))
    if not ims:
        return None
    ImageSequenceClip(ims, fps=args.fps).write_videofile(path, codec="libx264", audio=False,
                                                         logger=None)
    print(f"  wrote {path} ({len(ims)} frames @ {args.fps} fps)")
    return path


WRITTEN = []


def tip_cycle(rung, state, nf, dr, ln, which):
    """UNLOADED reference (object parked 1.3 m away) -> ramp the object up into finger @ln's tip until
    contact, then --tip-past steps further -> park it again and check the deflection comes back."""
    grip = GRIP[state]
    f0 = len(rows)
    lab = f"{rung}_{which}"
    hdr(f"RUNG {rung}  JAWS {state}  PRESSING {ln} ({which})   inner-mimic nf={nf} dr={dr}")
    apply_override(nf, dr, label=rung)
    park_pusher()
    for _ in range(args.rest_steps):
        do_step(cmd, grip, f"{lab}_{state}_rest", rung, state)
    rest_rows = [r for r in rows if r["tag"] == f"{lab}_{state}_rest"]
    ref = set_rest(rung, state, rest_rows)
    for i in range(f0, len(rows)):
        rows[i].update(signed(rows[i], ref))
    # The reference is only a reference if nothing is touching it. A reference taken IN CONTACT is
    # exactly what voided the first run (job 191032): the hover was already on the table, so every
    # delta was measured against a loaded pose.
    if ref["n_contact"] != 0 or ref["touching"]:
        print(f"  *** REFERENCE_NOT_UNLOADED: {ref['n_contact']} pad contacts, touching "
              f"{ref['touching']}. Every number below is a delta from a LOADED pose. ***")
    print(f"  UNLOADED reference (object parked, {ref['n_contact']} contacts): pad_sep "
          f"{ref['pad_sep'] * 1000:.3f} mm  tip_sep {ref['tip_sep'] * 1000:.3f} mm  gripper q "
          f"{ref['q'][grip_idx]}")
    c, quat, l_w = place_under_tip(ln)
    print(f"  object pinned at {args.pin_mass} kg, {args.tip_gap * 1000:.0f} mm below the tip of {ln}")
    first_contact = None
    for t in range(args.tip_steps):
        c = c - l_w * args.tip_dz                      # LONG points away from the flange, so -LONG = up
        PUSHER.set_position_orientation(th.tensor(c, dtype=th.float32),
                                        th.tensor(quat, dtype=th.float32))
        PUSHER.keep_still()
        r = do_step(cmd, grip, f"{lab}_{state}_press", rung, state)
        if first_contact is None and r["n_contact"] > 0:
            first_contact = t
            print(f"    FIRST CONTACT at ramp step {t} ({t * args.tip_dz * 1000:.1f} mm of rise); "
                  f"touching {r['touching']}")
        if t % 10 == 0 or (first_contact is not None and t == first_contact + args.tip_past):
            print(f"    ramp t={t:>3} rise={(t + 1) * args.tip_dz * 1000:5.1f}mm ncon={r['n_contact']} "
                  f"F=({r['f0']:6.2f},{r['f1']:6.2f})N d_pad_sep={r['d_pad_sep'] * 1000:+7.3f}mm "
                  f"curl={r['curl_in_deg']:+7.3f}deg "
                  f"piv=({np.degrees(r.get('piv_in0', float('nan'))):+6.2f},"
                  f"{np.degrees(r.get('piv_in1', float('nan'))):+6.2f})deg "
                  f"armdev={r['arm_dev']:.1e}", flush=True)
        if first_contact is not None and t >= first_contact + args.tip_past:
            break
    if first_contact is None:
        print(f"  *** NEVER TOUCHED: {args.tip_steps * args.tip_dz * 1000:.0f} mm of rise and no "
              f"contact. The tip estimate or the ramp length is wrong; nothing below means anything.")
    rec = summarise(rung, state, ref, (f0, len(rows)), ln=ln, which=which)
    park_pusher()
    PUSHER.root_link.mass = PUSH_MASS0
    for _ in range(args.retract_steps):
        r = do_step(cmd, grip, f"{lab}_{state}_release", rung, state)
    print(f"  after RELEASE: d_pad_sep {r['d_pad_sep'] * 1000:+.3f} mm, curl {r['curl_in_deg']:+.3f} "
          f"deg  (back near zero = elastic, not a snap-through)")
    rec["recovered_pad_sep_mm"] = r["d_pad_sep"] * 1e3
    rec["recovered_curl_deg"] = r["curl_in_deg"]
    rec["first_contact_step"] = first_contact
    finish_cycle(rung, state, which, f0)
    return rec


def press_cycle(rung, state, nf, dr):
    """hover (unloaded reference) -> descend -> press -> retract, for one rung and gripper state."""
    grip = GRIP[state]
    f0 = len(rows)
    hdr(f"RUNG {rung}  JAWS {state}   inner-mimic nf={nf} dr={dr}")
    apply_override(nf, dr, label=rung)
    # rise to the hover height first, with the gripper command for this state already applied
    cmd[2] = Z_CONTACT + args.hover
    for _ in range(args.rest_steps):
        do_step(cmd, grip, f"{rung}_{state}_hover", rung, state)
    ref = set_rest(rung, state, [r for r in rows if r["tag"] == f"{rung}_{state}_hover"])
    # the reference row itself, now that REST exists, so the video has numbers from frame 0
    for i in range(f0, len(rows)):
        rows[i].update(signed(rows[i], ref))
    print(f"  unloaded reference: tip_sep {ref['tip_sep'] * 1000:.3f} mm  "
          f"base_sep {ref['base_sep'] * 1000:.3f} mm  pad_sep {ref['pad_sep'] * 1000:.3f} mm  "
          f"gripper q {ref['q'][grip_idx]}")
    z0 = cmd[2]
    # Per-state contact height: opening the jaws swings the fingers, so the lowest hull point sits at
    # a different depth below the flange than it does shut. Recomputed here so "press_depth past
    # contact" means the same tip-below-surface in both states.
    tb = ref["ee_world_z"] - min(ref["low0"], ref["low1"])
    z_contact = Z_ROBOT_OF_TABLE + tb
    print(f"  lowest finger point is {tb * 1000:.1f} mm below panda_link8 in this state -> contact "
          f"at commanded z {z_contact:+.4f}, pressing to {z_contact - args.press_depth:+.4f}")
    z_target = z_contact - args.press_depth
    n_desc = max(1, int(np.ceil((z0 - z_target) / args.dz)))
    for t in range(n_desc):
        cmd[2] = max(z_target, z0 - args.dz * (t + 1))
        r = do_step(cmd, grip, f"{rung}_{state}_descend", rung, state)
        if t % 6 == 0 or t == n_desc - 1:
            print(f"    desc t={t:>3} cmd_z={cmd[2]:+.4f} ach_z={r['ach_z']:+.4f} "
                  f"low={min(r['low0'], r['low1']):.4f} (table {TABLE_TOP:.4f}) "
                  f"d_tip_sep={r['d_tip_sep'] * 1000:+7.3f}mm curl={r['curl_in_deg']:+6.3f}deg",
                  flush=True)
    for t in range(args.press_steps):
        r = do_step(cmd, grip, f"{rung}_{state}_press", rung, state)
        if t % 10 == 0 or t == args.press_steps - 1:
            print(f"    press t={t:>3} shortfall={r['dz_track'] * 1000:+6.2f}mm "
                  f"pen={(TABLE_TOP - min(r['low0'], r['low1'])) * 1000:+6.2f}mm "
                  f"d_tip_sep={r['d_tip_sep'] * 1000:+7.3f}mm d_base={r['d_base_sep'] * 1000:+7.3f}mm "
                  f"curl={r['curl_in_deg']:+6.3f}deg", flush=True)
    rec = summarise(rung, state, ref, (f0, len(rows)))
    # RETRACT: does the deflection come back? An elastic constraint violation recovers; a linkage
    # that has snapped through does not.
    cmd[2] = Z_CONTACT + args.hover
    for _ in range(args.retract_steps):
        r = do_step(cmd, grip, f"{rung}_{state}_retract", rung, state)
    print(f"  after RETRACT to the hover pose: d_tip_sep {r['d_tip_sep'] * 1000:+.3f} mm, "
          f"curl {r['curl_in_deg']:+.3f} deg  (back near zero = elastic, not a snap-through)")
    rec["recovered_tip_sep_mm"] = r["d_tip_sep"] * 1e3
    rec["recovered_curl_deg"] = r["curl_in_deg"]
    finish_cycle(rung, state, "", f0)
    return rec


def finish_cycle(rung, state, which, f0):
    """Flush this cycle's json, mp4s, peak still and npz. Called per press so an allocation that
    expires mid-sweep still leaves everything measured so far on disk."""
    name = f"{rung}_{which}" if which else rung
    with open(os.path.join(OUT, f"{PFX}_curl.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    SPANS.append((name, state, f0, len(rows)))
    if args.video:
        sel = list(range(f0, len(rows)))
        for suffix, crop in (("closeup", False), ("closeup_ZOOM", True)):
            p = write_clip(os.path.join(OUT, f"{PFX}_{name}_{state}_{suffix}.mp4"), sel, crop)
            if p:
                WRITTEN.append(p)
        # a still at peak load, which is the frame worth putting in front of a human
        try:
            from PIL import Image
            ip = f0 + len(sel) - args.retract_steps - 1
            Image.fromarray(annotate(np.ascontiguousarray(frames[ip]), ip)).save(
                os.path.join(OUT, f"{PFX}_{name}_{state}_peak.png"))
        except Exception as e:
            print(f"  [warn] still failed: {e!r}")
    np.savez_compressed(
        os.path.join(OUT, f"{PFX}_curl.npz"),
        q=np.stack([r["q"] for r in rows]), tag=np.array([r["tag"] for r in rows]),
        rung=np.array([r["rung"] for r in rows]), state=np.array([r["state"] for r in rows]),
        tip_sep=np.array([r["tip_sep"] for r in rows]),
        base_sep=np.array([r["base_sep"] for r in rows]),
        pad_sep=np.array([r["pad_sep"] for r in rows]),
        d_tip_sep=np.array([r.get("d_tip_sep", np.nan) for r in rows]),
        d_base_sep=np.array([r.get("d_base_sep", np.nan) for r in rows]),
        curl_in_deg=np.array([r.get("curl_in_deg", np.nan) for r in rows]),
        rot_in0=np.array([r.get("rot_in0", np.nan) for r in rows]),
        rot_in1=np.array([r.get("rot_in1", np.nan) for r in rows]),
        piv_in0=np.array([r.get("piv_in0", np.nan) for r in rows]),
        piv_in1=np.array([r.get("piv_in1", np.nan) for r in rows]),
        tip_in0=np.array([r.get("tip_in0", np.nan) for r in rows]),
        tip_in1=np.array([r.get("tip_in1", np.nan) for r in rows]),
        low0=np.array([r["low0"] for r in rows]), low1=np.array([r["low1"] for r in rows]),
        cmd_z=np.array([r["cmd_z"] for r in rows]), ach_z=np.array([r["ach_z"] for r in rows]),
        ee_world_z=np.array([r["ee_world_z"] for r in rows]),
        joint_names=np.array(joint_names), table_top=TABLE_TOP,
    )


# ---------------------------------------------------------------- RUN
def parse_rungs(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, vals = part.partition("=")
        f = (vals.split("/") + ["-"] * 2)[:2]
        conv = lambda s: None if s.strip() in ("-", "") else float(s)  # noqa: E731
        out.append((name.strip(), conv(f[0]), conv(f[1])))
    return out


STATES = [s.strip() for s in args.states.split(",") if s.strip()]
assert all(s in GRIP for s in STATES), f"--states must be from {list(GRIP)}"
RUNGS = parse_rungs(args.rungs)
hdr(f"{len(RUNGS)} RUNGS x {len(STATES)} GRIPPER STATES, one process")
for name, nf, dr in RUNGS:
    print(f"  {name:<12} inner nf={nf} dr={dr}")
print(f"  states: {STATES};  '-' means LEAVE WHAT THE PREVIOUS RUNG LEFT, not 'restore authored'")

TIP_FINGERS = (FL if args.tip_fingers == "both" else [args.tip_fingers])
assert all(f in FL for f in TIP_FINGERS), f"--tip-fingers must name one of {FL} or 'both'"
for name, nf, dr in RUNGS:
    for state in STATES:
        if LOAD == "tip":
            # Each fingertip in turn: two independent replicates of the same claim per rung, and the
            # per-finger sign is what the target behaviour is actually about.
            for ln in TIP_FINGERS:
                tip_cycle(name, state, nf, dr, ln, "L" if ln == FL[0] else "R")
        else:
            press_cycle(name, state, nf, dr)

# ---------------------------------------------------------------- the table
hdr("CURL TABLE -- every press, signed. + = INWARD, - = OUTWARD SPLAY")
print(f"  {'rung':<12} {'jaws':<7} {'curl deg':>9} {'d tip-tip':>10} {'d base':>9} {'pen':>7} "
      f"{'short':>7} {'dir':>8} {'agree':>6} {'recov curl':>11}")
for rec in summary["presses"]:
    print(f"  {rec['rung']:<12} {rec['state']:<7} {rec['curl_in_deg']:>+9.3f} "
          f"{rec['d_tip_sep_mm']:>+10.3f} {rec['d_base_sep_mm']:>+9.3f} "
          f"{rec['penetration_mm']:>+7.2f} {rec['shortfall_mm']:>+7.2f} {rec['direction']:>8} "
          f"{str(rec['observables_agree']):>6} {rec.get('recovered_curl_deg', float('nan')):>+11.3f}")
base = {}
for rec in summary["presses"]:
    base.setdefault((rec["rung"].rstrip("ab"), rec["state"]), []).append(rec["curl_in_deg"])
reps = {k: v for k, v in base.items() if len(v) > 1}
if reps:
    noise = max(max(v) - min(v) for v in reps.values())
    print(f"\n  REPEATABILITY from repeated rungs {list(reps)}: "
          + "; ".join(f"{k}: {['%+.3f' % x for x in v]}" for k, v in reps.items())
          + f"  -> noise floor {noise:.3f} deg")
    print(f"  CURL_NOISE_FLOOR_DEG {noise:.4f}")
with open(os.path.join(OUT, f"{PFX}_curl.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\n  wrote {OUT}/{PFX}_curl.json / .jsonl / .npz  ({len(rows)} steps)")
print("MP4S: " + " ".join(WRITTEN))
print("CURL_PROBE_OK")
og.shutdown()
