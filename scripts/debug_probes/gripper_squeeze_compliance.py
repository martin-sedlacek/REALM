"""Squeeze an object between the gripper pads and measure whether the fingers comply, to settle:

    are the robolab 2F-85 fingers meaningfully COMPLIANT, or effectively rigid like stock droid.usd?

Why a squeeze and not a press
-----------------------------
`ee_press_compliance.py` drove the CLOSED gripper straight down into the table. That loads the
four-bar along its STIFF axis (the fingers are braced against each other through the closed linkage)
and found ~0.13 mm of deformation on both assets -- a 9x joint-level ratio on an invisible base.
Squeezing an object between the pads loads the linkage along the axis it actually articulates on,
which is the axis any claimed compliance has to live on.

Design (JOINT control -- no IK anywhere; closing a gripper needs none)
---------------------------------------------------------------------
The arm is commanded to hold `env.reset_qpos[:7]` for the entire run and never moves; only the
1-element binary gripper command changes. Rather than driving the hand to the object, the OBJECT is
moved to the hand: the task cube is teleported to the midpoint between the two pads and oriented so
one face normal lies exactly along the closing axis, so the pads meet two opposite faces flat-on.
That removes the arm trajectory, the table, and IK from the measurement, and makes the two assets'
conditions identical by construction (the placement is defined relative to each asset's own pads).

Five phases, all at the same arm pose:
  1. OPEN         -- binary command -1, jaws to the open extreme. Reference state.
  2. FREE CLOSE   -- binary command +1 with NOTHING between the jaws. This is the calibration run:
                     it records the unloaded kinematic relation q_ref -> (every other gripper joint,
                     pad gap) that the loaded run is compared against, plus the fully-shut gap.
  3. SQUEEZE-FREE -- reopen, place the cube (gravity off so it does not fall out of an open hand,
                     mass as authored), close, then RESTORE GRAVITY and see whether the cube is
                     held, crushed, penetrated, or squirts out.
  4. SQUEEZE-PIN  -- reopen, replace the cube, raise its mass to --pin-mass kg (gravity still off)
                     so it is effectively immovable, and close again. With the object unable to
                     recoil, every bit of commanded overtravel has to go into the linkage, the pads
                     or contact penetration -- this is the clean compliance number.

What compliance MEANS here, and why the raw gap change is not it
----------------------------------------------------------------
A binary close is a position drive parked at the joint limit, so under load the driven joint simply
stalls somewhere short of it. Two different things then look the same in a raw pad measurement:
the linkage having travelled less far (kinematics), and the linkage having FLEXED (compliance). They
are separated by comparing against phase 2 at the SAME driven-joint angle:

    flex(pad gap) = gap_loaded - gap_free(q_ref_loaded)
    flex(joint j) =   q_j_loaded -   q_j_free(q_ref_loaded)

i.e. how much further apart the object holds the pads than the unloaded linkage would put them with
the driven joint at that same angle. Zero means a rigid jaw that merely stopped early; positive means
the fingers are being pushed back, which is what "compliant" has to mean physically.

Pad geometry is measured from the two inner-finger links' CONVEX COLLISION HULLS, not from their
knuckles and not only from their origins:
  * `realm/config/robots/DROID_robolab.yaml` records that the four-bar swings the KNUCKLES apart as
    the PADS close, so a knuckle separation reports the exact opposite of the truth -- that mistake
    once inverted the gripper for a whole eval batch;
  * a link ORIGIN separation is only a pad measurement if the origins sit on the pads
    (scripts/fix_robolab_link_origins.py); the hull extremes along the closing axis are the actual
    opposing pad faces on ANY asset, so the same code is fair to both. `gap_hull` is a real jaw gap
    in millimetres and goes NEGATIVE on penetration.
Both are logged, plus the pads in the `panda_link8` frame so arm motion is factored out.

    ./run python -u scripts/debug_probes/gripper_squeeze_compliance.py --robot DROID_robolab_v2
    ./run python -u scripts/debug_probes/gripper_squeeze_compliance.py --robot DROID

Isaac segfaults at teardown on every run; grep for SQUEEZE_PROBE_OK, never the exit code.
"""
import argparse
import json
import os

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=220)

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2",
                help="realm/config/robots/<name>.yaml. JOINT control only -- an *_ee_control config "
                     "would need a 7-vector pose command instead of the 7 joint targets used here.")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze")
ap.add_argument("--open-steps", type=int, default=20)
ap.add_argument("--close-steps", type=int, default=25, help="free (unloaded) close")
ap.add_argument("--load-steps", type=int, default=45, help="close onto the object")
ap.add_argument("--grav-steps", type=int, default=40, help="hold with gravity restored")
ap.add_argument("--cal-steps", type=int, default=40,
                help="slow unloaded sweep used to calibrate the linkage's kinematics; 0 disables")
ap.add_argument("--cal-kp", type=float, default=3e4)
ap.add_argument("--cal-kd", type=float, default=3e2)
ap.add_argument("--pin-mass", type=float, default=200.0,
                help="mass (kg) the cube is given for the immovable-obstacle squeeze")
ap.add_argument("--cam-dist", type=float, default=0.17, help="close-up camera distance from the pads")
ap.add_argument("--fps", type=int, default=15)
args = ap.parse_args()

ROBOT = args.robot
OUT = args.out

import torch as th  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

import omnigibson as og  # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402
from realm.inference.utils import get_robot_obs_profile  # noqa: E402

try:
    from realm.environments.contact_utils import _live_impulse_matrix
except Exception:  # pragma: no cover
    _live_impulse_matrix = lambda scene_idx: None  # noqa: E731

GRIP_OPEN, GRIP_CLOSE = -1.0, +1.0  # verified by scripts/debug_probes/verify_gripper_mapping.py
L8 = "panda_link8"


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


# ---------------------------------------------------------------- build
print(f"[squeeze] robot={ROBOT} task={args.task_cfg}", flush=True)
set_sim_config(robot=ROBOT)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=True, no_rendering=False, rendering_mode="rt", robot=ROBOT,
)
assert not env.ee_control, f"{ROBOT} is an EE-control config; this probe is joint control only"
obs, _ = env.reset()
obs, _, _, _, _ = env.warmup(obs)          # ends on an OPEN command
robot = env.robot
cube = env.main_objects[0]
scene_idx = robot.scene.idx

FL = list(robot.finger_link_names[robot.default_arm])   # [left, right], definition order
ARM_Q = np.asarray(env.reset_qpos[:7], dtype=np.float64)

q_all = _np(robot.get_joint_positions())
# Name the DOFs by asking each joint where its own DOF sits, rather than assuming that iterating
# robot.joints reproduces the articulation's DOF order. Only single-DOF joints exist on these assets;
# anything else would be a new asset and should be noticed rather than mislabelled.
joint_names = [None] * len(q_all)
for n, j in robot.joints.items():
    try:
        idxs = list(j.dof_indices)
    except Exception:
        idxs = []
    assert len(idxs) == 1, f"joint {n} has {len(idxs)} DOFs; this probe assumes one per joint"
    joint_names[idxs[0]] = n
assert all(n is not None for n in joint_names), f"unmapped DOFs: {joint_names}"
arm_joint_names = list(robot.arm_joint_names[robot.default_arm])
grip_names = [n for n in joint_names if n not in arm_joint_names]
grip_idx = np.array([joint_names.index(n) for n in grip_names])

# ---------------------------------------------------------------- identity block
hdr("CODE / ASSET IDENTITY -- read this before any number below")
print(f"  robot class      = {type(robot).__name__}   name = {robot.name}")
print(f"  prim path        = {robot.prim_path}")
print(f"  n_dof            = {len(q_all)}   arm dofs = {len(arm_joint_names)}   "
      f"gripper dofs = {len(grip_names)}")
print(f"  eef link         = {robot.eef_link_names[robot.default_arm]}")
print(f"  finger links     = {FL}")
print(f"  obs profile      = {get_robot_obs_profile(robot.name)}")
print(f"  arm hold qpos    = {ARM_Q}")
def jget(j, attr):
    try:
        v = getattr(j, attr)
        if isinstance(v, bool) or v is None:
            return v
        return float(_np(v))
    except Exception:
        return None


print("\n  ALL JOINTS -- drive gains read back from the LIVE articulation view, not from the config:")
for i, n in enumerate(joint_names):
    j = robot.joints[n]
    tag = "ARM" if n in arm_joint_names else "GRIP"
    print(f"   [{i:>2}] {tag:4} {n:<36} {str(j.joint_type):<22} "
          f"lim=({jget(j, 'lower_limit')}, {jget(j, 'upper_limit')})  stiff={jget(j, 'stiffness')} "
          f"damp={jget(j, 'damping')} maxeff={jget(j, 'max_effort')} "
          f"maxvel={jget(j, 'max_velocity')} fric={jget(j, 'friction')} "
          f"driven={jget(j, 'driven')} mimic={jget(j, 'is_mimic_joint')} "
          f"ctrl_type={getattr(j, 'control_type', None)}")

# The USD-authored drive block and the PhysX mimic coupling, per gripper joint. This is the
# difference the RoboLab stack leans on: it keeps the asset's authored gains, whereas OmniGibson
# overwrites them with controller_base.m.DEFAULT_ISAAC_KP/KD (1e7/1e5) for any POSITION controller
# that does not name isaac_kp/isaac_kd -- and joint_prim.set_control_type() forces kp=kd=0 on every
# mimic DOF, so the followers have no drive at all and are held purely by the mimic constraint.
# naturalFrequency == 0 (or unauthored) means that constraint is solved as a HARD equality, which is
# the mechanism that would keep the followers from flexing no matter what the leader's gains are.
print("\n  GRIPPER JOINT USD ATTRIBUTES (authored drive block + PhysX mimic coupling):")
for i in grip_idx:
    n = joint_names[i]
    j = robot.joints[n]
    print(f"   [{i:>2}] {n}")
    try:
        prim = j.prim
        schemas = [s for s in prim.GetAppliedSchemas()
                   if any(k in s for k in ("Drive", "Mimic", "Physx", "Limit"))]
        print(f"        applied schemas: {schemas}")
        got = False
        for a in sorted(prim.GetAttributes(), key=lambda x: x.GetName()):
            nm = a.GetName()
            if not (nm.startswith("drive:") or "imicJoint" in nm or nm.startswith("physxJoint:")
                    or nm.startswith("physics:")):
                continue
            try:
                v = a.Get()
            except Exception:
                v = "<unreadable>"
            if v is None and not a.IsAuthored():
                continue
            print(f"        {nm} = {v}")
            got = True
        for r in prim.GetRelationships():
            if "imicJoint" in r.GetName() or "body" in r.GetName().lower():
                print(f"        REL {r.GetName()} -> {[str(t) for t in r.GetTargets()]}")
                got = True
        if not got:
            print("        (no authored drive/mimic attributes)")
    except Exception as e:
        print(f"        [warn] USD read failed: {e!r}")

try:
    from omnigibson.controllers.controller_view import ControllerView
    gk, gci = robot._controllers["gripper_0"]
    gc = ControllerView._controller_groups[gk]
    from omnigibson.controllers import ControlType
    lim = gc._control_limits[ControlType.get_type(gc._motor_type)]
    dof = _np(gc.dof_idx).astype(int)
    print(f"\n  gripper controller: {type(gc).__name__} mode={gc._mode} inverted={gc._inverted} "
          f"motor={gc._motor_type} control_dim={gc.control_dim}")
    print(f"    isaac_kp     = {gc.isaac_kp}   isaac_kd = {gc.isaac_kd}   "
          f"(None -> controller_base defaults 1e7 / 1e5)")
    print(f"    dof_idx      = {dof} -> {[joint_names[i] for i in dof]}")
    print(f"    _open_qpos   = {gc._open_qpos}   _closed_qpos = {gc._closed_qpos}")
    print(f"    ctrl limits  = lo {_np(lim[0])[dof]}  hi {_np(lim[1])[dof]}")
    # What the binary close actually commands: OG's "should_open" branch is target>=0 and sends the
    # UPPER limit, which on both these assets is the physically SHUT pose (see DROID_robolab.yaml).
    CLOSE_TARGET = _np(lim[1])[dof] if gc._open_qpos is None else _np(gc._open_qpos)
    OPEN_TARGET = _np(lim[0])[dof] if gc._closed_qpos is None else _np(gc._closed_qpos)
    CTRL_DOF = dof
    print(f"    => binary CLOSE (+1) drives those dofs to {CLOSE_TARGET}")
    print(f"       binary OPEN  (-1) drives those dofs to {OPEN_TARGET}")
except Exception as e:
    print(f"  [warn] could not resolve the gripper controller: {e!r}")
    CLOSE_TARGET = OPEN_TARGET = None
    CTRL_DOF = grip_idx

print("\n  finger link geometry:")
p8_0, q8_0 = robot.links[L8].get_position_orientation()
for ln in FL:
    lk = robot.links[ln]
    pts = lk.collision_boundary_points_world
    d = float(np.linalg.norm(_np(lk.get_position_orientation()[0]) - _np(p8_0)))
    print(f"   {ln:<28} origin->flange = {d * 1000:7.1f} mm   "
          f"collision meshes = {len(lk.collision_meshes)}  hull points = "
          f"{0 if pts is None else len(pts)}")
    if d < 0.05:
        print("      WARNING: this origin looks like it is at the MOUNT, not the pad. Origin-based "
              "separations below are linkage, not jaw -- trust gap_hull instead.")

cube_hull0 = cube.root_link.collision_boundary_points_world
print(f"\n  object: {cube.name} ({type(cube).__name__})  mass = {cube.root_link.mass:.4f} kg  "
      f"aabb extent = {_np(cube.aabb_extent) * 1000} mm  hull points = "
      f"{0 if cube_hull0 is None else len(cube_hull0)}")


# ---------------------------------------------------------------- measurement
def T8():
    p, q = robot.links[L8].get_position_orientation()
    return _np(p), Rot.from_quat(_np(q))


def _origin(ln):
    return _np(robot.links[ln].get_position_orientation()[0])


# The closing axis is defined ONCE, in the panda_link8 frame, from the open pose, then carried with
# the hand. Recomputing it from the live origins each step would let the linkage's own motion rotate
# the axis the deflection is measured along.
_p8, _R8 = T8()
_pl, _pr = _origin(FL[0]), _origin(FL[1])
AXIS_LOCAL = _R8.inv().apply(_pr - _pl)
AXIS_LOCAL /= np.linalg.norm(AXIS_LOCAL)
LONG_LOCAL = _R8.inv().apply((_pl + _pr) / 2.0 - _p8)
LONG_LOCAL /= np.linalg.norm(LONG_LOCAL)
print(f"\n  closing axis in the panda_link8 frame = {AXIS_LOCAL}")
print(f"  finger long axis in the same frame    = {LONG_LOCAL}")

CUBE_ROWS = RigidContactAPI.get_contact_row_indices(scene_idx, {cube})
FING_COLS = {ln: RigidContactAPI.get_contact_col_indices(scene_idx, {robot.links[ln]}) for ln in FL}
ROBOT_LINKS = {ln: lk for ln, lk in robot.links.items()}
CUBE_HALF = float(_np(cube.aabb_extent).max()) / 2.0   # 15 mm for the 30 mm task cube


def contact_force(M, ln):
    """|net contact force| (N) between the cube and finger link @ln, from the live contact view."""
    cols = FING_COLS[ln]
    if M is None or len(CUBE_ROWS) == 0 or len(cols) == 0:
        return float("nan")
    sub = M[CUBE_ROWS][:, cols]                     # (nr, nc, 3)
    return float(np.linalg.norm(_np(sub).reshape(-1, 3).sum(axis=0)))


def measure(tag, cmd):
    M = _live_impulse_matrix(scene_idx)             # one fetch per step, shared by both fingers
    p8, R8 = T8()
    a = R8.apply(AXIS_LOCAL)                        # world closing axis, left -> right
    pl, pr = _origin(FL[0]), _origin(FL[1])
    hl = _np(robot.links[FL[0]].collision_boundary_points_world)
    hr = _np(robot.links[FL[1]].collision_boundary_points_world)
    # Extremes of each finger's convex collision hull along the closing axis. NOTE: the hull of an
    # L-shaped finger fills in its concavity, so these sit a fixed distance INSIDE the real pad
    # faces -- measured constant to 0.01 mm across the whole travel on robolab v2. Both this and the
    # origin separation are therefore reported RELATIVE to their own value at full unloaded closure,
    # where the pads touch and the physical gap is zero by definition. That self-calibration is what
    # makes a jaw gap comparable between two assets whose link origins sit in different places.
    l_in = float((hl @ a).max())
    r_in = float((hr @ a).min())
    q = _np(robot.get_joint_positions())
    cp = _np(cube.get_position_orientation()[0])
    cp_root = _np(cube.root_link.get_position_orientation()[0])
    ch = _np(cube.root_link.collision_boundary_points_world)
    proj = ch @ a
    # Cube faces from its POSE and half-extent, not from its hull: place_cube() puts one face normal
    # exactly on the closing axis, so this is exact -- and on 2026-08-14 the cube's
    # collision_boundary_points_world was measured ~120 mm away from its own pose after a teleport
    # (hull_off below), while the physics clearly used the pose (the jaws stalled at exactly the cube
    # width). Diagnose with hull_off; do not build a measurement on the hull.
    c_a = float(cp @ a)
    pads_mid = (pl + pr) / 2.0
    return dict(
        tag=tag, cmd=cmd,
        q=q,
        sep_origin=float(np.linalg.norm(pr - pl)),           # link-origin separation
        gap_hull=r_in - l_in,                                # hull gap (constant offset, see above)
        l_in=l_in, r_in=r_in, pl_a=float(pl @ a), pr_a=float(pr @ a), cube_a=c_a,
        pad_l8=np.stack([R8.inv().apply(pl - p8), R8.inv().apply(pr - p8)]),
        arm_dev=float(np.abs(q[:7] - ARM_Q).max()),
        p8=p8,
        cube_pos=cp,
        cube_off=float(np.linalg.norm(cp - pads_mid)),        # drift from the pad midpoint
        cube_off_a=float((cp - pads_mid) @ a),                # ... resolved along the closing axis
        hull_off=float(0.5 * (proj.max() + proj.min()) - c_a),  # hull centre vs pose (diagnostic)
        root_off=float(np.linalg.norm(cp_root - cp)),           # root link vs entity (diagnostic)
        cube_w=float(proj.max() - proj.min()),                # hull extent along the closing axis
        f_l=contact_force(M, FL[0]), f_r=contact_force(M, FL[1]),
        n_contact=len({f for _, f in RigidContactAPI.get_contact_pairs(
            scene_idx=scene_idx, query_set={cube},
            with_set={robot.links[ln] for ln in FL}, current_only=True)}),
        # EVERY robot link, not only the two pads: on the first run the pad count read 1 while the
        # jaws were visibly held apart at the object's width, so "which link is actually taking the
        # load" has to be observable rather than assumed.
        touching=sorted({f.rsplit("/", 1)[-1] for _, f in RigidContactAPI.get_contact_pairs(
            scene_idx=scene_idx, query_set={cube},
            with_set=set(ROBOT_LINKS.values()), current_only=True)}),
    )


frames, frames_wide, rows = [], [], []


def snap(o):
    ext = o.get("external", {})
    if "external_sensor0" in ext:
        frames.append(ext["external_sensor0"]["rgb"].cpu().numpy()[..., :3].copy())
    if "external_sensor1" in ext:
        frames_wide.append(ext["external_sensor1"]["rgb"].cpu().numpy()[..., :3].copy())


ZERO = dict(sep=None, hull=None)   # jaw-gap zeros, filled in from the free close (see measure())


def jaw(r, which="sep"):
    """Physical jaw gap (m): the raw measure minus its own value at full unloaded closure."""
    z = ZERO[which]
    v = r["sep_origin"] if which == "sep" else r["gap_hull"]
    return v if z is None else v - z


def do(tag, cmd, n, verbose_every=0):
    global obs
    for t in range(n):
        obs, _, _, _, _ = env.step(np.concatenate([ARM_Q, [cmd]]), n_render_iterations=1)
        r = measure(tag, cmd)
        r["step"] = len(rows)
        rows.append(r)
        snap(obs)
        if verbose_every and (t % verbose_every == 0 or t == n - 1):
            print(f"  {tag:<12} t={t:>3} jaw={jaw(r) * 1000:8.2f}mm "
                  f"(raw sep {r['sep_origin'] * 1000:7.2f} hull {r['gap_hull'] * 1000:7.2f}) "
                  f"ncon={r['n_contact']} F=({r['f_l']:7.2f},{r['f_r']:7.2f})N "
                  f"armdev={r['arm_dev']:.2e} gq={r['q'][grip_idx]}"
                  + (f" touch={r['touching']}" if r["touching"] else ""), flush=True)
    return rows[-1]


def phase(tag):
    return [r for r in rows if r["tag"] == tag]


# ---------------------------------------------------------------- close-up camera
hdr("AIMING THE CLOSE-UP CAMERA AT THE PADS")
# The wrist camera looks ALONG the fingers and hides any bending, so the view has to be an external
# one, placed perpendicular to the closing plane. Defined as an offset in the panda_link8 frame, so
# both assets get an identically framed shot (same arm, same held qpos, same flange).
try:
    p8, R8 = T8()
    a_w = R8.apply(AXIS_LOCAL)
    f_w = R8.apply(LONG_LOCAL)
    M_pads = (_origin(FL[0]) + _origin(FL[1])) / 2.0
    view = np.cross(a_w, f_w)
    view /= np.linalg.norm(view)
    C = M_pads + view * args.cam_dist
    z_c = C - M_pads
    z_c /= np.linalg.norm(z_c)                      # a USD camera looks along -z
    up = -f_w                                       # pads at the bottom of the frame
    x_c = np.cross(up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    quat = Rot.from_matrix(np.stack([x_c, y_c, z_c], axis=1)).as_quat()
    cam = env.omnigibson_env.external_sensors["external_sensor0"]
    cam.set_position_orientation(th.tensor(C, dtype=th.float32),
                                 th.tensor(quat, dtype=th.float32), "world")
    print(f"  pad midpoint = {M_pads}\n  camera at    = {C}  (dist {args.cam_dist} m)")
    print(f"  quat         = {quat}")
    for _ in range(3):
        og.sim.render()
except Exception as e:
    print(f"  [warn] could not reposition external_sensor0, keeping the scene view: {e!r}")

# ---------------------------------------------------------------- phase 1
hdr("PHASE 1: OPEN -- the reference state")
r_open = do("open", GRIP_OPEN, args.open_steps, verbose_every=10)

# ---------------------------------------------------------------- phase 2
# Park the cube out of the way so it cannot interact; it comes back for phases 3 and 4. Gravity goes
# off first and stays off for the parking (below the floor plane, where nothing can be touched) as
# well as for the squeezes themselves -- an OPEN hand cannot hold an unsupported object up, and
# letting it free-fall while the jaws close would end the experiment before it starts.
cube_home = _np(cube.get_position_orientation()[0])
cube.disable_gravity()
cube.set_position_orientation(th.tensor(cube_home + np.array([0.0, 0.0, -1.3]), dtype=th.float32))
cube.keep_still()

hdr("PHASE 1b: SLOW UNLOADED SWEEP -- the linkage's KINEMATICS, densely sampled")
# The binary drive is a position target with isaac_kp=1e7, which slews the whole 0.785 rad in ONE
# 15 Hz control step: the free close below yields only ~3 distinct driven-joint angles, and linear
# interpolation between them misses the four-bar's curvature by 2-3 mm -- twenty times the deflection
# being measured. So the calibration curve is taken with the leader's drive gains temporarily
# softened, which slows the sweep without changing the KINEMATIC relation it records (the sweep is
# unloaded and quasi-static; only the geometry is being read).
#
# The poke goes straight onto the joint rather than into the controller config because it must be
# reversible inside one process. That is only safe because this probe never calls og.sim.stop()/
# play(): simulator.py re-applies update_controller_mode() on every play, which would silently
# restore isaac_kp/isaac_kd. Any PERMANENT gain change belongs in the gripper_0 controller config.
CAL_ROWS = []
if args.cal_steps > 0:
    # EVERY controlled DOF, not just the first: the stock asset drives four gripper joints, so
    # softening one of them leaves the close as fast (and the sweep as sparse) as before.
    leads = [robot.joints[joint_names[int(i)]] for i in CTRL_DOF]
    gains0 = [(float(_np(j.stiffness)), float(_np(j.damping))) for j in leads]
    try:
        for j in leads:
            j.stiffness = args.cal_kp
            j.damping = args.cal_kd
        print(f"  driven-joint gains {[f'{k:.1e}/{d:.1e}' for k, d in gains0]} -> "
              f"{[f'{float(_np(j.stiffness)):.1e}/{float(_np(j.damping)):.1e}' for j in leads]}")
        do("cal_close", GRIP_CLOSE, args.cal_steps, verbose_every=8)
        do("cal_open", GRIP_OPEN, args.cal_steps, verbose_every=8)
        CAL_ROWS = phase("cal_close") + phase("cal_open")
    finally:
        for j, (k, d) in zip(leads, gains0):
            j.stiffness = k
            j.damping = d
        print(f"  gains restored to "
              f"{[f'{float(_np(j.stiffness)):.1e}/{float(_np(j.damping)):.1e}' for j in leads]} "
              f"(must equal {[f'{k:.1e}/{d:.1e}' for k, d in gains0]})")
    do("reopen_cal", GRIP_OPEN, 10, verbose_every=9)
    qs = np.array([r["q"][int(CTRL_DOF[0])] for r in CAL_ROWS])
    print(f"  swept {len(CAL_ROWS)} samples, driven joint {qs.min():+.4f} .. {qs.max():+.4f}, "
          f"{len(np.unique(np.round(qs, 4)))} distinct values")

hdr("PHASE 2: FREE CLOSE -- nothing between the jaws, at the real drive gains")
r_free = do("free_close", GRIP_CLOSE, args.close_steps, verbose_every=6)
# Full unloaded closure = pads touching = zero physical gap. Everything reported as a "jaw gap" from
# here on is relative to this, which is what makes the two assets comparable.
ZERO["sep"] = r_free["sep_origin"]
ZERO["hull"] = r_free["gap_hull"]
print(f"  jaw-gap zeros taken at full closure: sep_origin={ZERO['sep'] * 1000:.3f} mm, "
      f"gap_hull={ZERO['hull'] * 1000:.3f} mm -> both now read 0.000 mm when shut")
print(f"  jaws OPEN therefore measured {jaw(r_open) * 1000:.2f} mm (sep) / "
      f"{jaw(r_open, 'hull') * 1000:.2f} mm (hull) -- these two agreeing is the cross-check")

# ---------------------------------------------------------------- object placement
def place_cube():
    """Cube centred between the pads, one face normal exactly along the closing axis."""
    p8, R8 = T8()
    a_w = R8.apply(AXIS_LOCAL)
    f_w = R8.apply(LONG_LOCAL)
    third = np.cross(a_w, f_w)
    third /= np.linalg.norm(third)
    quat = Rot.from_matrix(np.stack([a_w, f_w, third], axis=1)).as_quat()
    M_pads = (_origin(FL[0]) + _origin(FL[1])) / 2.0
    cube.set_position_orientation(th.tensor(M_pads, dtype=th.float32),
                                  th.tensor(quat, dtype=th.float32))
    cube.keep_still()
    return M_pads


for label, mass, grav_after in (("A", None, True), ("B", args.pin_mass, False)):
    hdr(f"PHASE 3{label}: REOPEN, then SQUEEZE "
        f"({'mass as authored, free to move' if mass is None else f'mass {mass} kg = immovable'})")
    do(f"reopen_{label}", GRIP_OPEN, args.open_steps, verbose_every=15)
    if mass is not None:
        cube.root_link.mass = float(mass)
    cube.disable_gravity()                     # an open hand cannot hold it up; see the docstring
    M_pads = place_cube()
    cube.keep_still()
    print(f"  cube placed at the pad midpoint {M_pads}, mass = {cube.root_link.mass:.4f} kg, "
          f"gravity DISABLED")
    do(f"settle_{label}", GRIP_OPEN, 12, verbose_every=6)
    r_pre = rows[-1]
    print(f"  pre-squeeze: jaw={jaw(r_pre) * 1000:.2f} mm, object {2000 * CUBE_HALF:.2f} mm wide, "
          f"so {(jaw(r_pre) - 2 * CUBE_HALF) * 500:.2f} mm of approach per pad; "
          f"drift from the midpoint={r_pre['cube_off'] * 1000:.2f} mm "
          f"({r_pre['cube_off_a'] * 1000:+.2f} along the closing axis); "
          f"hull-vs-pose diagnostic {r_pre['hull_off'] * 1000:+.1f} mm")

    hdr(f"PHASE 4{label}: SQUEEZE -- binary close onto the object")
    do(f"squeeze_{label}", GRIP_CLOSE, args.load_steps, verbose_every=5)

    if grav_after:
        hdr(f"PHASE 5{label}: RESTORE GRAVITY while still commanding CLOSE -- is it HELD?")
        cube.enable_gravity()
        try:
            cube.wake()
        except Exception:
            pass
        p_before = _np(cube.get_position_orientation()[0])
        do(f"gravity_{label}", GRIP_CLOSE, args.grav_steps, verbose_every=10)
        p_after = _np(cube.get_position_orientation()[0])
        drop = float(p_before[2] - p_after[2])
        print(f"  cube fell {drop * 1000:+.2f} mm in {args.grav_steps} steps "
              f"({args.grav_steps / 15.0:.1f} s); total displacement "
              f"{np.linalg.norm(p_after - p_before) * 1000:.2f} mm")
        print(f"  VERDICT: {'HELD' if abs(drop) < 0.01 else 'DROPPED / SLIPPED'}")
        cube.disable_gravity()

# ---------------------------------------------------------------- analysis
hdr("ANALYSIS")
REF = int(CTRL_DOF[0])          # the driven joint the linkage is parameterised by
print(f"  reference (driven) joint = [{REF}] {joint_names[REF]}")

free = phase("free_close")
# The unloaded reference curve: the dense slow sweep when there is one, else the 3-point free close.
cal = CAL_ROWS if len(CAL_ROWS) > len(free) else free
print(f"  unloaded reference curve = {'slow sweep' if cal is CAL_ROWS else 'free close'}, "
      f"{len(cal)} samples")
q_free = np.stack([r["q"] for r in cal])
ref_free = q_free[:, REF]
gap_free = np.array([jaw(r, "hull") for r in cal])
sep_free = np.array([jaw(r) for r in cal])
order = np.argsort(ref_free)


def free_at(vals, x):
    """The unloaded value of @vals at driven-joint angle @x, from the calibration sweep."""
    return np.interp(x, ref_free[order], np.asarray(vals)[order])


# Mimic gearing, fitted on the unloaded sweep: q_follower = G_j * q_leader + O_j. On this asset the
# PhysX mimic joints make that relation exact to well under a milliradian when unloaded, so the
# residual under load is a deflection measurement that needs no interpolation at all -- it is the
# strongest of the compliance numbers here. Fitted rather than read from the USD so that it also
# works on the stock asset, whose four gripper joints are independently driven and have no mimic API.
GEAR = {}
for i in grip_idx:
    A = np.stack([ref_free, np.ones_like(ref_free)], axis=1)
    sol, *_ = np.linalg.lstsq(A, q_free[:, i], rcond=None)
    resid = q_free[:, i] - (A @ sol)
    GEAR[int(i)] = (float(sol[0]), float(sol[1]), float(np.abs(resid).max()))
print("  unloaded follower relation q_j = G*q_ref + O (residual = how rigid the coupling is "
      "with NO load):")
for i in grip_idx:
    G, O, res = GEAR[int(i)]
    print(f"    {joint_names[i]:<36} G={G:+.5f} O={O:+.6f}  max residual {res:.6f}")


print(f"\n  UNLOADED EXTREMES (phase 1 open -> phase 2 shut)")
print(f"    open : jaw={jaw(r_open) * 1000:8.2f} mm (hull {jaw(r_open, 'hull') * 1000:.2f})  "
      f"raw sep_origin={r_open['sep_origin'] * 1000:7.2f} mm  q_ref={r_open['q'][REF]:+.5f}")
print(f"    shut : jaw={jaw(free[-1]) * 1000:8.2f} mm (hull {jaw(free[-1], 'hull') * 1000:.2f})  "
      f"raw sep_origin={free[-1]['sep_origin'] * 1000:7.2f} mm  q_ref={free[-1]['q'][REF]:+.5f}")
print(f"    gripper joints open -> shut:")
for i in grip_idx:
    print(f"      {joint_names[i]:<36} {r_open['q'][i]:+.6f} -> {free[-1]['q'][i]:+.6f}  "
          f"(travel {free[-1]['q'][i] - r_open['q'][i]:+.6f})")
if CLOSE_TARGET is not None:
    print(f"    commanded close target = {CLOSE_TARGET}; unloaded residual "
          f"{CLOSE_TARGET - free[-1]['q'][CTRL_DOF]}")

summary = dict(robot=ROBOT, task=args.task_cfg, joint_names=joint_names,
               ref_joint=joint_names[REF],
               isaac_kp=None if CLOSE_TARGET is None else (
                   None if gc.isaac_kp is None else _np(gc.isaac_kp).tolist()),
               isaac_kd=None if CLOSE_TARGET is None else (
                   None if gc.isaac_kd is None else _np(gc.isaac_kd).tolist()),
               grip_joint_params={joint_names[i]: dict(
                   stiffness=jget(robot.joints[joint_names[i]], "stiffness"),
                   damping=jget(robot.joints[joint_names[i]], "damping"),
                   max_effort=jget(robot.joints[joint_names[i]], "max_effort"),
                   is_mimic=jget(robot.joints[joint_names[i]], "is_mimic_joint"),
                   driven=jget(robot.joints[joint_names[i]], "driven")) for i in grip_idx},
               gearing={joint_names[i]: GEAR[int(i)] for i in grip_idx},
               open=dict(jaw_mm=jaw(r_open) * 1e3, jaw_hull_mm=jaw(r_open, "hull") * 1e3,
                         sep_mm=r_open["sep_origin"] * 1e3, q=r_open["q"].tolist()),
               free_shut=dict(jaw_mm=jaw(free[-1]) * 1e3, sep_mm=free[-1]["sep_origin"] * 1e3,
                              q=free[-1]["q"].tolist()),
               close_target=None if CLOSE_TARGET is None else CLOSE_TARGET.tolist(),
               ctrl_dof=[int(i) for i in CTRL_DOF], squeezes={})

for label in ("A", "B"):
    sq = phase(f"squeeze_{label}")
    if not sq:
        continue
    pre = phase(f"settle_{label}")[-1]
    hdr(f"SQUEEZE {label}: {'free object' if label == 'A' else 'immovable object'}")
    touch = [i for i, r in enumerate(sq) if r["n_contact"] > 0]
    both = [i for i, r in enumerate(sq) if r["n_contact"] == 2]
    if not touch:
        print("  NO CONTACT AT ALL -- the jaws never reached the object. Nothing to report.")
        continue
    c0 = sq[touch[0]]
    cb = sq[both[0]] if both else None
    last = sq[-1]
    OBJ_W = 2.0 * CUBE_HALF
    print(f"  first contact at squeeze step {touch[0]} (of {len(sq)}), both pads at "
          f"{both[0] if both else None}; links touching the object at the end: {last['touching']}")
    print(f"    at first contact : jaw={jaw(c0) * 1000:8.3f} mm  q_ref={c0['q'][REF]:+.6f}")
    if cb is not None:
        print(f"    at both-pad     : jaw={jaw(cb) * 1000:8.3f} mm  q_ref={cb['q'][REF]:+.6f}")
    print(f"    at the end      : jaw={jaw(last) * 1000:8.3f} mm (hull {jaw(last, 'hull') * 1000:.3f}) "
          f"q_ref={last['q'][REF]:+.6f}  F=({last['f_l']:.2f}, {last['f_r']:.2f}) N")
    print(f"\n  VALIDATION of the jaw-gap zero: the object is {OBJ_W * 1000:.2f} mm wide, so a "
          f"correct measure must read ~that at first contact.")
    print(f"    jaw at first contact {jaw(c0) * 1000:.3f} mm  ->  error "
          f"{(jaw(c0) - OBJ_W) * 1000:+.3f} mm")
    print(f"    jaw at the end       {jaw(last) * 1000:.3f} mm  ->  the pads closed "
          f"{(OBJ_W - jaw(last)) * 1000:+.3f} mm PAST the object's own width")
    print(f"    (this last number IS the pad-level squeeze: deflection plus contact penetration)")
    print(f"    cube pose drift from the pad midpoint: {last['cube_off'] * 1000:.3f} mm total, "
          f"{last['cube_off_a'] * 1000:+.3f} mm along the closing axis")
    print(f"    diagnostics: hull centre vs pose offset {last['hull_off'] * 1000:+.1f} mm, "
          f"root-link vs entity pose {last['root_off'] * 1000:.3f} mm, hull extent along axis "
          f"{last['cube_w'] * 1000:.2f} mm")

    ref_l = last["q"][REF]
    print(f"\n  HOW FAR THE COMMANDED CLOSE STILL WANTS TO GO (the drive is parked at its target "
          f"the whole time)")
    if CLOSE_TARGET is not None:
        for k, i in enumerate(CTRL_DOF):
            span = free[-1]["q"][i] - r_open["q"][i]
            res = CLOSE_TARGET[k] - last["q"][i]
            print(f"    {joint_names[i]:<36} stalled at {last['q'][i]:+.6f}, target "
                  f"{CLOSE_TARGET[k]:+.6f}  -> unresolved {res:+.6f} "
                  f"({100.0 * res / span if span else float('nan'):+.1f}% of its full travel)")
    print(f"    in jaw terms: the pads sit at {jaw(last) * 1000:.2f} mm while the command "
          f"asks for 0.00 mm -> {jaw(last) * 1000:.2f} mm of jaw closure blocked by the object")

    print(f"\n  COMPLIANCE = deviation from the UNLOADED linkage at the SAME driven angle "
          f"(q_ref={ref_l:+.6f})")
    gap_pred = free_at(gap_free, ref_l)
    sep_pred = free_at(sep_free, ref_l)
    print(f"    jaw gap (hull) : loaded {jaw(last, 'hull') * 1000:8.3f} mm vs unloaded-at-same-q "
          f"{gap_pred * 1000:8.3f} mm  -> FLEX {(jaw(last, 'hull') - gap_pred) * 1000:+8.3f} mm")
    print(f"    jaw gap (orig) : loaded {jaw(last) * 1000:8.3f} mm vs "
          f"{sep_pred * 1000:8.3f} mm  -> FLEX {(jaw(last) - sep_pred) * 1000:+8.3f} mm")
    devs, devs_gear = {}, {}
    print(f"    per gripper joint (rad or m), two independent estimators:")
    print(f"      {'joint':<36} {'loaded':>10} {'interp':>10} {'FLEX':>10} {'G*q+O':>10} {'FLEX':>10}")
    for i in grip_idx:
        pred = free_at(q_free[:, i], ref_l)
        d = last["q"][i] - pred
        G, O, _ = GEAR[int(i)]
        pg = G * ref_l + O
        dg = last["q"][i] - pg
        devs[joint_names[i]] = float(d)
        devs_gear[joint_names[i]] = float(dg)
        print(f"      {joint_names[i]:<36} {last['q'][i]:+10.6f} {pred:+10.6f} {d:+10.6f} "
              f"{pg:+10.6f} {dg:+10.6f}")
    # Worst flex over the whole squeeze, not only at the last step.
    flex_traj = np.array([jaw(r, "hull") - free_at(gap_free, r["q"][REF]) for r in sq])
    joint_flex_traj = {joint_names[i]: float(np.abs(
        np.array([r["q"][i] - free_at(q_free[:, i], r["q"][REF]) for r in sq])).max())
        for i in grip_idx}
    joint_flex_gear = {joint_names[i]: float(np.abs(
        np.array([r["q"][i] - (GEAR[int(i)][0] * r["q"][REF] + GEAR[int(i)][1]) for r in sq])).max())
        for i in grip_idx}
    worst_j = max(joint_flex_traj, key=joint_flex_traj.get)
    worst_g = max(joint_flex_gear, key=joint_flex_gear.get)
    print(f"\n    max |joint flex| by the GEARING estimator = {joint_flex_gear[worst_g]:.6f} "
          f"({worst_g})")
    print(f"\n    max |pad-gap flex| over the squeeze  = {np.abs(flex_traj).max() * 1000:.3f} mm")
    print(f"    max |joint flex| over the squeeze    = {joint_flex_traj[worst_j]:.6f} "
          f"({worst_j})")
    print(f"    max contact force                    = "
          f"{max(np.nanmax([r['f_l'] for r in sq]), np.nanmax([r['f_r'] for r in sq])):.2f} N")
    print(f"    cube drift from the pad midpoint     = "
          f"{last['cube_off'] * 1000:.2f} mm (pre-squeeze {pre['cube_off'] * 1000:.2f} mm)")
    print(f"    max arm joint deviation from hold    = "
          f"{max(r['arm_dev'] for r in sq):.2e} rad (the arm is NOT what moved)")
    print(f"    jaw travel from pre-squeeze to end   = {(jaw(pre) - jaw(last)) * 1000:.3f} mm")
    summary["squeezes"][label] = dict(
        first_contact_step=int(touch[0]), both_pads_step=None if not both else int(both[0]),
        touching_final=last["touching"],
        obj_width_mm=OBJ_W * 1e3,
        jaw_at_contact_mm=jaw(c0) * 1e3, jaw_final_mm=jaw(last) * 1e3,
        jaw_hull_final_mm=jaw(last, "hull") * 1e3,
        past_object_width_mm=float((OBJ_W - jaw(last)) * 1e3),
        zero_validation_mm=float((jaw(c0) - OBJ_W) * 1e3),
        q_ref_final=float(ref_l), cube_w_hull_mm=last["cube_w"] * 1e3,
        hull_off_mm=last["hull_off"] * 1e3, root_off_mm=last["root_off"] * 1e3,
        force_l_N=last["f_l"], force_r_N=last["f_r"],
        max_force_N=float(max(np.nanmax([r["f_l"] for r in sq]), np.nanmax([r["f_r"] for r in sq]))),
        gap_flex_mm=float((jaw(last, "hull") - gap_pred) * 1e3),
        sep_flex_mm=float((jaw(last) - sep_pred) * 1e3),
        max_gap_flex_mm=float(np.abs(flex_traj).max() * 1e3),
        joint_flex=devs, max_joint_flex=joint_flex_traj,
        joint_flex_gear=devs_gear, max_joint_flex_gear=joint_flex_gear,
        cube_drift_mm=last["cube_off"] * 1e3, cube_drift_axis_mm=last["cube_off_a"] * 1e3,
        blocked_jaw_mm=float(jaw(last) * 1e3),
        max_arm_dev=float(max(r["arm_dev"] for r in sq)),
    )

# ---------------------------------------------------------------- outputs
hdr("WRITING VIDEO AND RAW DATA")
os.makedirs(OUT, exist_ok=True)
tags = [r["tag"] for r in rows]
gaps = np.array([jaw(r) for r in rows])       # calibrated jaw gap, used for the video overlay too

np.savez_compressed(
    os.path.join(OUT, f"{ROBOT}_squeeze.npz"),
    q=np.stack([r["q"] for r in rows]), tag=np.array(tags),
    # Raw measures AND their calibrated forms, named so they cannot be confused: `jaw_*` are
    # zeroed at full unloaded closure, `*_raw` are not. (Runs before 2026-08-14 16:00 wrote the
    # calibrated origin measure under the key `gap_hull`, which read as if it were the raw hull.)
    gap_hull_raw=np.array([r["gap_hull"] for r in rows]),
    sep_origin=np.array([r["sep_origin"] for r in rows]),
    cmd=np.array([r["cmd"] for r in rows]),
    pad_l8=np.stack([r["pad_l8"] for r in rows]),
    cube_pos=np.stack([r["cube_pos"] for r in rows]),
    cube_w=np.array([r["cube_w"] for r in rows]),
    jaw_sep=gaps, jaw_hull=np.array([jaw(r, "hull") for r in rows]),
    cube_off=np.array([r["cube_off"] for r in rows]),
    cube_off_a=np.array([r["cube_off_a"] for r in rows]),
    hull_off=np.array([r["hull_off"] for r in rows]),
    f_l=np.array([r["f_l"] for r in rows]), f_r=np.array([r["f_r"] for r in rows]),
    n_contact=np.array([r["n_contact"] for r in rows]),
    arm_dev=np.array([r["arm_dev"] for r in rows]),
    joint_names=np.array(joint_names),
)
with open(os.path.join(OUT, f"{ROBOT}_squeeze.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"  wrote {OUT}/{ROBOT}_squeeze.npz and .json ({len(rows)} steps)")


def annotate(im, i):
    """Burn the phase, step and live jaw gap into the frame so the video is self-documenting."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return im
    img = Image.fromarray(im)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(18, im.shape[0] // 26))
    except TypeError:
        font = ImageFont.load_default()
    txt = (f"{ROBOT}   {tags[i]}   step {i}\n"
           f"jaw gap {gaps[i] * 1000:7.2f} mm   contacts {rows[i]['n_contact']}   "
           f"F {rows[i]['f_l']:.1f}/{rows[i]['f_r']:.1f} N")
    d.rectangle([0, 0, im.shape[1], int(im.shape[0] * 0.13)], fill=(0, 0, 0))
    d.multiline_text((10, 6), txt, fill=(255, 255, 80), font=font)
    return np.asarray(img)


from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # noqa: E402

written = []
for name, seq, crop in (("closeup", frames, False), ("closeup_ZOOM", frames, True),
                        ("wide", frames_wide, False)):
    if not seq:
        continue
    out = os.path.join(OUT, f"{ROBOT}_squeeze_{name}.mp4")
    ims = []
    for i, fr in enumerate(seq):
        if crop:
            h, w = fr.shape[:2]
            # The camera is aimed AT the pad midpoint, so it projects to the image centre and a
            # centre crop is exactly the zoom.
            ch, cw = h // 4, w // 4
            fr = fr[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
        ims.append(annotate(np.ascontiguousarray(fr), min(i, len(rows) - 1)))
    ImageSequenceClip(ims, fps=args.fps).write_videofile(out, codec="libx264", audio=False,
                                                          logger=None)
    print(f"  wrote {out}  ({len(ims)} frames @ {args.fps} fps = {len(ims) / args.fps:.1f} s)")
    written.append(out)

print("\nMP4S: " + " ".join(written))
print("SQUEEZE_PROBE_OK")
og.shutdown()
