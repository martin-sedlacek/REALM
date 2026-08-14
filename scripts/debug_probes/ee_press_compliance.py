"""Drive the OPEN end effector straight down onto the table and record it, to answer:
do the robolab 2F-85 fingertips CURL INWARD when they catch on a surface?

The jaw state is REALM_GRIP (default `open`, was hardcoded CLOSED until 2026-08-14). Pressing a shut
jaw braces the two pads against each other along the linkage's stiff axis and geometrically forbids
the inward curl this exists to look for; every press number taken before that flip is about a
different experiment. See the GRIP_CLOSE block for the full account.

No policy and no server -- the action is synthesised here. Runs under EE control
(DroidEndEffectorController, absolute_pose), so it also exercises the dm_robotics IK path.

Three phases, all at the SAME orientation:
  0. HOLD    -- command the pose the arm is already in and check it does not drift. This is the
                rotation sanity check: the commanded rpy is compared against the achieved rpy, so a
                flip or an axis permutation shows up as a large residual BEFORE any descent. The
                orientation is never authored by hand -- it is read off the robot and held -- which
                is what keeps the pads facing down.
  1. DESCEND -- ramp the commanded z down past the table surface.
  2. PRESS   -- keep commanding the (now sub-surface) target so the controller keeps pushing.

Compliance is measured, not just filmed: every step logs the gripper DOFs and the two inner-finger
pad links IN THE panda_link8 FRAME, so arm motion is factored out. Rigid fingers hold that geometry
constant under load; compliant ones do not.

    REALM_ROBOT=DROID_robolab_v2_ee_control python -u /app/tmp/press_video.py
"""
import os

import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as Rot

np.set_printoptions(precision=4, suppress=True, linewidth=200)

ROBOT = os.environ.get("REALM_ROBOT", "DROID_robolab_v2_ee_control")
TASK_CFG = os.environ.get("REALM_TASK_CFG", "REALM_DROID10/put_green_block_into_bowl/default.yaml")
OUT_DIR = os.environ.get("REALM_OUT", "/logs/ee_press")
HOLD_STEPS = int(os.environ.get("REALM_HOLD", "20"))
DESC_STEPS = int(os.environ.get("REALM_DESC", "170"))
PRESS_STEPS = int(os.environ.get("REALM_PRESS", "110"))
DZ = float(os.environ.get("REALM_DZ", "0.004"))   # m per step of commanded descent
FPS = 15

import omnigibson as og

from realm.sim_config import set_sim_config
from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.environments.constants import DROID_BASE_HEIGHT
from realm.inference.utils import wrist_camera_obs_key


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


print(f"[press] robot={ROBOT} task={TASK_CFG}", flush=True)
set_sim_config(robot=ROBOT)

env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=TASK_CFG, perturbations=["Default"],
    multi_view=False, no_rendering=False, rendering_mode="rt", robot=ROBOT,
)
obs, _ = env.reset()
obs, _, _, _, _ = env.warmup(obs)
robot = env.robot

finger_links = robot.finger_link_names[robot.default_arm]
eef_link = robot.eef_link_names[robot.default_arm]
print(f"[press] eef={eef_link} fingers={finger_links}", flush=True)


def world_pose(link_name):
    p, q = robot.links[link_name].get_position_orientation()
    return _np(p), _np(q)


def ee_pose_robot_frame():
    """Current eef pose as a 6-vector in the frame env._world2robot produces (= what we command)."""
    p, q = env.get_ee_pose()
    p, q = _np(p), _np(q)
    w = np.concatenate([p, Rot.from_quat(q).as_euler("xyz")])
    return env._world2robot(np.concatenate([w, [0.0]]))[:6]


def finger_geom():
    """Pad links expressed in the panda_link8 frame -- arm motion removed."""
    ep, eq = world_pose(eef_link)
    R8 = Rot.from_quat(eq)
    out = []
    for ln in finger_links:
        fp, _ = world_pose(ln)
        out.append(R8.inv().apply(fp - ep))
    sep = float(np.linalg.norm(out[0] - out[1]))
    return np.array(out), sep


# --------------------------------------------------------------------------------------------
# TIP vs HEEL: is the pad CURLING INWARD, splaying outward, or just translating?
#
# The pad-origin separation above cannot tell those apart -- it is one number per pad. The 2F-85
# capability being chased is specifically the fingertip rotating INWARD when the tip meets
# resistance, and "the deflection was large" is not evidence of that: an outward splay of the same
# magnitude is a failure. So each pad's own convex collision hull is split along the finger's long
# axis and the INNERMOST point (the one facing the other pad) is taken in the distal half and in the
# proximal half:
#
#   tip_sep  = distance between the two distal  innermost points
#   heel_sep = distance between the two proximal innermost points
#
# both reported signed against their free-hanging values. Then, unambiguously:
#   tip_sep DOWN and heel_sep UP    -> the pads ROTATE about their pivots, tips INWARD  <- wanted
#   tip_sep UP   and heel_sep DOWN  -> the pads rotate the other way, tips splay OUTWARD
#   both DOWN (or both UP) together -> the pads TRANSLATE; the pivots are not what yielded
#
# The hull points are captured once in each pad's own link frame and carried by the live link pose,
# so this measures link motion only and is immune to the hull-centre-vs-pose offset that bit the
# squeeze probe. Everything is in the panda_link8 frame, like the rest of this file.
_HULL_LOCAL = {}
AXIS8 = None      # closing axis (left pad -> right pad), unit, in the panda_link8 frame
LONG8 = None      # finger long axis (flange -> pads), unit, same frame


def capture_reference_geometry():
    """Fix the hulls and the two reference axes once, from the free-hanging closed pose."""
    global AXIS8, LONG8
    for ln in finger_links:
        pts = robot.links[ln].collision_boundary_points_world
        if pts is None or len(pts) == 0:
            print(f"  [warn] {ln} has no collision hull points -- tip/heel geometry unavailable")
            _HULL_LOCAL.clear()
            return False
        p, q = world_pose(ln)
        _HULL_LOCAL[ln] = Rot.from_quat(q).inv().apply(_np(pts) - p)
    fg, _ = finger_geom()
    a = fg[1] - fg[0]
    a = a / np.linalg.norm(a)
    lg = (fg[0] + fg[1]) / 2.0
    lg = lg - a * float(lg @ a)                     # orthogonalise against the closing axis
    AXIS8, LONG8 = a, lg / np.linalg.norm(lg)
    print(f"  hull points: " + ", ".join(f"{ln}={len(_HULL_LOCAL[ln])}" for ln in finger_links))
    print(f"  closing axis (link8) = {AXIS8}\n  finger long axis     = {LONG8}")
    return True


def tip_heel():
    """(tip_sep, heel_sep) in metres from the pads' hulls; (nan, nan) if unavailable."""
    if not _HULL_LOCAL or AXIS8 is None:
        return float("nan"), float("nan")
    ep, eq = world_pose(eef_link)
    R8i = Rot.from_quat(eq).inv()
    inner = {}
    for side, ln in enumerate(finger_links):
        p, q = world_pose(ln)
        P = R8i.apply(Rot.from_quat(q).apply(_HULL_LOCAL[ln]) + p - ep)   # (n,3) in the link8 frame
        along = P @ LONG8
        mid = 0.5 * (along.min() + along.max())     # distal half = further from the flange
        sgn = +1.0 if side == 0 else -1.0           # "inner" is +AXIS8 for left, -AXIS8 for right
        for half, mask in (("tip", along >= mid), ("heel", along < mid)):
            inner[(half, side)] = float(((P[mask] @ AXIS8) * sgn).max()) * sgn
    return (abs(inner[("tip", 1)] - inner[("tip", 0)]),
            abs(inner[("heel", 1)] - inner[("heel", 0)]))


hdr("PICKING A CLEAR COLUMN OVER THE TABLE")
# The first attempt descended straight down from the reset pose and stopped dead 10 cm above the
# table, at a perfectly constant pose -- it had landed on a task object (a glass), not on the table.
# So choose the descent column instead of inheriting it: find the support surface, then the point on
# it that is furthest from every other object AND inside the arm's reach.
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
# The point 45 cm straight in front of the robot, in world coords. The support surface we want is
# the one UNDER the arm -- picking "biggest footprint with a table-height top" instead selected
# countertop_tpuwys_4, the kitchen counter on the far side of the room, and no reachable column
# existed on it.
FWD = env._robot2world(np.array([0.45, 0.0, base_pose0[2], *base_pose0[3:], 0.0]))[:2]


def hdist(lo, hi, px, py):
    return float(np.hypot(max(lo[0] - px, 0.0, px - hi[0]), max(lo[1] - py, 0.0, py - hi[1])))


cands = [(hdist(lo, hi, FWD[0], FWD[1]), n, lo, hi)
         for n, lo, hi in objs if 0.55 <= hi[2] <= 1.20
         and (hi[0] - lo[0]) * (hi[1] - lo[1]) > 0.15]
assert cands, "no table-like object found"
_, table_name, t_lo, t_hi = min(cands)
print(f"  robot forward point (world) = ({FWD[0]:.3f}, {FWD[1]:.3f})")
TABLE_TOP = float(t_hi[2])
print(f"  support surface = {table_name}  top z = {TABLE_TOP:.4f}  "
      f"footprint x[{t_lo[0]:.2f},{t_hi[0]:.2f}] y[{t_lo[1]:.2f},{t_hi[1]:.2f}]")

# Obstacles are only the things occupying the band just above the table top. Without the vertical
# test the CARPET wins -- its floor-level AABB spans the whole room footprint, so every candidate
# column scored 0 mm clearance.
others = [(n, lo, hi) for n, lo, hi in objs
          if n != table_name and hi[2] > TABLE_TOP + 0.01 and lo[2] < TABLE_TOP + 0.45]
print(f"  obstacles above the surface ({len(others)}): {[n for n, _, _ in others]}")


def clearance(px, py):
    best = 1e9
    for n, lo, hi in others:
        dx = max(lo[0] - px, 0.0, px - hi[0])
        dy = max(lo[1] - py, 0.0, py - hi[1])
        best = min(best, float(np.hypot(dx, dy)))
    return best


base_pose = base_pose0
best = None
for rx in np.arange(0.30, 0.63, 0.02):
    for ry in np.arange(-0.35, 0.36, 0.02):
        w = env._robot2world(np.array([rx, ry, base_pose[2], *base_pose[3:], 0.0]))
        if not (t_lo[0] + 0.07 <= w[0] <= t_hi[0] - 0.07 and t_lo[1] + 0.07 <= w[1] <= t_hi[1] - 0.07):
            continue
        c = clearance(w[0], w[1])
        if best is None or c > best[0]:
            best = (c, rx, ry, w[:2])
if best is None:   # fall back to the table centre projected into reach
    cx, cy = (t_lo[0] + t_hi[0]) / 2, (t_lo[1] + t_hi[1]) / 2
    rp = env._world2robot(np.array([cx, cy, base_pose[2], *base_pose[3:], 0.0]))
    best = (0.0, float(np.clip(rp[0], 0.30, 0.62)), float(np.clip(rp[1], -0.35, 0.35)),
            np.array([cx, cy]))
    print("  WARNING: no clear column found; falling back to the table centre")
CLEAR, RX, RY, WXY = best
print(f"  chosen column: robot-frame x={RX:.3f} y={RY:.3f}  world xy=({WXY[0]:.3f},{WXY[1]:.3f})  "
      f"clearance to nearest object = {CLEAR * 1000:.0f} mm")
print(f"  nearest objects: " + ", ".join(
    f"{n}@{clearance_:.3f}m" for n, clearance_ in sorted(
        ((n, float(np.hypot(max(lo[0] - WXY[0], 0, WXY[0] - hi[0]),
                            max(lo[1] - WXY[1], 0, WXY[1] - hi[1])))) for n, lo, hi in others),
        key=lambda kv: kv[1])[:4]))

hdr("PHASE 0: HOLD -- does a commanded pose come back unchanged? (rotation sanity check)")
cmd = ee_pose_robot_frame()
print(f"  commanded (robot frame) xyz={cmd[:3]}  rpy={cmd[3:]}")
# THE JAW STATE THE PRESS HAPPENS AT. droid_gripper_controller: target >= 0 -> the joint's UPPER
# limit = jaws SHUT; -1 = OPEN. This probe used to hardcode +1 and press with the jaws shut, and that
# silently made it the wrong experiment for this question: a shut jaw has the two pads flat against
# each other, so the four-bar is braced pad-to-pad along its STIFF axis and an inward tip curl is
# blocked by the opposing pad -- geometrically, whatever the physics says. The 2F-85 behaviour being
# chased is the OPEN gripper's splayed fingertips catching on a surface and the underactuated linkage
# curling them inward. Default flipped to OPEN on 2026-08-14; `curl_WRIST_kp15_sep.npy` from the
# closed version runs 70.2 -> 32.6 mm (it shut the jaw, then pressed it), so every number taken that
# way -- including "the default already curls inward, -0.117 mm tip" -- is about a different test.
GRIP_NAME = os.environ.get("REALM_GRIP", "open").strip().lower()
assert GRIP_NAME in ("open", "closed"), f"REALM_GRIP must be open|closed, got {GRIP_NAME!r}"
GRIP_CLOSE = -1.0 if GRIP_NAME == "open" else 1.0
print(f"[press] jaw state for the WHOLE run: {GRIP_NAME.upper()} (gripper command {GRIP_CLOSE:+.0f})")
frames, frames_wrist, rows = [], [], []

# THE WRIST VIEW IS THE RIGHT CAMERA FOR *THIS* MOTION, and the standing advice is about a different
# one. "The wrist camera looks along the fingers and hides bending" was written for the SQUEEZE case,
# where the pads move within the plane the wrist camera looks along -- edge-on, invisible. An inward
# TIP CURL is the opposite: the wrist camera looks down the approach axis, so the tips converging is
# a lateral motion in that frame and reads directly as the jaw gap closing, while `external_sensor0`
# sees the same motion nearly edge-on. Both are recorded here; neither replaces the tip/heel numbers,
# and note the camera is `wrist_camera_flipped` -- check the image orientation before reading any
# direction off it. The key is RESOLVED (env.wrist_camera_key, set by assert_wrist_camera) rather
# than guessed, because a miss degrades to some other camera with only a warning.
WRIST_KEY = getattr(env, "wrist_camera_key", None) or wrist_camera_obs_key(robot.name)
_have_wrist = robot.name in obs and WRIST_KEY in obs[robot.name]
print(f"[press] wrist camera key = {WRIST_KEY}  present in obs = {_have_wrist}")
if not _have_wrist:
    # Degrade the way extract_from_obs does -- to SOME camera on the robot, loudly -- rather than
    # losing the whole video to a renamed link after a ten-minute boot.
    _cams = [k for k in obs.get(robot.name, {}) if ":Camera:" in k]
    print(f"  [warn] '{WRIST_KEY}' not in obs; robot camera keys = {_cams}")
    if _cams:
        WRIST_KEY, _have_wrist = _cams[0], True
        print(f"  [warn] FALLING BACK to '{WRIST_KEY}' -- this may not be the wrist view; check the "
              f"first frame before showing the clip to anyone.")


def do_step(cmd6, tag):
    global obs
    action = np.concatenate([cmd6, [GRIP_CLOSE]])
    if AXIS8 is not None:
        aim_camera()             # ride with the hand: it descends ~45 cm during the run
    obs, _, _, _, _ = env.step(action, n_render_iterations=1)
    frames.append(obs["external"]["external_sensor0"]["rgb"].cpu().numpy()[..., :3].copy())
    if _have_wrist:
        frames_wrist.append(obs[robot.name][WRIST_KEY]["rgb"].cpu().numpy()[..., :3].copy())
    ach = ee_pose_robot_frame()
    q = _np(obs[robot.name]["proprio"])
    fg, sep = finger_geom()
    ee_w, _ = world_pose(eef_link)
    # Geodesic angle between commanded and achieved orientation. A flip or an axis swap lands
    # near pi; correct tracking stays in the milliradians.
    rpy_err = float(np.linalg.norm(
        (Rot.from_euler("xyz", ach[3:]) * Rot.from_euler("xyz", cmd6[3:]).inv()).as_rotvec()))
    tip, heel = tip_heel()
    rows.append(dict(tag=tag, cmd_z=cmd6[2], ach_z=ach[2], ee_world_z=float(ee_w[2]),
                     rpy_err=rpy_err, sep=sep, gq=q[7:].copy(), fg=fg.copy(),
                     tip_sep=tip, heel_sep=heel))
    return ach


# Settle at the commanded jaw state BEFORE the reference geometry is taken. The old comment here was
# "close first: the reference axes are defined between the pads and must be taken at the closed pose
# the whole press then happens at". The requirement is real -- the hulls and the two reference axes
# must belong to the pose being pressed -- but it was never a reason to CLOSE the jaw, only a reason
# to capture the reference at whatever pose the press happens at. Keeping a closed-pose reference
# while pressing open would be the same bug wearing the opposite hat.
for t in range(10):
    do_step(cmd, f"settle_{GRIP_NAME}")
hdr(f"REFERENCE GEOMETRY (free-hanging, jaws {GRIP_NAME.upper()} -- the pose the press happens at)")
capture_reference_geometry()

# ---------------------------------------------------------------------------- close-up camera
# external_sensor0, placed PERPENDICULAR TO THE CLOSING PLANE and re-aimed every step so it rides
# with the hand. Looking along H = AXIS x LONG puts both the closing axis and the finger long axis in
# the image plane, so an inward tip motion is fully in-plane (not foreshortened) and the table edge
# shows up as the contact line. The wrist camera is NOT the view for this: rendered out, the gripper
# sits in a corner seen obliquely with the tips barely in frame.
#
# The aim point comes from the pad link ORIGINS, never from collision_boundary_points_world: those
# sit ~116 mm off the origins along the closing axis on this asset, and aiming at them put the first
# attempt's frame on the knuckles instead of the fingertips.
CAM = None
try:
    CAM = env.omnigibson_env.external_sensors["external_sensor0"]
except Exception as e:
    print(f"  [warn] no external_sensor0 to re-aim: {e!r}")
CAM_DIST = float(os.environ.get("REALM_CAM_DIST", "0.22"))
CAM_AHEAD = float(os.environ.get("REALM_CAM_AHEAD", "0.030"))   # m past the pad origins, toward the tips
CAM_MIN_ABOVE = float(os.environ.get("REALM_CAM_MIN_ABOVE", "0.075"))   # m above the table top


def aim_camera():
    if CAM is None or AXIS8 is None:
        return
    ep, eq = world_pose(eef_link)
    R8 = Rot.from_quat(eq)
    a_w, l_w = R8.apply(AXIS8), R8.apply(LONG8)
    fg, _ = finger_geom()
    mid = ep + R8.apply((fg[0] + fg[1]) / 2.0) + l_w * CAM_AHEAD
    n = np.cross(a_w, l_w)                        # normal of the closing plane
    n /= np.linalg.norm(n)
    C = mid + n * CAM_DIST
    # KEEP THE CAMERA ABOVE THE TABLE. Perpendicular-and-level puts it at fingertip height, which IS
    # table height by the end of a press: measured 2026-08-14 on curl_OPEN_kp15, the clip's mean pixel
    # value fell 78 -> 2 as the camera descended into the slab and the entire press phase came out
    # black. Clamping its world z and re-aiming AT the tips costs about asin(rise/CAM_DIST) of tilt,
    # which still leaves the closing axis essentially in the image plane, and keeps the tips lit.
    C[2] = max(float(C[2]), TABLE_TOP + CAM_MIN_ABOVE)
    z_c = C - mid                                 # a USD camera looks along -z, so it sits at +z_c
    z_c /= np.linalg.norm(z_c)
    up = -l_w                                     # fingers pointing down the frame
    x_c = np.cross(up, z_c)
    if np.linalg.norm(x_c) < 1e-6:                # degenerate only looking straight down the finger
        x_c = np.cross(a_w, z_c)                  # axis, which the clamp cannot produce
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    quat = Rot.from_matrix(np.stack([x_c, y_c, z_c], axis=1)).as_quat()
    CAM.set_position_orientation(th.tensor(C, dtype=th.float32),
                                 th.tensor(quat, dtype=th.float32), "world")


aim_camera()
for _ in range(3):
    og.sim.render()
print(f"  external_sensor0 re-aimed perpendicular to the closing plane, {CAM_DIST:.2f} m from the "
      f"pad midpoint + {CAM_AHEAD * 1000:.0f} mm toward the tips (re-aimed every step)")

for t in range(HOLD_STEPS):
    ach = do_step(cmd, "hold")
r = rows[-1]
print(f"  achieved  (robot frame) xyz={ach[:3]}  rpy={ach[3:]}")
print(f"  position residual = {np.linalg.norm(ach[:3] - cmd[:3]):.5f} m")
print(f"  ORIENTATION residual = {r['rpy_err']:.5f} rad "
      f"({np.degrees(r['rpy_err']):.2f} deg)  <-- a flip would show as ~pi here")
print(f"  finger pad separation at rest ({GRIP_NAME}) = {r['sep'] * 1000:.1f} mm")
print(f"  tip separation at rest = {r['tip_sep'] * 1000:.2f} mm, heel separation = "
      f"{r['heel_sep'] * 1000:.2f} mm")
print(f"  gripper DOFs = {r['gq']}")



# Pad offset along the eef z-axis, measured on this robot: how far below panda_link8 the pads sit.
PAD_OFF = float(rows[-1]["fg"][:, 2].mean())
Z_ROBOT_OF_TABLE = TABLE_TOP - (env.robot_pos[2] + (DROID_BASE_HEIGHT if env.use_droid_with_base else 0.0))
Z_CONTACT = Z_ROBOT_OF_TABLE + PAD_OFF          # commanded z at which the pads just touch
Z_TARGET = Z_CONTACT - 0.05                     # keep pushing 5 cm past contact
print(f"  pad offset below panda_link8 = {PAD_OFF:.4f} m")
print(f"  table top in robot frame     = {Z_ROBOT_OF_TABLE:+.4f}  -> contact at commanded z "
      f"{Z_CONTACT:+.4f}, pressing to {Z_TARGET:+.4f}")

hdr(f"PHASE 0b: TRAVERSE to the clear column (robot-frame x={RX:.3f} y={RY:.3f})")
x0, y0 = cmd[0], cmd[1]
TRAV = 60
for t in range(TRAV):
    a = (t + 1) / TRAV
    cmd[0], cmd[1] = x0 + a * (RX - x0), y0 + a * (RY - y0)
    do_step(cmd, "traverse")
r = rows[-1]
print(f"  arrived: cmd xy=({cmd[0]:.3f},{cmd[1]:.3f})  ach z={r['ach_z']:.4f} "
      f"ee_world_z={r['ee_world_z']:.4f} sep={r['sep'] * 1000:.1f}mm")

# PHASE 1, adaptive. The geometric contact estimate above is only a guess -- PAD_OFF is the pad
# ORIGIN offset, the tips hang further down, and with the jaws OPEN they hang further down again.
# Guessing it wrong is what produced a "hover" that was already 30 mm into the table on 2026-08-14
# (job 191032), so the reference was taken in contact. So: descend until CONTACT IS DETECTED from the
# arm's own tracking error -- in free air the controller holds the commanded z to a few mm, and the
# moment the fingertips land the shortfall grows monotonically -- then overtravel a controlled
# distance PAST the achieved contact height and hold. The overtravel is what is reported, and the
# curl should grow with it, which the descent rows record on the way down.
# SIGN, measured rather than assumed: during a free-air descent the controller LAGS, so the achieved
# z sits ABOVE the commanded one -- 5 to 8 mm of it at DZ=4 mm/step in the 2026-08-14 runs. Landing
# makes the arm stop while the command keeps going down, so the same quantity keeps GROWING (it hit
# 64 mm in the closed press). So the signal is `ach_z - cmd_z`, and the threshold has to clear the
# free-air lag, not just zero. Getting this backwards is a detector that can never fire.
SHORT_TH = float(os.environ.get("REALM_SHORT_TH", "0.018"))    # m of lag that means "landed"
OVERTRAVEL = float(os.environ.get("REALM_OVERTRAVEL", "0.030"))  # m past the achieved contact height
hdr(f"PHASE 1: DESCEND at {DZ} m/step until the tips land (tracking lag > {SHORT_TH * 1000:.0f} mm), "
    f"then {OVERTRAVEL * 1000:.0f} mm of OVERTRAVEL, then PHASE 2: PRESS {PRESS_STEPS} steps")
z0 = cmd[2]
Z_FLOOR = Z_CONTACT - 0.12                       # hard stop, in case contact is never detected
ach_contact = None
z_land = None
for t in range(DESC_STEPS):
    cmd[2] = max(Z_FLOOR, z0 - DZ * (t + 1))
    r = do_step(cmd, "descend")
    r = rows[-1]
    short = r["ach_z"] - r["cmd_z"]      # POSITIVE = the arm is holding above the command
    if ach_contact is None and short > SHORT_TH:
        ach_contact, z_land = r["ach_z"], r["cmd_z"]
        print(f"  *** TIPS LANDED at descend step {t}: commanded z {z_land:+.4f}, achieved "
              f"{ach_contact:+.4f}, lag {short * 1000:.1f} mm, "
              f"tip {r['tip_sep'] * 1000:.3f} mm heel {r['heel_sep'] * 1000:.3f} mm", flush=True)
    if t % 20 == 0 or ach_contact is not None:
        print(f"  desc t={t:>3} cmd_z={cmd[2]:+.4f} ach_z={r['ach_z']:+.4f} lag={short * 1000:6.1f}mm "
              f"ee_world_z={r['ee_world_z']:.4f} tip={r['tip_sep'] * 1000:7.3f} "
              f"heel={r['heel_sep'] * 1000:7.3f} sep={r['sep'] * 1000:6.1f}mm", flush=True)
    if ach_contact is not None and cmd[2] <= ach_contact - OVERTRAVEL:
        print(f"  reached {OVERTRAVEL * 1000:.0f} mm of overtravel past the landing height "
              f"(commanded {cmd[2]:+.4f} vs contact {ach_contact:+.4f})")
        break
    if cmd[2] <= Z_FLOOR:
        print(f"  *** hit the descent floor {Z_FLOOR:+.4f} without detecting contact "
              f"(the tracking lag never exceeded {SHORT_TH * 1000:.0f} mm) -- the press may not have "
              f"loaded the tips at all; read the tip/heel numbers with that in mind")
        break

z_press = cmd[2]
for t in range(PRESS_STEPS):
    do_step(cmd, "press")
    if t % 20 == 0:
        r = rows[-1]
        print(f"  press t={t:>3} cmd_z={cmd[2]:+.4f} ach_z={r['ach_z']:+.4f} "
              f"ee_world_z={r['ee_world_z']:.4f} pad_world_z={r['ee_world_z'] - PAD_OFF:.4f} "
              f"sep={r['sep'] * 1000:6.1f}mm gq={r['gq']}", flush=True)

hdr("OVERTRAVEL LADDER -- the curl has to GROW with how hard the tips are pressed, or it is not a "
    "response to the press")
_ref = [r for r in rows if r["tag"] == "hold"][-1]
if ach_contact is not None:
    print(f"  landing height (achieved z) = {ach_contact:+.4f}; each row is one commanded depth past it")
    print(f"  {'overtravel':>10} {'lag':>10} {'tip delta':>10} {'heel delta':>11}   verdict")
    _seen = set()
    for r in rows:
        if r["tag"] not in ("descend", "press"):
            continue
        ot = round((ach_contact - r["cmd_z"]) * 1000.0)
        # Every distinct commanded depth past the landing height. NOT "multiples of 5 mm": the
        # command steps by DZ (4 mm), so a 5 mm grid samples the ladder about once per 20 mm and can
        # miss a 30 mm overtravel almost entirely.
        if ot < 0 or ot in _seen:
            continue
        _seen.add(ot)
        dt = (r["tip_sep"] - _ref["tip_sep"]) * 1000.0
        dh = (r["heel_sep"] - _ref["heel_sep"]) * 1000.0
        print(f"  {ot:>8.0f}mm {(r['ach_z'] - r['cmd_z']) * 1000:>9.1f}mm {dt:>+9.3f}mm "
              f"{dh:>+10.3f}mm   {'tips IN' if dt < 0 and dh > 0 else ('tips OUT' if dt > 0 and dh < 0 else 'translating' if dt * dh > 0 else '-')}")
else:
    print("  contact was never detected, so there is no ladder to print")

hdr("COMPLIANCE SUMMARY (pad links in the panda_link8 frame -- arm motion removed)")
rest = [r for r in rows if r["tag"] == "hold"][-1]
press_rows = [r for r in rows if r["tag"] == "press"]
last = press_rows[-1]
seps = np.array([r["sep"] for r in rows])
print(f"  free-hanging (end of HOLD):  sep={rest['sep'] * 1000:7.3f} mm  gripper qpos={rest['gq']}")
print(f"  under load   (end of PRESS): sep={last['sep'] * 1000:7.3f} mm  gripper qpos={last['gq']}")
print(f"  delta separation           = {(last['sep'] - rest['sep']) * 1000:+7.3f} mm")
print(f"  max |delta| over whole run = {np.abs(seps - rest['sep']).max() * 1000:7.3f} mm")
print(f"  pad L in eef frame: rest={rest['fg'][0]}  press={last['fg'][0]}  "
      f"|d|={np.linalg.norm(last['fg'][0] - rest['fg'][0]) * 1000:.3f} mm")
print(f"  pad R in eef frame: rest={rest['fg'][1]}  press={last['fg'][1]}  "
      f"|d|={np.linalg.norm(last['fg'][1] - rest['fg'][1]) * 1000:.3f} mm")
print(f"  gripper qpos delta         = {last['gq'] - rest['gq']}")

hdr("DIRECTION: DO THE FINGERTIPS CURL INWARD? (tip vs heel, signed)")
d_tip = (last["tip_sep"] - rest["tip_sep"]) * 1000.0
d_heel = (last["heel_sep"] - rest["heel_sep"]) * 1000.0
print(f"  tip  separation  rest {rest['tip_sep'] * 1000:8.3f} -> press {last['tip_sep'] * 1000:8.3f} mm"
      f"   delta {d_tip:+8.3f} mm")
print(f"  heel separation  rest {rest['heel_sep'] * 1000:8.3f} -> press {last['heel_sep'] * 1000:8.3f} mm"
      f"   delta {d_heel:+8.3f} mm")
tips = np.array([r["tip_sep"] for r in rows])
heels = np.array([r["heel_sep"] for r in rows])
if np.isfinite(tips).all():
    i_ex = int(np.nanargmax(np.abs(tips - rest["tip_sep"])))
    print(f"  worst tip excursion over the run = {(tips[i_ex] - rest['tip_sep']) * 1000:+.3f} mm "
          f"at step {i_ex} ({rows[i_ex]['tag']})")
if not np.isfinite(d_tip) or not np.isfinite(d_heel):
    verdict = "UNAVAILABLE (no collision hull points)"
elif abs(d_tip) < 0.02 and abs(d_heel) < 0.02:
    verdict = "NO MEASURABLE PAD MOTION (both under 20 um)"
elif d_tip < 0 and d_heel > 0:
    verdict = "PADS CURL INWARD  <-- the 2F-85 behaviour being chased"
elif d_tip > 0 and d_heel < 0:
    verdict = "PADS SPLAY OUTWARD  <-- wrong direction; a failure however large"
elif d_tip < 0 and d_heel < 0:
    verdict = ("PADS TRANSLATE INWARD (tip and heel both converge) -- the yield is not at the "
               "pad pivots" if abs(d_tip) - abs(d_heel) < 0 else
               "PADS CURL INWARD ON TOP OF AN INWARD TRANSLATION (tip converges more than heel)")
else:
    verdict = "PADS TRANSLATE OUTWARD (tip and heel both diverge)"
print(f"\n  PRESS_DIRECTION: {verdict}")
print(f"  per-pad pivot deviation (rad): the two pad joints are gripper DOFs [2] and [3] in gq; "
      f"full gq delta printed above")
np.save(os.path.join(OUT_DIR if os.path.isdir(OUT_DIR) else "/tmp", f"{ROBOT}_tipheel.npy"),
        np.stack([tips, heels]))
_low = min(r['ee_world_z'] for r in rows)
print(f"  lowest eef world z reached = {_low:.4f} m -> lowest PAD world z = {_low - PAD_OFF:.4f} m")
print(f"  table top world z          = {TABLE_TOP:.4f} m  -> pads went "
      f"{(TABLE_TOP - (_low - PAD_OFF)) * 1000:+.1f} mm BELOW the surface "
      f"(positive = the arm actually pressed in)")
print(f"  commanded z at end of press = {z_press:+.4f} (robot frame); "
      f"achieved {last['ach_z']:+.4f}; tracking shortfall {last['ach_z'] - z_press:+.4f} m")

hdr("WRITING VIDEO")
os.makedirs(OUT_DIR, exist_ok=True)
TAG = os.environ.get("REALM_VIDTAG", ROBOT)     # so two configs can be told apart in one out dir
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def burn(im, i, label):
    """Phase, step and the live tip/heel numbers, burned in -- a clip of a sub-millimetre motion is
    unreadable without them, and it is what stops a viewer reading direction off pixels."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return im
    r = rows[min(i, len(rows) - 1)]
    img = Image.fromarray(np.ascontiguousarray(im))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(15, im.shape[0] // 30))
    except TypeError:
        font = ImageFont.load_default()
    dt = (r["tip_sep"] - rest["tip_sep"]) * 1000.0
    dh = (r["heel_sep"] - rest["heel_sep"]) * 1000.0
    d.rectangle([0, 0, img.size[0], int(im.shape[0] * 0.15)], fill=(0, 0, 0))
    d.multiline_text((8, 4), f"{TAG}  {label}\n{r['tag']}  step {i}\n"
                             f"tip {dt:+.3f} mm   heel {dh:+.3f} mm  (tip DOWN + heel UP = curl in)",
                     fill=(255, 235, 120), font=font)
    return np.asarray(img)


def write(path, seq, label, crop=1.0):
    ims = []
    for i, fr in enumerate(seq):
        if crop > 1.0:
            h, w = fr.shape[:2]
            ch, cw = int(h / (2 * crop)), int(w / (2 * crop))
            fr = fr[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
        ims.append(burn(fr, i, label))
    ImageSequenceClip(ims, fps=FPS).write_videofile(path, codec="libx264", audio=False, logger=None)
    print(f"  wrote {path}  ({len(ims)} frames @ {FPS} fps = {len(ims) / FPS:.1f} s)")


write(os.path.join(OUT_DIR, f"{TAG}_press.mp4"), frames, "external_sensor0")
if frames_wrist:
    # The wrist view, plus a 3x centre crop: the motion is a few tenths of a millimetre and needs
    # magnification before it is legible at all.
    write(os.path.join(OUT_DIR, f"{TAG}_press_wrist.mp4"), frames_wrist, "wrist_camera_flipped")
    write(os.path.join(OUT_DIR, f"{TAG}_press_wrist_ZOOM3.mp4"), frames_wrist,
          "wrist_camera_flipped 3x", crop=3.0)
    np.save(os.path.join(OUT_DIR, f"{TAG}_wrist_lastframe.npy"), frames_wrist[-1])
else:
    print("  [warn] no wrist frames captured")
np.save(os.path.join(OUT_DIR, f"{TAG}_sep.npy"), seps)
print(f"PRESS_FRAMES external={len(frames)} wrist={len(frames_wrist)}")
print("PRESS_VIDEO_OK")
og.shutdown()
