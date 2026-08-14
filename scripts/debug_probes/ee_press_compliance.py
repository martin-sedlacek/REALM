"""Drive the CLOSED end effector straight down into the table and record it, to answer:
are the robolab 2F-85 fingers actually compliant, or effectively rigid?

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
GRIP_CLOSE = 1.0   # droid_gripper_controller: target >= 0 -> joint UPPER limit = jaws SHUT
frames, rows = [], []


def do_step(cmd6, tag):
    global obs
    action = np.concatenate([cmd6, [GRIP_CLOSE]])
    obs, _, _, _, _ = env.step(action, n_render_iterations=1)
    frames.append(obs["external"]["external_sensor0"]["rgb"].cpu().numpy()[..., :3].copy())
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


# env.warmup() ends on an OPEN command, so close first: the reference axes are defined between the
# pads and must be taken at the closed pose the whole press then happens at.
for t in range(8):
    do_step(cmd, "shut")
hdr("REFERENCE GEOMETRY (free-hanging, jaws closed)")
capture_reference_geometry()

for t in range(HOLD_STEPS):
    ach = do_step(cmd, "hold")
r = rows[-1]
print(f"  achieved  (robot frame) xyz={ach[:3]}  rpy={ach[3:]}")
print(f"  position residual = {np.linalg.norm(ach[:3] - cmd[:3]):.5f} m")
print(f"  ORIENTATION residual = {r['rpy_err']:.5f} rad "
      f"({np.degrees(r['rpy_err']):.2f} deg)  <-- a flip would show as ~pi here")
print(f"  finger pad separation at rest (closed) = {r['sep'] * 1000:.1f} mm")
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

hdr(f"PHASE 1: DESCEND to {Z_TARGET:+.4f} at {DZ} m/step, then PHASE 2: PRESS {PRESS_STEPS} steps")
z0 = cmd[2]
n_desc = max(1, int(np.ceil((z0 - Z_TARGET) / DZ)))
for t in range(n_desc):
    cmd[2] = max(Z_TARGET, z0 - DZ * (t + 1))
    do_step(cmd, "descend")
    if t % 20 == 0:
        r = rows[-1]
        print(f"  desc t={t:>3} cmd_z={cmd[2]:+.4f} ach_z={r['ach_z']:+.4f} "
              f"ee_world_z={r['ee_world_z']:.4f} pad_world_z={r['ee_world_z'] - PAD_OFF:.4f} "
              f"sep={r['sep'] * 1000:6.1f}mm gq={r['gq']}", flush=True)

z_press = cmd[2]
for t in range(PRESS_STEPS):
    do_step(cmd, "press")
    if t % 20 == 0:
        r = rows[-1]
        print(f"  press t={t:>3} cmd_z={cmd[2]:+.4f} ach_z={r['ach_z']:+.4f} "
              f"ee_world_z={r['ee_world_z']:.4f} pad_world_z={r['ee_world_z'] - PAD_OFF:.4f} "
              f"sep={r['sep'] * 1000:6.1f}mm gq={r['gq']}", flush=True)

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
out = os.path.join(OUT_DIR, f"{ROBOT}_press.mp4")
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
ImageSequenceClip([f for f in frames], fps=FPS).write_videofile(out, codec="libx264", audio=False)
print(f"  wrote {out}  ({len(frames)} frames @ {FPS} fps = {len(frames) / FPS:.1f} s)")
np.save(os.path.join(OUT_DIR, f"{ROBOT}_sep.npy"), seps)
print("PRESS_VIDEO_OK")
og.shutdown()
