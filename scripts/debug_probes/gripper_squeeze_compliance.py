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
# ---- mimic-constraint / effort overrides. See "MIMIC OVERRIDES" below for what these do and why
# they are probe-local rather than config or asset edits.
ap.add_argument("--tag", default="",
                help="prefix for the output filenames, so several configurations share one --out")
ap.add_argument("--mimic-nf", type=float, default=None,
                help="physxMimicJoint:<inst>:naturalFrequency written on the four INNER mimic joints "
                     "(authored 1000). LOWER = softer; 0 means RIGID, not soft -- do not sweep to 0.")
ap.add_argument("--mimic-dr", type=float, default=None,
                help="... dampingRatio on the same four (authored 0.05)")
ap.add_argument("--outer-nf", type=float, default=None,
                help="naturalFrequency on right_outer_knuckle_joint (authored 1e6 = effectively hard)")
ap.add_argument("--outer-dr", type=float, default=None, help="... dampingRatio on that one (authored 0)")
ap.add_argument("--max-effort", type=float, default=None,
                help="max_effort (N m) on the driven finger_joint. Authored 16.5, and the position "
                     "drive saturates there in every gain rung, so it plausibly matters more than kp.")
ap.add_argument("--mimic-joints", default=None,
                help="comma-separated joint names that --mimic-nf/--mimic-dr (and every rung's nf/dr) "
                     "apply to. Default = the four INNER mimic joints. Set it to "
                     "'left_inner_finger_joint,right_inner_finger_joint' to soften ONLY the two pad "
                     "pivots and leave the knuckle couplings stiff -- that localises the yield at the "
                     "pads instead of letting the whole linkage go slack.")
ap.add_argument("--pad-cc-stiffness", type=float, default=None,
                help="physxMaterial:compliantContactStiffness on the two PAD links' physics materials. "
                     "PhysX 5's compliant contact: a finite value makes the contact a spring instead "
                     "of a hard constraint, so the object visibly sinks into the pad. A third avenue "
                     "if the mimic constraint turns out not to be tunable. 0 = hard (the default).")
ap.add_argument("--pad-cc-damping", type=float, default=None,
                help="physxMaterial:compliantContactDamping on the same materials")
ap.add_argument("--variant-usd", default=None,
                help="load a VARIANT robolab USD instead of the shipped one (see "
                     "scripts/debug_probes/make_mimic_variant.py). Use this when a runtime mimic write "
                     "turns out to be parse-time only. Implemented as a probe-local monkeypatch of "
                     "Robot.usd_path, NOT as a new robot `model`: the model lookup goes through "
                     "data/datasets/omnigibson-robot-assets/models/*, which is a symlink tree shared "
                     "between worktrees.")
ap.add_argument("--drive-kp", type=float, default=None,
                help="finger_joint drive stiffness, in the LIVE ARTICULATION VIEW's per-radian "
                     "convention (OmniGibson forces 1e7 there; the USD authors 100 per DEGREE, which "
                     "is 5729.578 per radian -- pass 5729.578 for 'the value RoboLab runs')")
ap.add_argument("--drive-kd", type=float, default=None,
                help="finger_joint drive damping, same convention (OmniGibson 1e5; the USD's 0.0002 "
                     "per degree is 0.011459 per radian)")
ap.add_argument("--solver-pos-iter", type=int, default=None,
                help="physxArticulation:solverPositionIterationCount on the ROBOT. The mimic "
                     "constraint is solved by the articulation solver, so fewer iterations leave a "
                     "bigger constraint residual = apparent compliance. Blunt: it softens the arm's "
                     "joints too. NOTE (measured 2026-08-14): this is NOT a RoboLab-vs-REALM "
                     "difference. RoboLab's articulation asks for 64, but its own scene sets "
                     "physxScene:maxPositionIterationCount=32, which the PhysxSchema documents as "
                     "overriding actors that request more -- so RoboLab solves at 32 too.")
ap.add_argument("--solver-vel-iter", type=int, default=None,
                help="physxArticulation:solverVelocityIterationCount on the ROBOT. This IS a real "
                     "difference: OmniGibson sets 1, RoboLab runs 0 (articulation asks 0, scene caps "
                     "at 1). Pass 0 for 'the value RoboLab runs'.")
ap.add_argument("--follower-max-effort", type=float, default=None,
                help="max_effort (N m) on the FOUR INNER MIMIC joints -- distinct from --max-effort, "
                     "which is the DRIVEN finger_joint. This is the ONLY joint-level difference "
                     "between the RoboLab asset and droid_robolab_v2 (measured by a full USD attr "
                     "diff, 2026-08-14): RoboLab's four inner mimic joints each carry a DriveAPI "
                     "with stiffness=0, damping=0, maxForce=INF, so Isaac Lab reads their effort "
                     "limit as FLT_MAX=3.4028235e38; REALM's converter strips that DriveAPI "
                     "(convert_robolab_gripper_usd.py:70, forced by OmniGibson's robot.py:658 "
                     "assertion that a DOF no controller claims must have no DriveAPI) and "
                     "OmniGibson then reports max_effort=0. Pass 3.4028235e38 for 'RoboLab's value'.")
ap.add_argument("--rungs", default="",
                help="SWEEP MODE: 'name=nf/dr/onf/odr/me/spi/kp/kd/ccs/ccd,name2=...'. Each rung gets its OWN unloaded "
                     "calibration sweep, free close (jaw-gap zero) and squeeze, all in one process. "
                     "*** RUNGS ARE CUMULATIVE: '-' means 'leave whatever the PREVIOUS rung left', "
                     "NOT 'the authored value'. Restate every field an earlier rung "
                     "touched, or an effort rung following an nf rung silently measures "
                     "both. Repeat a rung to get the error bar -- do that. ***"
                     "repeatability -- do that, it is the error bar on every other rung.")
ap.add_argument("--rung-free", type=int, default=1,
                help="in sweep mode, also do the free-mass squeeze + gravity-hold check per rung")
ap.add_argument("--video", type=int, default=1, help="0 skips every mp4 (frames are still captured)")
ap.add_argument("--play-cycle-check", type=int, default=1,
                help="at the very END, stop()/play() and read every override back: does it survive "
                     "the update_controller_mode() that simulator.py re-runs on each play?")
args = ap.parse_args()

ROBOT = args.robot
OUT = args.out
PFX = args.tag or ROBOT                     # output filename prefix
RUNGS_SPEC = args.rungs.strip()

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
if args.variant_usd:
    # Swap the asset BEFORE anything is loaded. Only the robolab v2 path is redirected, so a stock
    # A/B in the same session is unaffected, and the shipped file is never written to.
    assert os.path.exists(args.variant_usd), f"no variant USD at {args.variant_usd}"
    from omnigibson.robots.robot import Robot  # noqa: E402
    _orig_usd_path = Robot.usd_path.fget

    def _patched_usd_path(self):
        p = _orig_usd_path(self)
        if "droid_robolab_v2" in str(p):
            print(f"[variant] usd_path {p} -> {args.variant_usd}", flush=True)
            return args.variant_usd
        return p

    Robot.usd_path = property(_patched_usd_path)
    print(f"[variant] Robot.usd_path patched -> {args.variant_usd}", flush=True)
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

# ================================================================= MIMIC OVERRIDES
# The gain A/B (2026-08-14) REFUTED isaac_kp/isaac_kd as the compliance knob: the jaw penetrated the
# object by 1.20-1.39 mm at every rung from 1e7/1e5 down to 1e3/40, because the position drive
# saturates at max_effort=16.5 in all of them. That leaves two knobs that nothing in REALM's config
# touches, and this block is the only place either is written:
#
#   1. the PhysX MIMIC CONSTRAINT. The five followers have no drive at all (stiffness 0, damping 0,
#      max_effort 0 -- joint_prim.set_control_type() forces kp=kd=0 on any mimic DOF), so they are
#      held purely by PhysxMimicJointAPI. The asset authors naturalFrequency=1000 / dampingRatio=0.05
#      on the four inner joints and 1e6 / 0 on right_outer_knuckle_joint.
#      *** naturalFrequency is a SPRING frequency: bigger = stiffer, and ZERO means the constraint is
#      solved rigidly. So the softening direction is DOWNWARD from 1000, and 0 is the wrong end. ***
#   2. max_effort on the driven finger_joint, which is what the drive actually saturates at.
#
# Both are deliberately probe-local. They are physics hyperparameters: they change grasp behaviour on
# every task, not just this probe, so realm/config/robots/DROID_robolab_v2.yaml and the shipped USD
# are left alone and the values are passed on the command line.
#
# The mimic attributes are written straight onto the LIVE stage prims (USD-level). Whether omni.physx
# picks such a write up on an already-parsed articulation is NOT documented anywhere we can rely on,
# so it is measured rather than assumed: run a sweep whose first and last rungs are the same values
# and whose middle rungs are extreme. If every rung reports the same jaw gap, the write did not
# propagate and these numbers mean nothing -- IDENTICAL_RUNGS below says so out loud.
MIMIC_ATTRS = ("naturalFrequency", "dampingRatio", "gearing", "offset")
OUTER_J = "right_outer_knuckle_joint"

import omnigibson.lazy as lazy  # noqa: E402


def mimic_insts(prim):
    """The PhysxMimicJointAPI instance names applied to @prim (e.g. ['rotX']), read at runtime.

    The instance token is NOT the joint's physics:axis -- these joints author axis Z and instance
    rotX -- so it has to be discovered, never guessed.
    """
    return [s.split(":", 1)[1] for s in prim.GetAppliedSchemas()
            if s.startswith("PhysxMimicJointAPI:")]


MIMIC_JOINTS = [joint_names[i] for i in grip_idx if mimic_insts(robot.joints[joint_names[i]].prim)]
INNER_MIMIC = [n for n in MIMIC_JOINTS if n != OUTER_J]
if args.mimic_joints:
    want = [x.strip() for x in args.mimic_joints.split(",") if x.strip()]
    bad = [n for n in want if n not in MIMIC_JOINTS]
    assert not bad, f"--mimic-joints names {bad} which are not mimic joints; have {MIMIC_JOINTS}"
    INNER_MIMIC = want
DRIVEN_J = joint_names[int(CTRL_DOF[0])]


def mimic_state():
    """Every mimic attribute on every mimic gripper joint, keyed 'joint.inst.attr'."""
    out = {}
    for n in MIMIC_JOINTS:
        prim = robot.joints[n].prim
        for inst in mimic_insts(prim):
            for a in MIMIC_ATTRS:
                at = prim.GetAttribute(f"physxMimicJoint:{inst}:{a}")
                out[f"{n}.{inst}.{a}"] = None if not at.IsValid() else at.Get()
    return out


def mimic_set(names, nf=None, dr=None):
    """Write naturalFrequency / dampingRatio on @names' mimic APIs. Returns {key: value written}.

    The write MUST sit inside `og.sim.editing_usd()`. Without it OmniGibson's guard raises
    "USD edit detected outside of og.sim.editing_usd() context!" (simulator.py:1651) -- and that
    context is also what synchronises the edit into Fabric, i.e. it is the thing that gives the write
    any chance of reaching PhysX at all. Measured 2026-08-14: writing bare raises immediately.
    """
    wrote = {}
    with og.sim.editing_usd():
        for n in names:
            prim = robot.joints[n].prim
            insts = mimic_insts(prim)
            assert insts, f"{n} has no PhysxMimicJointAPI -- nothing to soften"
            for inst in insts:
                for a, v in (("naturalFrequency", nf), ("dampingRatio", dr)):
                    if v is None:
                        continue
                    at = prim.GetAttribute(f"physxMimicJoint:{inst}:{a}")
                    assert at.IsValid(), f"{n} has no physxMimicJoint:{inst}:{a}"
                    at.Set(float(v))
                    wrote[f"{n}.{inst}.{a}"] = float(v)
    return wrote


def effort_set(jname, v):
    """max_effort on @jname, through the articulation view. Returns (before, after)."""
    j = robot.joints[jname]
    before = jget(j, "max_effort")
    j.max_effort = float(v)
    return before, jget(j, "max_effort")


def usd_drive(jname):
    """The USD-authored drive block for @jname, for comparison against the live view."""
    prim = robot.joints[jname].prim
    out = {}
    for a in prim.GetAttributes():
        nm = a.GetName()
        if nm.startswith("drive:"):
            out[nm] = a.Get()
    return out


def pad_material_prims():
    """The physics-material prims bound to the two PAD links' collision geoms.

    PhysX applies compliant contact per MATERIAL, so softening "the pads" means finding whatever
    material their collision geoms resolved to. Returned keyed by link/geom so the log says which.
    """
    out = {}
    for ln in FL:
        for gname, gm in robot.links[ln].collision_meshes.items():
            pm = None
            try:
                pm = gm.get_applied_physics_material()
            except Exception as e:
                print(f"  [warn] no physics material for {ln}/{gname}: {e!r}")
            if pm is None:
                continue
            prim = getattr(pm, "prim", None)
            if prim is None:
                prim = og.sim.stage.GetPrimAtPath(pm.prim_path)
            out[f"{ln}/{gname}"] = prim
    return out


def pad_cc_state():
    out = {}
    for k, prim in PAD_MATS.items():
        for a in ("compliantContactStiffness", "compliantContactDamping"):
            at = prim.GetAttribute(f"physxMaterial:{a}")
            out[f"{k}.{a}"] = None if not at.IsValid() else at.Get()
    return out


def pad_cc_set(stiff=None, damp=None):
    """Write compliant-contact parameters on the pad materials. Same editing_usd() rule as mimic_set."""
    wrote = {}
    with og.sim.editing_usd():
        for k, prim in PAD_MATS.items():
            for a, v in (("compliantContactStiffness", stiff), ("compliantContactDamping", damp)):
                if v is None:
                    continue
                nm = f"physxMaterial:{a}"
                at = prim.GetAttribute(nm)
                if not at.IsValid():
                    at = prim.CreateAttribute(nm, lazy.pxr.Sdf.ValueTypeNames.Float, custom=False)
                at.Set(float(v))
                wrote[f"{k}.{a}"] = float(v)
    return wrote


hdr("MIMIC CONSTRAINT / EFFORT STATE -- as authored, before any override")
# FIRST: is naturalFrequency even a thing in THIS build? An attribute that is not in the schema the
# runtime was built against is inert text in the file -- omni.physx never reads it, and the mimic
# constraint is then solved as a rigid equality from gearing/offset alone. Measured on
# isaacsim 5.1.0 / omni.physx 107.3.26: PhysxMimicJointAPI declares only gearing, offset,
# referenceJoint, referenceJointAxis, and the shipped _physxSchema.so does not contain the string
# "naturalFrequency" at all. Printed on every run rather than trusted.
_reg = lazy.pxr.Usd.SchemaRegistry()
_pd = _reg.FindAppliedAPIPrimDefinition("PhysxMimicJointAPI")
SCHEMA_PROPS = list(_pd.GetPropertyNames()) if _pd is not None else []
NF_IN_SCHEMA = "physxMimicJoint:__INSTANCE_NAME__:naturalFrequency" in SCHEMA_PROPS
print(f"  PhysxMimicJointAPI schema properties = {SCHEMA_PROPS}")
print(f"  naturalFrequency in the schema = {NF_IN_SCHEMA};  "
      f"hasattr(PhysxMimicJointAPI, 'GetNaturalFrequencyAttr') = "
      f"{hasattr(lazy.pxr.PhysxSchema.PhysxMimicJointAPI, 'GetNaturalFrequencyAttr')}")
if not NF_IN_SCHEMA:
    print("  *** MIMIC_NF_NOT_IN_SCHEMA *** the asset authors physxMimicJoint:<inst>:naturalFrequency\n"
          "      and :dampingRatio, but this build's PhysxMimicJointAPI does not declare either, and\n"
          "      the shipped _physxSchema.so does not contain the string at all. omni.physx cannot\n"
          "      read an attribute that is not in its schema, so those values are INERT TEXT and the\n"
          "      mimic constraint is solved RIGIDLY from gearing/offset alone. A --mimic-nf sweep is\n"
          "      then expected to change nothing, and that null result IS the measurement.")
_mpd = _reg.FindAppliedAPIPrimDefinition("PhysxMaterialAPI")
_mprops = list(_mpd.GetPropertyNames()) if _mpd is not None else []
CC_IN_SCHEMA = "physxMaterial:compliantContactStiffness" in _mprops
print(f"  PhysxMaterialAPI compliant-contact props = "
      f"{[x for x in _mprops if 'ompliant' in x] or '(none)'}   in-schema={CC_IN_SCHEMA}")
PAD_MATS = pad_material_prims()
print(f"  pad physics materials ({len(PAD_MATS)}): "
      + (", ".join(f"{k} -> {v.GetPath()}" for k, v in PAD_MATS.items()) or "(none found)"))
PAD_CC0 = pad_cc_state()
for k, v in PAD_CC0.items():
    print(f"    {k:<62} = {v}")
print(f"  mimic joints ({len(MIMIC_JOINTS)}): {MIMIC_JOINTS}")
print(f"  the ones this sweep softens ({len(INNER_MIMIC)}): {INNER_MIMIC}"
      + ("   [--mimic-joints]" if args.mimic_joints else "   [default: the four inner]"))
print(f"  the driven joint: {DRIVEN_J}")
MIMIC0 = mimic_state()
for k, v in MIMIC0.items():
    print(f"    {k:<62} = {v}")
print(f"  {DRIVEN_J} live view: stiffness={jget(robot.joints[DRIVEN_J], 'stiffness')} "
      f"damping={jget(robot.joints[DRIVEN_J], 'damping')} "
      f"max_effort={jget(robot.joints[DRIVEN_J], 'max_effort')}")
print(f"  {DRIVEN_J} USD drive block: {usd_drive(DRIVEN_J)}")
EFFORT0 = jget(robot.joints[DRIVEN_J], "max_effort")


def apply_override(nf=None, dr=None, onf=None, odr=None, me=None, spi=None, kp=None, kd=None,
                   ccs=None, ccd=None, label="", fme=None, svi=None):
    """Apply one rung's mimic / effort values and READ THEM BACK. Returns what is live afterwards.

    @fme and @svi are the two wrapper-diff knobs and are single-shot only (not rung fields), so the
    cumulative-rung caller keeps its existing positional contract.
    """
    wrote = {}
    if nf is not None or dr is not None:
        wrote.update(mimic_set(INNER_MIMIC, nf=nf, dr=dr))
    if onf is not None or odr is not None:
        assert OUTER_J in MIMIC_JOINTS, f"{OUTER_J} is not a mimic joint on this asset"
        wrote.update(mimic_set([OUTER_J], nf=onf, dr=odr))
    if ccs is not None or ccd is not None:
        w = pad_cc_set(ccs, ccd)
        print(f"  [override {label or 'single'}] pad compliant contact -> {w}")
        wrote.update(w)
    eff = None
    if me is not None:
        eff = effort_set(DRIVEN_J, me)
    if kp is not None or kd is not None:
        j = robot.joints[DRIVEN_J]
        was = (jget(j, "stiffness"), jget(j, "damping"))
        if kp is not None:
            j.stiffness = float(kp)
        if kd is not None:
            j.damping = float(kd)
        print(f"  [override {label or 'single'}] {DRIVEN_J} drive gains {was} -> "
              f"({jget(j, 'stiffness')}, {jget(j, 'damping')})  [per-radian view convention]")
    if spi is not None:
        was = robot.solver_position_iteration_count
        robot.solver_position_iteration_count = int(spi)
        print(f"  [override {label or 'single'}] solverPositionIterationCount {was} -> "
              f"{robot.solver_position_iteration_count}")
    if svi is not None:
        was = robot.solver_velocity_iteration_count
        robot.solver_velocity_iteration_count = int(svi)
        print(f"  [override {label or 'single'}] solverVelocityIterationCount {was} -> "
              f"{robot.solver_velocity_iteration_count}  (RoboLab runs 0, OmniGibson sets 1)")
    if fme is not None:
        # The followers' effort limit, NOT the driven joint's. Read back per joint: these DOFs have
        # no DriveAPI at all on droid_robolab_v2, so it is not a given that the articulation view
        # will accept a max-effort write on them -- if it silently does not stick, that is itself
        # the finding, so print before/after rather than assuming.
        fw = {n: effort_set(n, fme) for n in INNER_MIMIC}
        print(f"  [override {label or 'single'}] follower max_effort -> "
              + ", ".join(f"{n}: {b} -> {a}" for n, (b, a) in fw.items()))
        stuck = [n for n, (b, a) in fw.items() if a is None or abs(a - float(fme)) > 1e-3 * abs(float(fme))]
        if stuck:
            print(f"  [override {label or 'single'}] *** follower max_effort did NOT stick on "
                  f"{stuck} -- treat any null result below as untested, not as refuted ***")
    print(f"  [override {label or 'single'}] wrote {wrote or '(no mimic change)'}"
          + (f"; max_effort {eff[0]} -> {eff[1]}" if eff else "; max_effort unchanged"))
    live = mimic_state()
    bad = [k for k, v in wrote.items() if live.get(k) is None or abs(live[k] - v) > 1e-6 * max(1.0, abs(v))]
    print(f"  [override {label or 'single'}] READBACK "
          + ", ".join(f"{k.split('.')[0]}.{k.split('.')[-1]}={live[k]}" for k in sorted(wrote))
          + (f"   *** MISMATCH on {bad} ***" if bad else "   (all writes read back)"))
    assert not bad, f"mimic write did not stick: {bad}"
    return dict(live_mimic=live, max_effort=jget(robot.joints[DRIVEN_J], "max_effort"),
                stiffness=jget(robot.joints[DRIVEN_J], "stiffness"),
                damping=jget(robot.joints[DRIVEN_J], "damping"),
                solver_pos_iter=int(robot.solver_position_iteration_count),
                solver_vel_iter=int(robot.solver_velocity_iteration_count))


OVERRIDE_SINGLE = None
if any(v is not None for v in (args.mimic_nf, args.mimic_dr, args.outer_nf, args.outer_dr,
                              args.max_effort, args.solver_pos_iter, args.drive_kp,
                              args.drive_kd, args.pad_cc_stiffness, args.pad_cc_damping,
                              args.follower_max_effort, args.solver_vel_iter)):
    hdr("APPLYING THE SINGLE-SHOT OVERRIDE (before the calibration sweep, so the reference curve "
        "belongs to THIS configuration)")
    OVERRIDE_SINGLE = apply_override(args.mimic_nf, args.mimic_dr, args.outer_nf, args.outer_dr,
                                     args.max_effort, args.solver_pos_iter, args.drive_kp,
                                     args.drive_kd, args.pad_cc_stiffness, args.pad_cc_damping,
                                     fme=args.follower_max_effort, svi=args.solver_vel_iter)

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
        tag=tag, cmd=cmd, rung=RUNG,
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
    # --video 0 skips the frame COPY, not just the encode: a 12-rung sweep is ~2500 steps and at
    # 720x1280x3 that would be 20+ GB of retained RGB.
    if not args.video:
        return
    ext = o.get("external", {})
    if "external_sensor0" in ext:
        frames.append(ext["external_sensor0"]["rgb"].cpu().numpy()[..., :3].copy())
    if "external_sensor1" in ext and not RUNGS_SPEC:
        frames_wide.append(ext["external_sensor1"]["rgb"].cpu().numpy()[..., :3].copy())


# Jaw-gap zeros, one set PER RUNG. Softening the mimic constraint can in principle shift the
# unloaded closed pose too, so every rung takes its own free close and every row remembers which
# rung it belongs to; sharing one zero across rungs would fold that shift into the "compliance".
RUNG = ""
ZEROS = {"": dict(sep=None, hull=None)}


def jaw(r, which="sep"):
    """Physical jaw gap (m): the raw measure minus ITS OWN RUNG's value at full unloaded closure."""
    z = ZEROS.get(r.get("rung", ""), ZEROS[""])[which]
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

# ---------------------------------------------------------------- reusable phase blocks
# Every phase below is a function so that a SWEEP (--rungs) can repeat the whole open -> calibrate ->
# free-close -> squeeze cycle once per mimic/effort configuration inside a single process. That
# matters for more than boot time: all rungs then share one scene instance, one contact-view layout
# and one arm pose, so a rung-to-rung difference cannot be a boot-to-boot difference. Repeating a
# rung gives the error bar directly.
cube_home = _np(cube.get_position_orientation()[0])
CUBE_MASS0 = float(cube.root_link.mass)


def park_cube():
    """Park the cube 1.3 m below its home, gravity off, where it can touch nothing.

    Gravity stays off for the parking AND for the squeezes: an OPEN hand cannot hold an unsupported
    object up, and letting it free-fall while the jaws travel would end the experiment before it
    starts. It is restored only in the explicit hold test.
    """
    cube.disable_gravity()
    cube.set_position_orientation(th.tensor(cube_home + np.array([0.0, 0.0, -1.3]), dtype=th.float32))
    cube.keep_still()


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


def cal_sweep(pfx=""):
    """Slow UNLOADED sweep -> the linkage's kinematics, densely sampled. Returns the rows.

    The binary drive is a position target with isaac_kp=1e7, which slews the whole 0.785 rad in ONE
    15 Hz control step: the free close yields only ~3-6 distinct driven-joint angles, and linear
    interpolation between them misses the four-bar's curvature by 2-3 mm -- twenty times the
    deflection being measured. So this curve is taken with the leader's drive gains temporarily
    softened, which slows the sweep without changing the KINEMATIC relation it records (the sweep is
    unloaded and quasi-static; only the geometry is being read).

    The poke goes straight onto the joint rather than into the controller config because it must be
    reversible inside one process. That is only safe because this probe does not call og.sim.play()
    between here and the measurement: simulator.py re-applies update_controller_mode() on every play,
    which would silently restore isaac_kp/isaac_kd. A PERMANENT gain change belongs in gripper_0.
    """
    if args.cal_steps <= 0:
        return []
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
        do(f"{pfx}cal_close", GRIP_CLOSE, args.cal_steps, verbose_every=12)
        do(f"{pfx}cal_open", GRIP_OPEN, args.cal_steps, verbose_every=12)
        out = phase(f"{pfx}cal_close") + phase(f"{pfx}cal_open")
    finally:
        for j, (k, d) in zip(leads, gains0):
            j.stiffness = k
            j.damping = d
        print(f"  gains restored to "
              f"{[f'{float(_np(j.stiffness)):.1e}/{float(_np(j.damping)):.1e}' for j in leads]} "
              f"(must equal {[f'{k:.1e}/{d:.1e}' for k, d in gains0]})")
    do(f"{pfx}reopen_cal", GRIP_OPEN, 10, verbose_every=9)
    qs = np.array([r["q"][int(CTRL_DOF[0])] for r in out])
    print(f"  swept {len(out)} samples, driven joint {qs.min():+.4f} .. {qs.max():+.4f}, "
          f"{len(np.unique(np.round(qs, 4)))} distinct values")
    return out


def free_close_phase(pfx="", r_open=None):
    """Unloaded close at the REAL gains. Sets this rung's jaw-gap zeros. Returns the rows."""
    r_free = do(f"{pfx}free_close", GRIP_CLOSE, args.close_steps, verbose_every=12)
    # Full unloaded closure = pads touching = zero physical gap. Everything reported as a "jaw gap"
    # from here on is relative to this, which is what makes rungs and assets comparable.
    ZEROS[RUNG] = dict(sep=r_free["sep_origin"], hull=r_free["gap_hull"])
    print(f"  jaw-gap zeros for rung '{RUNG}': sep_origin={ZEROS[RUNG]['sep'] * 1000:.3f} mm, "
          f"gap_hull={ZEROS[RUNG]['hull'] * 1000:.3f} mm -> both read 0.000 mm when shut")
    if r_open is not None:
        print(f"  jaws OPEN therefore measured {jaw(r_open) * 1000:.2f} mm (sep) / "
              f"{jaw(r_open, 'hull') * 1000:.2f} mm (hull) -- these two agreeing is the cross-check")
    return phase(f"{pfx}free_close")


def squeeze_phase(label, mass, grav_after, pfx=""):
    """Reopen, place the cube at the pad midpoint, close on it. Returns the hold verdict, if tested."""
    hdr(f"{pfx}SQUEEZE {label}: REOPEN, then close "
        f"({'mass as authored, free to move' if mass is None else f'mass {mass} kg = immovable'})")
    do(f"{pfx}reopen_{label}", GRIP_OPEN, args.open_steps, verbose_every=15)
    cube.root_link.mass = CUBE_MASS0 if mass is None else float(mass)
    cube.disable_gravity()                     # an open hand cannot hold it up; see park_cube()
    M_pads = place_cube()
    cube.keep_still()
    print(f"  cube placed at the pad midpoint {M_pads}, mass = {cube.root_link.mass:.4f} kg, "
          f"gravity DISABLED")
    do(f"{pfx}settle_{label}", GRIP_OPEN, 12, verbose_every=6)
    r_pre = rows[-1]
    print(f"  pre-squeeze: jaw={jaw(r_pre) * 1000:.2f} mm, object {2000 * CUBE_HALF:.2f} mm wide, "
          f"so {(jaw(r_pre) - 2 * CUBE_HALF) * 500:.2f} mm of approach per pad; "
          f"drift from the midpoint={r_pre['cube_off'] * 1000:.2f} mm "
          f"({r_pre['cube_off_a'] * 1000:+.2f} along the closing axis); "
          f"hull-vs-pose diagnostic {r_pre['hull_off'] * 1000:+.1f} mm")

    print(f"  --- squeezing (binary close onto the object)")
    do(f"{pfx}squeeze_{label}", GRIP_CLOSE, args.load_steps, verbose_every=8)

    held = None
    if grav_after:
        print(f"  --- RESTORE GRAVITY while still commanding CLOSE -- is it HELD?")
        cube.enable_gravity()
        try:
            cube.wake()
        except Exception:
            pass
        p_before = _np(cube.get_position_orientation()[0])
        do(f"{pfx}gravity_{label}", GRIP_CLOSE, args.grav_steps, verbose_every=20)
        p_after = _np(cube.get_position_orientation()[0])
        drop = float(p_before[2] - p_after[2])
        held = bool(abs(drop) < 0.01)
        print(f"  cube fell {drop * 1000:+.2f} mm in {args.grav_steps} steps "
              f"({args.grav_steps / 15.0:.1f} s); total displacement "
              f"{np.linalg.norm(p_after - p_before) * 1000:.2f} mm")
        print(f"  VERDICT: {'HELD' if held else 'DROPPED / SLIPPED'}  (drop_mm={drop * 1000:+.3f})")
        cube.disable_gravity()
        return dict(held=held, drop_mm=drop * 1e3)
    return dict(held=None, drop_mm=None)

# ---------------------------------------------------------------- analysis
REF = int(CTRL_DOF[0])          # the driven joint the linkage is parameterised by


def build_cal(cal_rows, free_rows):
    """The unloaded reference curve + fitted mimic gearing for ONE rung. Returns a dict.

    Per-rung rather than global: softening the mimic constraint could in principle change the
    unloaded kinematics too, and comparing a soft rung's loaded pose against the STIFF rung's
    unloaded curve would book that change as compliance.
    """
    # The dense slow sweep when there is one, else the sparse free close.
    cal = cal_rows if len(cal_rows) > len(free_rows) else free_rows
    q_free = np.stack([r["q"] for r in cal])
    ref_free = q_free[:, REF]
    order = np.argsort(ref_free)
    # Mimic gearing, fitted on the unloaded sweep: q_follower = G_j * q_leader + O_j. When the mimic
    # constraint is rigid this relation is exact to well under a milliradian unloaded, so the residual
    # under load is a deflection measurement that needs no interpolation at all -- the strongest of
    # the compliance numbers here. Fitted rather than read from the USD so it also works on the stock
    # asset, whose four gripper joints are independently driven and have no mimic API.
    gear = {}
    A = np.stack([ref_free, np.ones_like(ref_free)], axis=1)
    for i in grip_idx:
        sol, *_ = np.linalg.lstsq(A, q_free[:, i], rcond=None)
        resid = q_free[:, i] - (A @ sol)
        gear[int(i)] = (float(sol[0]), float(sol[1]), float(np.abs(resid).max()))
    return dict(rows=cal, is_sweep=cal is cal_rows, q_free=q_free, ref_free=ref_free, order=order,
                gap_free=np.array([jaw(r, "hull") for r in cal]),
                sep_free=np.array([jaw(r) for r in cal]), gear=gear)


def free_at(cal, vals, x):
    """The unloaded value of @vals at driven-joint angle @x, from @cal's sweep."""
    return np.interp(x, cal["ref_free"][cal["order"]], np.asarray(vals)[cal["order"]])


def report_cal(cal, r_open, free_rows):
    print(f"  reference (driven) joint = [{REF}] {joint_names[REF]}")
    print(f"  unloaded reference curve = {'slow sweep' if cal['is_sweep'] else 'free close'}, "
          f"{len(cal['rows'])} samples")
    print("  unloaded follower relation q_j = G*q_ref + O (residual = how rigid the coupling is "
          "with NO load):")
    for i in grip_idx:
        G, O, res = cal["gear"][int(i)]
        print(f"    {joint_names[i]:<36} G={G:+.5f} O={O:+.6f}  max residual {res:.6f}")
    print(f"\n  UNLOADED EXTREMES (open -> unloaded shut)")
    print(f"    open : jaw={jaw(r_open) * 1000:8.2f} mm (hull {jaw(r_open, 'hull') * 1000:.2f})  "
          f"raw sep_origin={r_open['sep_origin'] * 1000:7.2f} mm  q_ref={r_open['q'][REF]:+.5f}")
    fl = free_rows[-1]
    print(f"    shut : jaw={jaw(fl) * 1000:8.2f} mm (hull {jaw(fl, 'hull') * 1000:.2f})  "
          f"raw sep_origin={fl['sep_origin'] * 1000:7.2f} mm  q_ref={fl['q'][REF]:+.5f}")
    print(f"    gripper joints open -> shut:")
    for i in grip_idx:
        print(f"      {joint_names[i]:<36} {r_open['q'][i]:+.6f} -> {fl['q'][i]:+.6f}  "
              f"(travel {fl['q'][i] - r_open['q'][i]:+.6f})")
    if CLOSE_TARGET is not None:
        print(f"    commanded close target = {CLOSE_TARGET}; unloaded residual "
              f"{CLOSE_TARGET - fl['q'][CTRL_DOF]}")


def grip_params():
    return {joint_names[i]: dict(
        stiffness=jget(robot.joints[joint_names[i]], "stiffness"),
        damping=jget(robot.joints[joint_names[i]], "damping"),
        max_effort=jget(robot.joints[joint_names[i]], "max_effort"),
        is_mimic=jget(robot.joints[joint_names[i]], "is_mimic_joint"),
        driven=jget(robot.joints[joint_names[i]], "driven")) for i in grip_idx}


summary = dict(robot=ROBOT, task=args.task_cfg, joint_names=joint_names,
               ref_joint=joint_names[REF], tag=args.tag,
               isaac_kp=None if CLOSE_TARGET is None else (
                   None if gc.isaac_kp is None else _np(gc.isaac_kp).tolist()),
               isaac_kd=None if CLOSE_TARGET is None else (
                   None if gc.isaac_kd is None else _np(gc.isaac_kd).tolist()),
               mimic_authored={k: (None if v is None else float(v)) for k, v in MIMIC0.items()},
               mimic_nf_in_schema=NF_IN_SCHEMA, mimic_schema_props=SCHEMA_PROPS,
               compliant_contact_in_schema=CC_IN_SCHEMA,
               pad_materials=[str(v.GetPath()) for v in PAD_MATS.values()],
               mimic_joints=MIMIC_JOINTS, inner_mimic=INNER_MIMIC, driven_joint=DRIVEN_J,
               max_effort_authored=EFFORT0,
               override_single=None if OVERRIDE_SINGLE is None else dict(
                   mimic_nf=args.mimic_nf, mimic_dr=args.mimic_dr, outer_nf=args.outer_nf,
                   outer_dr=args.outer_dr, max_effort=args.max_effort,
                   solver_pos_iter=args.solver_pos_iter, drive_kp=args.drive_kp,
                   drive_kd=args.drive_kd, pad_cc_stiffness=args.pad_cc_stiffness,
                   pad_cc_damping=args.pad_cc_damping),
               close_target=None if CLOSE_TARGET is None else CLOSE_TARGET.tolist(),
               ctrl_dof=[int(i) for i in CTRL_DOF], squeezes={}, rungs={})


def analyse(label, cal, sq, pre, r_open, free_rows):
    """The full per-squeeze compliance report. Returns the summary dict (None if never in contact)."""
    hdr(f"ANALYSIS {label}")
    report_cal(cal, r_open, free_rows)
    touch = [i for i, r in enumerate(sq) if r["n_contact"] > 0]
    both = [i for i, r in enumerate(sq) if r["n_contact"] == 2]
    if not touch:
        print("  NO CONTACT AT ALL -- the jaws never reached the object. Nothing to report.")
        return None
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
            span = free_rows[-1]["q"][i] - r_open["q"][i]
            res = CLOSE_TARGET[k] - last["q"][i]
            print(f"    {joint_names[i]:<36} stalled at {last['q'][i]:+.6f}, target "
                  f"{CLOSE_TARGET[k]:+.6f}  -> unresolved {res:+.6f} "
                  f"({100.0 * res / span if span else float('nan'):+.1f}% of its full travel)")
    print(f"    in jaw terms: the pads sit at {jaw(last) * 1000:.2f} mm while the command "
          f"asks for 0.00 mm -> {jaw(last) * 1000:.2f} mm of jaw closure blocked by the object")

    print(f"\n  COMPLIANCE = deviation from the UNLOADED linkage at the SAME driven angle "
          f"(q_ref={ref_l:+.6f})")
    gap_pred = free_at(cal, cal["gap_free"], ref_l)
    sep_pred = free_at(cal, cal["sep_free"], ref_l)
    print(f"    jaw gap (hull) : loaded {jaw(last, 'hull') * 1000:8.3f} mm vs unloaded-at-same-q "
          f"{gap_pred * 1000:8.3f} mm  -> FLEX {(jaw(last, 'hull') - gap_pred) * 1000:+8.3f} mm")
    print(f"    jaw gap (orig) : loaded {jaw(last) * 1000:8.3f} mm vs "
          f"{sep_pred * 1000:8.3f} mm  -> FLEX {(jaw(last) - sep_pred) * 1000:+8.3f} mm")
    devs, devs_gear = {}, {}
    print(f"    per gripper joint (rad or m), two independent estimators:")
    print(f"      {'joint':<36} {'loaded':>10} {'interp':>10} {'FLEX':>10} {'G*q+O':>10} {'FLEX':>10}")
    for i in grip_idx:
        pred = free_at(cal, cal["q_free"][:, i], ref_l)
        d = last["q"][i] - pred
        G, O, _ = cal["gear"][int(i)]
        pg = G * ref_l + O
        dg = last["q"][i] - pg
        devs[joint_names[i]] = float(d)
        devs_gear[joint_names[i]] = float(dg)
        print(f"      {joint_names[i]:<36} {last['q'][i]:+10.6f} {pred:+10.6f} {d:+10.6f} "
              f"{pg:+10.6f} {dg:+10.6f}")
    # Worst flex over the whole squeeze, not only at the last step.
    flex_traj = np.array([jaw(r, "hull") - free_at(cal, cal["gap_free"], r["q"][REF]) for r in sq])
    joint_flex_traj = {joint_names[i]: float(np.abs(
        np.array([r["q"][i] - free_at(cal, cal["q_free"][:, i], r["q"][REF]) for r in sq])).max())
        for i in grip_idx}
    joint_flex_gear = {joint_names[i]: float(np.abs(
        np.array([r["q"][i] - (cal["gear"][int(i)][0] * r["q"][REF] + cal["gear"][int(i)][1]) for r in sq])).max())
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
    out = dict(
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
    # Per-newton jaw softness, the number the two assets were compared on: how far the pads went past
    # the object's own surface, divided by the peak pad force. Higher = softer.
    out["um_per_N"] = (float("nan") if not out["max_force_N"]
                       else out["past_object_width_mm"] * 1e3 / out["max_force_N"])
    out["n_contact_final"] = int(last["n_contact"])
    out["frac_both_pads"] = float(np.mean([r["n_contact"] == 2 for r in sq]))
    out["mimic_live"] = {k: (None if v is None else float(v)) for k, v in mimic_state().items()}
    out["pad_cc_live"] = {k: (None if v is None else float(v)) for k, v in pad_cc_state().items()}
    out["max_effort_live"] = jget(robot.joints[DRIVEN_J], "max_effort")
    out["stiffness_live"] = jget(robot.joints[DRIVEN_J], "stiffness")
    out["damping_live"] = jget(robot.joints[DRIVEN_J], "damping")
    print(f"    jaw softness per newton               = {out['um_per_N']:.1f} um/N")
    print(f"    pad contacts at the end / fraction of the squeeze with BOTH pads touching = "
          f"{out['n_contact_final']} / {out['frac_both_pads']:.2f}   "
          f"(is_grasping needs both -- a rung that bends but loses a pad breaks grasp detection)")
    return out


# ================================================================= RUN
if not RUNGS_SPEC:
    # ---- the original single-configuration run: OPEN -> sweep -> free close -> squeeze A -> squeeze B
    hdr("PHASE 1: OPEN -- the reference state")
    r_open = do("open", GRIP_OPEN, args.open_steps, verbose_every=10)
    park_cube()
    hdr("PHASE 1b: SLOW UNLOADED SWEEP -- the linkage's KINEMATICS, densely sampled")
    CAL_ROWS = cal_sweep()
    hdr("PHASE 2: FREE CLOSE -- nothing between the jaws, at the real drive gains")
    free = free_close_phase(r_open=r_open)
    CAL = build_cal(CAL_ROWS, free)
    holds = {}
    for label, mass, grav_after in (("A", None, True), ("B", args.pin_mass, False)):
        holds[label] = squeeze_phase(label, mass, grav_after)
    for label in ("A", "B"):
        sq = phase(f"squeeze_{label}")
        if not sq:
            continue
        res = analyse(f"{label}: {'free object' if label == 'A' else 'immovable object'}",
                      CAL, sq, phase(f"settle_{label}")[-1], r_open, free)
        if res is not None:
            res.update(holds[label])
            summary["squeezes"][label] = res
    summary["grip_joint_params"] = grip_params()
    summary["gearing"] = {joint_names[i]: CAL["gear"][int(i)] for i in grip_idx}
    summary["open"] = dict(jaw_mm=jaw(r_open) * 1e3, jaw_hull_mm=jaw(r_open, "hull") * 1e3,
                           sep_mm=r_open["sep_origin"] * 1e3, q=r_open["q"].tolist())
    summary["free_shut"] = dict(jaw_mm=jaw(free[-1]) * 1e3, sep_mm=free[-1]["sep_origin"] * 1e3,
                                q=free[-1]["q"].tolist())
    RUNG_SPANS = []
else:
    # ---- SWEEP: one full open/calibrate/free-close/squeeze cycle per mimic-or-effort configuration
    def parse_rungs(spec):
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, vals = part.partition("=")
            fields = (vals.split("/") + ["-"] * 10)[:10]
            conv = lambda s: None if s.strip() in ("-", "") else float(s)  # noqa: E731
            out.append(dict(name=name.strip(), nf=conv(fields[0]), dr=conv(fields[1]),
                            onf=conv(fields[2]), odr=conv(fields[3]), me=conv(fields[4]),
                            spi=None if conv(fields[5]) is None else int(conv(fields[5])),
                            kp=conv(fields[6]), kd=conv(fields[7]),
                            ccs=conv(fields[8]), ccd=conv(fields[9])))
        return out

    RUNGS = parse_rungs(RUNGS_SPEC)
    hdr(f"SWEEP MODE -- {len(RUNGS)} rungs in ONE process")
    for k, rg in enumerate(RUNGS):
        print(f"  [{k}] {rg['name']:<14} inner nf={rg['nf']} dr={rg['dr']}   "
              f"{OUTER_J} nf={rg['onf']} dr={rg['odr']}   max_effort={rg['me']}  "
              f"solver_pos_iter={rg['spi']}  kp={rg['kp']} kd={rg['kd']}  "
              f"cc={rg['ccs']}/{rg['ccd']}")
    print("  Every rung takes its OWN unloaded calibration sweep and free close, so its jaw-gap zero "
          "and reference kinematics belong to its own configuration.")
    print("  A repeated rung is the ERROR BAR: if 'default' and its repeat differ by X, no other "
          "rung's difference below X means anything.")
    RUNG_SPANS = []
    for k, rg in enumerate(RUNGS):
        RUNG = rg["name"] if rg["name"] not in ZEROS else f"{rg['name']}#{k}"
        ZEROS[RUNG] = dict(sep=None, hull=None)
        f0 = len(rows)
        hdr(f"RUNG {k + 1}/{len(RUNGS)}: {RUNG}   inner nf={rg['nf']} dr={rg['dr']}  "
            f"outer nf={rg['onf']} dr={rg['odr']}  max_effort={rg['me']}  spi={rg['spi']}  "
            f"kp={rg['kp']} kd={rg['kd']}  cc={rg['ccs']}/{rg['ccd']}")
        live = apply_override(rg["nf"], rg["dr"], rg["onf"], rg["odr"], rg["me"], rg["spi"],
                              rg["kp"], rg["kd"], rg["ccs"], rg["ccd"], label=RUNG)
        park_cube()
        p = f"{RUNG}_"
        r_open = do(f"{p}open", GRIP_OPEN, args.open_steps, verbose_every=15)
        CAL_ROWS = cal_sweep(p)
        free = free_close_phase(p, r_open=r_open)
        CAL = build_cal(CAL_ROWS, free)
        rec = dict(name=RUNG, spec=rg, live=live, squeezes={})
        for label, mass, grav_after in ((("A", None, True),) if args.rung_free else ()) + \
                                       (("B", args.pin_mass, False),):
            hold = squeeze_phase(label, mass, grav_after, pfx=p)
            sq = phase(f"{p}squeeze_{label}")
            res = analyse(f"{RUNG} / {label}", CAL, sq, phase(f"{p}settle_{label}")[-1], r_open, free)
            if res is not None:
                res.update(hold)
                rec["squeezes"][label] = res
        rec["open_jaw_mm"] = jaw(r_open) * 1e3
        rec["free_shut_q_ref"] = float(free[-1]["q"][REF])
        rec["gear_resid"] = {joint_names[i]: CAL["gear"][int(i)][2] for i in grip_idx}
        summary["rungs"][RUNG] = rec
        RUNG_SPANS.append((RUNG, f0, len(rows)))
        park_cube()
    RUNG = ""

    # ---- the cross-rung table, and the check that the knob did anything at all
    hdr("SWEEP TABLE -- squeeze B (immovable object), the clean compliance number")
    print(f"  {'rung':<16} {'inner nf':>9} {'dr':>6} {'outer nf':>10} {'odr':>5} {'maxeff':>7} {'spi':>4} {'kp':>9} "
          f"{'jaw@stall':>10} {'past obj':>9} {'jawflex':>8} {'unres q':>8} {'F_l':>7} {'F_r':>7} "
          f"{'Fmax':>7} {'jflex':>8} {'um/N':>7} {'pads':>5} {'held':>5} {'gresid':>8} {'cc':>10}")
    tbl = []
    for name, rec in summary["rungs"].items():
        b = rec["squeezes"].get("B")
        a = rec["squeezes"].get("A")
        sp = rec["spec"]
        if b is None:
            print(f"  {name:<16} (no contact)")
            continue
        jf = max(b["max_joint_flex_gear"].values()) if b["max_joint_flex_gear"] else float("nan")
        unres = (None if CLOSE_TARGET is None
                 else float(CLOSE_TARGET[0] - b["q_ref_final"]))
        print(f"  {name:<16} {str(sp['nf']):>9} {str(sp['dr']):>6} {str(sp['onf']):>10} "
              f"{str(sp['odr']):>5} {str(sp['me']):>7} {str(sp['spi']):>4} {str(sp['kp']):>9} "
              f"{b['jaw_final_mm']:>10.3f} {b['past_object_width_mm']:>+9.3f} "
              f"{b['gap_flex_mm']:>+8.3f} {(float('nan') if unres is None else unres):>8.4f} "
              f"{b['force_l_N']:>7.2f} {b['force_r_N']:>7.2f} {b['max_force_N']:>7.2f} "
              f"{jf:>8.5f} {b['um_per_N']:>7.1f} {b['n_contact_final']:>5} "
              f"{'-' if a is None else ('YES' if a.get('held') else 'NO'):>5} "
              f"{max(rec['gear_resid'].values()):>8.5f} {str(sp['ccs']):>10}")
        tbl.append((name, b))
    if len(tbl) > 1:
        jaws = np.array([b["jaw_final_mm"] for _, b in tbl])
        forces = np.array([b["max_force_N"] for _, b in tbl])
        spread = float(jaws.max() - jaws.min())
        print(f"\n  jaw-at-stall spread across all rungs = {spread:.4f} mm "
              f"({jaws.min():.3f} .. {jaws.max():.3f}); peak force {forces.min():.1f} .. "
              f"{forces.max():.1f} N")
        # Repeated rung names (default / default#N) bound the noise. Anything under that is nothing.
        base = {}
        for name, b in tbl:
            base.setdefault(name.split("#")[0], []).append(b["jaw_final_mm"])
        reps = {k: v for k, v in base.items() if len(v) > 1}
        if reps:
            noise = max(max(v) - min(v) for v in reps.values())
            print(f"  REPEATABILITY from repeated rungs {list(reps)}: "
                  + "; ".join(f"{k}: {['%.4f' % x for x in v]}" for k, v in reps.items())
                  + f"  -> noise floor {noise:.4f} mm")
            print(f"  VERDICT_SPREAD_VS_NOISE spread={spread:.4f} mm noise={noise:.4f} mm -> "
                  f"{'REAL EFFECT' if spread > 3 * noise else 'INDISTINGUISHABLE FROM NOISE'}")
        if spread < 1e-6:
            print("  IDENTICAL_RUNGS: every rung produced the same jaw gap to the micron. Either the "
                  "knob is inert in this build or the write never reached PhysX -- see the schema "
                  "check in the identity block; do NOT read these rows as a compliance measurement.")

# ---------------------------------------------------------------- outputs
hdr("WRITING VIDEO AND RAW DATA")
os.makedirs(OUT, exist_ok=True)
tags = [r["tag"] for r in rows]
gaps = np.array([jaw(r) for r in rows])       # calibrated jaw gap, used for the video overlay too

np.savez_compressed(
    os.path.join(OUT, f"{PFX}_squeeze.npz"),
    q=np.stack([r["q"] for r in rows]), tag=np.array(tags),
    rung=np.array([r.get("rung", "") for r in rows]),
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
with open(os.path.join(OUT, f"{PFX}_squeeze.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"  wrote {OUT}/{PFX}_squeeze.npz and .json ({len(rows)} steps)")


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
    txt = (f"{PFX}   {tags[i]}   step {i}\n"
           f"jaw gap {gaps[i] * 1000:7.2f} mm   contacts {rows[i]['n_contact']}   "
           f"F {rows[i]['f_l']:.1f}/{rows[i]['f_r']:.1f} N")
    d.rectangle([0, 0, im.shape[1], int(im.shape[0] * 0.13)], fill=(0, 0, 0))
    d.multiline_text((10, 6), txt, fill=(255, 255, 80), font=font)
    return np.asarray(img)


from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # noqa: E402


def write_mp4(path, seq, offset, crop):
    ims = []
    for i, fr in enumerate(seq):
        if crop:
            h, w = fr.shape[:2]
            # The camera is aimed AT the pad midpoint, so it projects to the image centre and a
            # centre crop is exactly the zoom.
            ch, cw = h // 4, w // 4
            fr = fr[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
        ims.append(annotate(np.ascontiguousarray(fr), min(offset + i, len(rows) - 1)))
    ImageSequenceClip(ims, fps=args.fps).write_videofile(path, codec="libx264", audio=False,
                                                        logger=None)
    print(f"  wrote {path}  ({len(ims)} frames @ {args.fps} fps = {len(ims) / args.fps:.1f} s)")


written = []
if args.video:
    if not RUNGS_SPEC:
        for name, seq, crop in (("closeup", frames, False), ("closeup_ZOOM", frames, True),
                                ("wide", frames_wide, False)):
            if not seq:
                continue
            out = os.path.join(OUT, f"{PFX}_squeeze_{name}.mp4")
            write_mp4(out, seq, 0, crop)
            written.append(out)
    # One clip per rung as well, so "the best rung next to the default" is a pair of short mp4s
    # rather than a timestamp inside a two-minute one.
    for rname, f0, f1 in RUNG_SPANS:
        if f1 <= f0 or f1 > len(frames):
            continue
        # Trim to the squeeze phases: the calibration sweep is 60% of a rung and shows nothing.
        idx = [i for i in range(f0, f1) if "squeeze_" in rows[i]["tag"] or "settle_" in rows[i]["tag"]
               or "gravity_" in rows[i]["tag"]]
        for suffix, sel, crop in (("closeup_ZOOM", idx, True), ("closeup", idx, False)):
            if not sel:
                continue
            out = os.path.join(OUT, f"{PFX}_{rname}_{suffix}.mp4")
            ims = [frames[i] for i in sel]
            # write_mp4 annotates by a contiguous offset, so annotate here against the real indices.
            clip = []
            for j, i in enumerate(sel):
                fr = ims[j]
                if crop:
                    h, w = fr.shape[:2]
                    ch, cw = h // 4, w // 4
                    fr = fr[h // 2 - ch: h // 2 + ch, w // 2 - cw: w // 2 + cw]
                clip.append(annotate(np.ascontiguousarray(fr), i))
            ImageSequenceClip(clip, fps=args.fps).write_videofile(out, codec="libx264", audio=False,
                                                                 logger=None)
            print(f"  wrote {out}  ({len(clip)} frames @ {args.fps} fps)")
            written.append(out)
else:
    print("  --video 0: no mp4s (and no frames were retained)")

print("\nMP4S: " + " ".join(written))

# ---------------------------------------------------------------- does the override survive play()?
# simulator.py:1374-1382 re-calls robot.update_controller_mode() on every play() after a stop(), which
# re-applies the CONFIG gains and would wipe a gain poked straight onto a joint. The mimic attributes
# are USD-level and max_effort goes through the articulation view, so both are EXPECTED to survive --
# but "expected" is not "measured", and this runs last so it cannot disturb anything above.
if args.play_cycle_check:
    hdr("PLAY-CYCLE CHECK: stop() / play(), then read every override back")
    before = dict(mimic=mimic_state(), max_effort=jget(robot.joints[DRIVEN_J], "max_effort"),
                  stiffness=jget(robot.joints[DRIVEN_J], "stiffness"),
                  damping=jget(robot.joints[DRIVEN_J], "damping"))
    try:
        og.sim.stop()
        og.sim.play()
        after = dict(mimic=mimic_state(), max_effort=jget(robot.joints[DRIVEN_J], "max_effort"),
                     stiffness=jget(robot.joints[DRIVEN_J], "stiffness"),
                     damping=jget(robot.joints[DRIVEN_J], "damping"))
        for k in ("max_effort", "stiffness", "damping"):
            same = before[k] == after[k]
            print(f"  {DRIVEN_J}.{k:<11} {before[k]} -> {after[k]}   "
                  f"{'SURVIVED' if same else '*** WIPED BY play() ***'}")
        diffs = [k for k in before["mimic"] if before["mimic"][k] != after["mimic"][k]]
        print(f"  mimic attributes: {len(before['mimic'])} read, "
              f"{'ALL SURVIVED' if not diffs else f'CHANGED: {diffs}'}")
        print(f"  PLAY_CYCLE_CHECK_OK survived_max_effort="
              f"{before['max_effort'] == after['max_effort']} survived_mimic={not diffs}")
    except Exception as e:
        print(f"  [warn] play-cycle check failed to run: {e!r}")

print("SQUEEZE_PROBE_OK")
og.shutdown()
