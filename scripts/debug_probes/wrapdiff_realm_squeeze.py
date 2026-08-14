"""Measure the MIMIC CONSTRAINT RESIDUAL under a KNOWN EXTERNAL FORCE on the REALM / OmniGibson side.

This is the apples-to-apples mirror of scripts/debug_probes/wrapdiff_robolab_squeeze.py.

Why it exists: the ~9x compliance gap between the two stacks was measured with DIFFERENT load cases.
RoboLab's residual came from an external force pushing the two pad bodies apart; REALM's came from a
CONTACT squeeze against a pinned object. Contact load is not known a priori -- it is whatever the
drive happens to push with, mediated by penetration and the contact solver -- so "0.24 deg at ~50 N"
on one side and "2.25 deg at ~50 N" on the other are only comparable if the 50 N is the same 50 N.
This probe removes that doubt by applying the SAME load, the SAME way, on the REALM side:

    force of magnitude F along the live pad-separation axis, OUTWARD, one on each inner_finger body,
    re-applied every physics step, at rungs 0 / 5 / 20 / 50 / 100 N.

That is the direction a squeezed object's reaction pushes, needs no geometry, and yields residual
per newton directly -- the quantity to compare.

Residual convention (do not "simplify" this): PhysxMimicJointAPI's documented relation is

    jointPosition + gearing * referenceJointPosition + offset = 0

so the residual is q_f + g*q_ref + o. It is NOT q_f - (g*q_ref + o); that form injects a constant
2*q_ref on every gearing=-1 follower and reads as a fake ~90 deg. Fixed in b426903, same trap here.

POSITIVE CONTROL: a force that silently fails to land reads as "residual ~0", i.e. exactly the
answer we are trying to test for. So the probe tracks pad separation and leader angle at every rung
and prints FORCE_NOT_LANDING if the pads do not respond at all to 100 N. Never accept a null from
this probe without checking that line.

    ./scripts/clara/interactive/rr python -u scripts/debug_probes/wrapdiff_realm_squeeze.py

Isaac exits 139 at teardown; grep for WRAPDIFF_REALM_SQUEEZE_OK, never the exit code.
"""
import argparse
import json
import os

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=220)

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="DROID_robolab_v2")
ap.add_argument("--task-cfg", default="REALM_DROID10/put_green_block_into_bowl/default.yaml")
ap.add_argument("--out", default="/logs/gripper_squeeze/gpudyn_realm_squeeze.json")
ap.add_argument("--tag", default="", help="suffix for the json, so A/B runs do not overwrite")
ap.add_argument("--steps", type=int, default=60, help="physics steps held at each force rung")
ap.add_argument("--open-steps", type=int, default=60)
ap.add_argument("--close-steps", type=int, default=80)
ap.add_argument("--rungs", default="0,5,20,50,100", help="force per pad, N, comma separated")
ap.add_argument("--variant-usd", default=None,
                help="TASK 2: a variant .usda from make_mimic_variant.py, swapped in for the "
                     "shipped droid_robolab_v2.usd BEFORE anything loads. Use with "
                     "--restore-follower-drive to put the followers' DriveAPI back. Needs "
                     "MODE=oglite with robot.py:658's assert relaxed -- see the task-2 notes.")
args = ap.parse_args()

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402

from realm.sim_config import set_sim_config  # noqa: E402
from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

OUT = {}
RUNGS = [float(x) for x in args.rungs.split(",") if x.strip()]


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


print(f"[wrapdiff-realm-squeeze] robot={args.robot} task={args.task_cfg} "
      f"variant_usd={args.variant_usd}", flush=True)
print(f"[wrapdiff-realm-squeeze] REALM_GPU_DYNAMICS={os.environ.get('REALM_GPU_DYNAMICS')!r}",
      flush=True)

if args.variant_usd:
    # Swap the asset BEFORE anything is loaded. Only the robolab v2 path is redirected, so the
    # shipped file is never written to and a stock A/B in the same session is unaffected.
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

set_sim_config(robot=args.robot)
env = RealmEnvironmentDynamic(
    config_path="/app/realm/config", task_cfg_path=args.task_cfg, perturbations=["Default"],
    multi_view=False, no_rendering=True, rendering_mode="rt", robot=args.robot,
)
obs, _ = env.reset()
robot = env.robot
stage = og.sim.stage

hdr("STACK IDENTITY -- what actually ran (never restate the config, read it back)")
ident = dict(
    gm_USE_GPU_DYNAMICS=str(og.gm.USE_GPU_DYNAMICS),
    og_sim_device=str(getattr(og.sim, "device", None)),
    og_physics_dt=str(og.sim.get_physics_dt()),
    env_REALM_GPU_DYNAMICS=str(os.environ.get("REALM_GPU_DYNAMICS")),
    restore_follower_drive=str(args.restore_follower_drive),
)
try:
    pc = og.sim._physics_context
    ident["is_gpu_dynamics_enabled"] = str(pc.is_gpu_dynamics_enabled())
    ident["broadphase"] = str(pc.get_broadphase_type())
    ident["solver_type"] = str(pc.get_solver_type())
except Exception as e:
    ident["physics_context"] = f"<{type(e).__name__}: {e}>"
for k, v in ident.items():
    print(f"  {k:<32} = {v}")
OUT["identity"] = ident

# ---- name the DOFs by asking each joint where its own DOF sits ---------------------------------
q_all = _np(robot.get_joint_positions())
joint_names = [None] * len(q_all)
for n, j in robot.joints.items():
    idxs = list(j.dof_indices)
    if len(idxs) == 1:
        joint_names[idxs[0]] = n
arm = list(robot.arm_joint_names[robot.default_arm])
LEAD = "finger_joint"
FOLLOWERS = [n for n in joint_names
             if n and n not in arm and n not in (LEAD, "rootJoint") and n.endswith("_joint")]
PADS = list(robot.finger_link_names[robot.default_arm])
print(f"\n  leader    = {LEAD}")
print(f"  followers = {FOLLOWERS}")
print(f"  pad bodies= {PADS}")
assert LEAD in joint_names, f"no {LEAD} among {joint_names}"
assert len(PADS) == 2, f"expected two pad links, got {PADS}"

# ---- gearing / offset / referenceJoint straight off the live prims, per follower ---------------
MIMIC = {}
for n in FOLLOWERS:
    prim = robot.joints[n].prim
    for s in [str(x) for x in prim.GetAppliedSchemas()]:
        if "MimicJoint" not in s:
            continue
        inst = s.split(":", 1)[1]
        g = prim.GetAttribute(f"physxMimicJoint:{inst}:gearing")
        o = prim.GetAttribute(f"physxMimicJoint:{inst}:offset")
        nf = prim.GetAttribute(f"physxMimicJoint:{inst}:naturalFrequency")
        dr = prim.GetAttribute(f"physxMimicJoint:{inst}:dampingRatio")
        rel = prim.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
        ref = [str(t).split("/")[-1] for t in rel.GetTargets()] if rel and rel.IsValid() else []
        MIMIC[n] = dict(inst=inst,
                        gearing=float(g.Get()) if g and g.IsValid() else None,
                        offset=float(o.Get()) if o and o.IsValid() else None,
                        naturalFrequency=float(nf.Get()) if nf and nf.IsValid() else None,
                        dampingRatio=float(dr.Get()) if dr and dr.IsValid() else None,
                        referenceJoint=ref[0] if ref else None)
print("\n  live mimic coupling (read by literal token name):")
for n, m in MIMIC.items():
    print(f"    {n:<34} inst={m['inst']} gearing={m['gearing']} offset={m['offset']} "
          f"nf={m['naturalFrequency']} dr={m['dampingRatio']} ref={m['referenceJoint']}")
OUT["mimic"] = MIMIC
assert MIMIC, "no mimic joints found -- the residual below would be meaningless"

# ---- the four INNER mimic joints, and their live drive block ------------------------------------
OUTER = "right_outer_knuckle_joint"
INNER = [n for n in MIMIC if n != OUTER]
drives = {}
for n in MIMIC:
    prim = robot.joints[n].prim
    d = {"schemas_drive": [s for s in [str(x) for x in prim.GetAppliedSchemas()] if "Drive" in s]}
    for a in prim.GetAttributes():
        an = a.GetName()
        if an.startswith("drive:"):
            try:
                d[an] = str(a.Get())
            except Exception:
                pass
    d["max_effort_runtime"] = str(getattr(robot.joints[n], "max_effort", None))
    drives[n] = d
print("\n  follower DriveAPI as the simulation has it (TASK 2 baseline):")
for n, d in drives.items():
    print(f"    {n:<34} drive_schemas={d['schemas_drive']} max_effort={d['max_effort_runtime']}")
OUT["follower_drives"] = drives


def q_of(name):
    return float(_np(robot.get_joint_positions())[joint_names.index(name)])


def q_all_now():
    return _np(robot.get_joint_positions())


def residuals(q=None):
    """Violation of the mimic constraint, per follower. 0 == constraint fully enforced.

    Relation is q_f + gearing*q_ref + offset = 0 (see module docstring on the sign trap).
    """
    if q is None:
        q = q_all_now()
    out = {}
    for n, m in MIMIC.items():
        if m["referenceJoint"] is None or m["referenceJoint"] not in joint_names:
            continue
        out[n] = float(q[joint_names.index(n)]
                       + m["gearing"] * q[joint_names.index(m["referenceJoint"])]
                       + m["offset"])
    return out


def link_world(ln):
    p, quat = robot.links[ln].get_position_orientation()
    return _np(p), _np(quat)


def com_world(ln):
    """World position of the link's centre of mass.

    Isaac Lab's set_external_force_and_torque applies the wrench at the body CoM, so the mirror
    must too -- applying at the body ORIGIN instead adds a spurious r x F torque that would show up
    as extra (or less) mimic residual and corrupt the comparison.
    """
    p, quat = link_world(ln)
    try:
        com_local = np.asarray(_np(robot.links[ln].center_of_mass[0]), dtype=np.float64).reshape(3)
    except Exception:
        com_local = np.zeros(3)
    x, y, z, w = quat  # omnigibson quaternions are xyzw
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return p + R @ com_local


def pad_gap():
    a, _ = link_world(PADS[0])
    b, _ = link_world(PADS[1])
    return float(np.linalg.norm(a - b))


# ---- force application -------------------------------------------------------------------------
# Two ways to put an external force on a link, and the choice is NOT cosmetic:
#
#   view  -- link._rigid_prim_view.apply_forces_and_torques_at_pos(...). This is the PhysX TENSOR
#            API, the same one Isaac Lab's set_external_force_and_torque uses, and it is the only
#            one that works when the GPU pipeline is on. Note the accessor lives on
#            RigidDynamicPrim (omnigibson/prims/rigid_dynamic_prim.py:32), NOT on RigidPrim -- the
#            name does not appear in rigid_prim.py at all, which is a good way to conclude wrongly
#            that OmniGibson exposes no force API.
#   psi   -- omnigibson.utils.physx_utils.apply_force_at_pos, PhysX's IMMEDIATE-mode call. Works on
#            the CPU pipeline and is a documented no-op under GPU dynamics, where it would silently
#            apply nothing and hand back a residual of ~0 -- i.e. a fake "the constraint is rigid".
#
# So: prefer the view, fall back to psi, and print which one ran. Either way the force lands on the
# body for the NEXT physics step and is then cleared, so it must be re-issued every step -- the same
# lifetime as Isaac Lab's set_external_force_and_torque + write_data_to_sim loop.
import torch as th  # noqa: E402
from omnigibson.utils.physx_utils import apply_force_at_pos  # noqa: E402

DEV = getattr(og.sim, "device", None) or "cpu"
_views = {ln: getattr(robot.links[ln], "_rigid_prim_view", None) for ln in PADS}
FORCE_PATH = ("view" if all(v is not None and hasattr(v, "apply_forces_and_torques_at_pos")
                            for v in _views.values()) else "psi")
print(f"\n  force path = {FORCE_PATH}   (og.sim.device={DEV})")
if FORCE_PATH == "psi" and str(og.gm.USE_GPU_DYNAMICS) == "True":
    print("  *** WARNING: falling back to psi.apply_force_at_pos WITH GPU DYNAMICS ON. That call is\n"
          "  a no-op on the GPU pipeline; expect FORCE_NOT_LANDING below and do not report a ratio.")


def _push_view(ln, vec):
    v = _views[ln]
    f = th.as_tensor(np.asarray(vec, dtype=np.float32).reshape(1, 3), device=DEV)
    t = th.zeros((1, 3), dtype=f.dtype, device=DEV)
    p = th.as_tensor(np.asarray(com_world(ln), dtype=np.float32).reshape(1, 3), device=DEV)
    v.apply_forces_and_torques_at_pos(forces=f, torques=t, positions=p, is_global=True)


def push_pads(F, axis):
    if F == 0.0:
        return
    if FORCE_PATH == "view":
        _push_view(PADS[0], -axis * F)
        _push_view(PADS[1], +axis * F)
    else:
        apply_force_at_pos(robot.links[PADS[0]].prim, -axis * F, com_world(PADS[0]))
        apply_force_at_pos(robot.links[PADS[1]].prim, +axis * F, com_world(PADS[1]))


def hold(n_steps, target, F=0.0, axis=None):
    """Hold the leader drive target for n_steps PHYSICS steps, re-issuing the force each one.

    og.sim.step() is the WRONG stepper here: it runs n_physics_timesteps_per_render (8, at REALM's
    120 Hz / decimation 8) physics substeps per call, while psi.apply_force_at_pos lands on exactly
    ONE substep and is then cleared. Driving the sweep with og.sim.step() would apply the load on 1
    substep in 8 and read back a residual ~8x too small -- which is the same order as the effect
    being measured, so it would have looked like a result. og.sim.step_physics() advances a single
    physics step, matching Isaac Lab's sim.step(render=False) on the RoboLab side one-for-one.
    """
    lj = robot.joints[LEAD]
    for _ in range(n_steps):
        lj.set_pos(target, drive=True)
        if F:
            push_pads(F, axis)
        og.sim.step_physics()


# ---- open the jaws ------------------------------------------------------------------------------
hdr("OPENING THE JAWS")
lo, hi = float(robot.joints[LEAD].lower_limit), float(robot.joints[LEAD].upper_limit)
print(f"  {LEAD} limits = [{lo:+.6f}, {hi:+.6f}] rad")
hold(args.open_steps, lo)
print(f"  q_lead={q_of(LEAD):+.6f}  pad-body separation={pad_gap() * 1000:.2f} mm")

# ---- close the jaws onto nothing, then load the pads with a KNOWN external force ----------------
hdr("CLOSING THE JAWS, THEN PUSHING THE PADS APART WITH A KNOWN EXTERNAL FORCE")
CLOSE = min(float(np.pi / 4), hi)   # RoboLab commands pi/4; clamp to this asset's limit
hold(args.close_steps, CLOSE)
print(f"  jaws shut: q_lead={q_of(LEAD):+.6f} (commanded {CLOSE:+.6f})  "
      f"pad-body separation={pad_gap() * 1000:.3f} mm")
base = residuals()
print(f"  baseline max|residual| with NO external load = "
      f"{max(abs(v) for v in base.values()):.6f} rad = "
      f"{np.degrees(max(abs(v) for v in base.values())):.4f} deg")

a0, _ = link_world(PADS[0])
b0, _ = link_world(PADS[1])
axis = b0 - a0
axis = axis / np.linalg.norm(axis)
print(f"  load axis (pad0 -> pad1, outward push) = {axis}")

traj = []
for F in RUNGS:
    hold(args.steps, CLOSE, F=F, axis=axis)
    r = residuals()
    mx = max(abs(v) for v in r.values())
    traj.append(dict(force_N=F, q_lead=q_of(LEAD), gap=pad_gap(),
                     q={n: q_of(n) for n in FOLLOWERS}, res=r, max_abs_res=mx))
    print(f"  F={F:6.1f} N per pad (outward)  q_lead={q_of(LEAD):+.6f}  "
          f"pad_sep={pad_gap() * 1000:8.3f} mm  max|residual|={mx:.6f} rad = "
          f"{np.degrees(mx):7.3f} deg", flush=True)

# ---- POSITIVE CONTROL --------------------------------------------------------------------------
hdr("POSITIVE CONTROL -- did the external force actually land on the bodies?")
gaps = [s["gap"] for s in traj]
spread_mm = (max(gaps) - min(gaps)) * 1000.0
lead_spread = max(s["q_lead"] for s in traj) - min(s["q_lead"] for s in traj)
print(f"  pad separation across the sweep: min={min(gaps) * 1000:.4f} mm  max={max(gaps) * 1000:.4f}"
      f" mm  spread={spread_mm:.4f} mm")
print(f"  q_lead spread across the sweep : {lead_spread:.6f} rad = {np.degrees(lead_spread):.4f} deg")
LANDED = spread_mm > 1e-3 or lead_spread > 1e-5
if LANDED:
    print("  FORCE_LANDED -- the articulation responded to the load, so a small residual is a real\n"
          "  measurement of constraint stiffness and not a silently dropped force.")
else:
    print("  *** FORCE_NOT_LANDING *** the pads did not move at all between 0 N and "
          f"{max(RUNGS):.0f} N.\n"
          "  Every residual below is then meaningless -- psi.apply_force_at_pos did not reach these\n"
          "  bodies (expected under GPU dynamics, where the immediate-mode API is a no-op). Do NOT\n"
          "  report a ratio from this run.")
OUT["force_landed"] = bool(LANDED)
OUT["force_path"] = FORCE_PATH
OUT["control"] = dict(pad_sep_spread_mm=spread_mm, q_lead_spread_rad=float(lead_spread))

# ---- result ------------------------------------------------------------------------------------
last = traj[-1]
hdr("RESULT -- REALM residual under EXTERNAL FORCE, directly comparable to the RoboLab sweep")
print(f"  {'follower':<34} {'q':>12} {'-(g*q_ref+o)':>14} {'RESIDUAL':>12}   {'deg':>9}   "
      f"(at {last['force_N']:.0f} N per pad)")
for n, m in MIMIC.items():
    if n not in last["res"]:
        continue
    ref_q = last["q_lead"] if m["referenceJoint"] == LEAD else last["q"].get(m["referenceJoint"])
    want = None if ref_q is None else -(m["gearing"] * ref_q + m["offset"])
    print(f"  {n:<34} {last['q'][n]:+12.6f} "
          f"{'      n/a' if want is None else f'{want:+14.6f}'} {last['res'][n]:+12.6f} "
          f"{np.degrees(last['res'][n]):+9.3f}")

print(f"\n  {'F per pad (N)':>14} {'max|residual| rad':>20} {'deg':>10} {'pad_sep mm':>12}")
for s in traj:
    print(f"  {s['force_N']:>14.1f} {s['max_abs_res']:>20.6f} "
          f"{np.degrees(s['max_abs_res']):>10.3f} {s['gap'] * 1000:>12.3f}")
mx = max(s["max_abs_res"] for s in traj)
print(f"\n  max |mimic residual| over the whole sweep = {mx:.6f} rad = {np.degrees(mx):.3f} deg")

# RoboLab's numbers from wrapdiff_robolab_squeeze.py, same rungs, same load case.
ROBOLAB = {5.0: 0.328, 50.0: 2.250}
print("\n  APPLES-TO-APPLES against RoboLab (both sides now loaded by EXTERNAL FORCE):")
print(f"  {'F per pad (N)':>14} {'RoboLab deg':>13} {'REALM deg':>12} {'ratio':>9}")
ratios = {}
for s in traj:
    if s["force_N"] not in ROBOLAB:
        continue
    rl = ROBOLAB[s["force_N"]]
    rm = float(np.degrees(s["max_abs_res"]))
    ratio = rl / rm if rm > 0 else float("inf")
    ratios[s["force_N"]] = ratio
    print(f"  {s['force_N']:>14.1f} {rl:>13.3f} {rm:>12.4f} {ratio:>9.2f}x")
if ratios:
    print(f"\n  MATCHED-LOAD RATIO (RoboLab / REALM) = "
          + "  ".join(f"{k:.0f}N: {v:.2f}x" for k, v in sorted(ratios.items())))
    print("  The provisional ~9x came from comparing RoboLab-under-external-force against\n"
          "  REALM-under-contact. THIS line is the corrected number: same load case both sides.")
OUT["ratios_vs_robolab"] = {str(k): v for k, v in ratios.items()}
OUT["result"] = dict(residual_final=last["res"], max_abs_residual_rad=mx,
                     max_abs_residual_deg=float(np.degrees(mx)),
                     q_final=last["q"], q_lead_cmd=CLOSE, baseline_res=base)
OUT["sweep"] = traj

out = args.out
if args.tag:
    root, ext = os.path.splitext(out)
    out = f"{root}_{args.tag}{ext}"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True, default=str)
print(f"\nwrote {out}")
print("WRAPDIFF_REALM_SQUEEZE_OK")
