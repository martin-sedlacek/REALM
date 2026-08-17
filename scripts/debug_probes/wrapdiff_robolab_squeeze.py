"""Measure the MIMIC CONSTRAINT RESIDUAL under load on the RoboLab / Isaac Lab side.

This is the discriminator between "the wrapper differs" and "the engine differs".

REALM, at the authored naturalFrequency=1000, holds the mimic relation
`q_follower = gearing * q_leader + offset` to <= 0.0007 rad (0.04 deg) under load -- i.e. the
constraint is effectively HARD, which is exactly what forbids a pad from rotating in response to
contact. Every wrapper setting that could be transplanted has now been transplanted without moving
that residual. So the question is whether the SAME authored constraint is softer in RoboLab's build:

  * if RoboLab's residual under a comparable load is LARGE, the constraint is genuinely softer there
    and the cause is the engine / solver pipeline, not the wrapper -- the wrapper diff is exhausted;
  * if RoboLab's residual is ALSO ~0, then RoboLab's finger curling does NOT come from mimic
    softness at all, and the whole "make the mimic constraint softer" line is aimed at the wrong
    mechanism.

Either answer redirects the investigation, which is why this is worth one run.

Load case: an IMMOVABLE static collider (no RigidBodyAPI, so PhysX cannot move it) placed at the
midpoint of the two inner-finger bodies, then the gripper commanded shut against it. Same geometry
as scripts/debug_probes/gripper_squeeze_compliance.py's pinned squeeze, so the residuals compare.

    scripts/debug_probes/wrapdiff_in_isaaclab.sh /realm/scripts/debug_probes/wrapdiff_robolab_squeeze.py

Isaac exits non-zero at teardown; grep for WRAPDIFF_ROBOLAB_SQUEEZE_OK, never the exit code.
"""
import argparse
import json
import os

import cv2  # noqa: F401  must be imported before isaaclab

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/logs/gripper_squeeze/wrapdiff_robolab_squeeze.json")
parser.add_argument("--cube-mm", type=float, default=30.0)
parser.add_argument("--steps", type=int, default=60, help="physics steps held at each force rung")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402,F401 -- loads Isaac extensions on import
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from robolab.core.environments.base import RobolabDefaultEnvCfg  # noqa: E402
from robolab.robots.droid import DroidCfg  # noqa: E402

OUT = {}


def np_(x):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        try:
            import warp as wp
            x = wp.to_torch(x)
        except Exception:
            return None
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}", flush=True)


env_cfg = RobolabDefaultEnvCfg()
sim_cfg = env_cfg.sim
sim_cfg.device = args_cli.device or "cuda:0"
sim = SimulationContext(sim_cfg)

robot = Articulation(DroidCfg().robot.replace(prim_path="/World/robot"))
sim.reset()

# ---- which dynamics pipeline is this actually running? -----------------------------------------
# `--device cpu` is the whole point of the GPU-vs-CPU comparison: OmniGibson runs CPU dynamics + MBP
# broadphase and RoboLab runs enableGPUDynamics=True + GPU broadphase, and that is the last large
# surviving difference between the two stacks. Turning GPU dynamics ON in OmniGibson is a ~30-site
# device port; turning it OFF here is one flag -- so the hypothesis is cheaper to test from this
# side, and it is the SAME hypothesis. Read it back rather than trusting the flag: a run that
# silently stayed on the GPU would look like a clean negative.
pc = sim.get_physics_context()
PIPE = {}
for _name in ("is_gpu_dynamics_enabled", "get_broadphase_type", "get_solver_type",
              "get_physics_dt", "is_fabric_enabled"):
    _fn = getattr(pc, _name, None)
    if _fn is not None:
        try:
            PIPE[_name] = str(_fn())
        except Exception as _e:                      # noqa: BLE001
            PIPE[_name] = f"<{type(_e).__name__}>"
PIPE["sim_cfg_device"] = str(sim_cfg.device)
OUT["pipeline"] = PIPE
print(f"\n  PIPELINE {json.dumps(PIPE)}", flush=True)

names = list(robot.data.joint_names)
bodies = list(robot.data.body_names)
LEAD = "finger_joint"
FOLLOWERS = [n for n in names if n.endswith("_joint") and not n.startswith("panda_") and n != LEAD]
PADS = [b for b in bodies if b.endswith("inner_finger")]
print(f"  leader    = {LEAD}")
print(f"  followers = {FOLLOWERS}")
print(f"  pad bodies= {PADS}")
assert len(PADS) == 2, f"expected two inner_finger bodies, got {PADS}"

# ---- gearing / offset / referenceJoint straight off the live prims, per follower --------------
from pxr import PhysxSchema, UsdPhysics  # noqa: E402,F401 -- PhysxSchema import REGISTERS USD schemas
import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
MIMIC = {}
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith("/World/robot") or not prim.IsA(UsdPhysics.Joint):
        continue
    nm = p.split("/")[-1]
    if nm not in FOLLOWERS:
        continue
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
        MIMIC[nm] = dict(inst=inst,
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


def q_of(name):
    return float(np_(robot.data.joint_pos)[0, names.index(name)])


def residuals():
    """Violation of the mimic constraint, per follower. 0 == constraint fully enforced.

    PhysxMimicJointAPI's documented relation is

        jointPosition + (gearing * referenceJointPosition) + offset = 0

    so the residual is q_f + g*q_ref + o. It is NOT q_f - (g*q_ref + o): that form injects a constant
    2*q_ref into every follower whose gearing is -1 (two of the five here), which on the first run of
    this probe showed up as a fake "90 deg residual" that was really just 2 * the pi/4 close command.
    """
    out = {}
    for n, m in MIMIC.items():
        if m["referenceJoint"] is None or m["referenceJoint"] not in names:
            continue
        out[n] = q_of(n) + m["gearing"] * q_of(m["referenceJoint"]) + m["offset"]
    return out


def pad_gap():
    bp = np_(robot.data.body_pos_w)
    return float(np.linalg.norm(bp[0, bodies.index(PADS[0])] - bp[0, bodies.index(PADS[1])]))


# ---- open the jaws, then place an immovable obstacle between the pads -------------------------
hdr("OPENING THE JAWS")
tgt = np_(robot.data.joint_pos).copy()
lead_i = names.index(LEAD)
tgt[0, lead_i] = 0.0
for _ in range(60):
    robot.set_joint_position_target(torch.tensor(tgt, dtype=torch.float32, device=sim.device))
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim_cfg.dt)
print(f"  q_lead={q_of(LEAD):+.6f}  pad-body separation={pad_gap() * 1000:.2f} mm")

# ---- load the pads with a KNOWN external force, instead of a contact -------------------------
# A contact squeeze would need an obstacle placed exactly between the pads, and any placement error
# silently turns into "no load" -- an uninformative null. Pushing the two pad bodies apart with a
# known force is the same load DIRECTION as a squeezed object's reaction, needs no geometry, and
# yields the residual per newton directly, which is the quantity to compare. Force is applied along
# the live pad-separation axis, outward, i.e. exactly how an object between the jaws would push back.
hdr("CLOSING THE JAWS, THEN PUSHING THE PADS APART WITH A KNOWN EXTERNAL FORCE")
tgt = np_(robot.data.joint_pos).copy()
tgt[0, lead_i] = float(np.pi / 4)          # RoboLab's own close command
for _ in range(80):
    robot.set_joint_position_target(torch.tensor(tgt, dtype=torch.float32, device=sim.device))
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim_cfg.dt)
print(f"  jaws shut: q_lead={q_of(LEAD):+.6f} (commanded {np.pi / 4:+.6f})  "
      f"pad-body separation={pad_gap() * 1000:.3f} mm")
print(f"  baseline max|residual| with NO external load = "
      f"{max(abs(v) for v in residuals().values()):.6f} rad")

bp = np_(robot.data.body_pos_w)
il, ir = bodies.index(PADS[0]), bodies.index(PADS[1])
axis = bp[0, ir] - bp[0, il]
axis /= np.linalg.norm(axis)
pad_idx = [il, ir]
traj = []
for F in (0.0, 5.0, 20.0, 50.0, 100.0):
    forces = torch.zeros((1, len(pad_idx), 3), dtype=torch.float32, device=sim.device)
    forces[0, 0] = torch.tensor(-axis * F, dtype=torch.float32, device=sim.device)
    forces[0, 1] = torch.tensor(+axis * F, dtype=torch.float32, device=sim.device)
    torques = torch.zeros_like(forces)
    robot.set_external_force_and_torque(forces, torques, body_ids=pad_idx)
    for _ in range(args_cli.steps):
        robot.set_joint_position_target(torch.tensor(tgt, dtype=torch.float32, device=sim.device))
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_cfg.dt)
    r = residuals()
    mx = max(abs(v) for v in r.values())
    traj.append(dict(force_N=F, q_lead=q_of(LEAD), gap=pad_gap(),
                     q={n: q_of(n) for n in FOLLOWERS}, res=r, max_abs_res=mx))
    print(f"  F={F:6.1f} N per pad (outward)  q_lead={q_of(LEAD):+.6f}  "
          f"pad_sep={pad_gap() * 1000:8.3f} mm  max|residual|={mx:.6f} rad = "
          f"{np.degrees(mx):7.3f} deg", flush=True)

last = traj[-1]
hdr("RESULT -- compare against REALM's <= 0.000743 rad at the SAME authored nf=1000")
print(f"  {'follower':<34} {'q':>12} {'-(g*q_ref+o)':>14} {'RESIDUAL':>12}   {'deg':>9}   "
      f"(at {last['force_N']:.0f} N per pad)")
for n, m in MIMIC.items():
    if n not in last["res"]:
        continue
    ref_q = last["q_lead"] if m["referenceJoint"] == LEAD else last["q"].get(m["referenceJoint"])
    # what q_follower WOULD be if the constraint were exactly satisfied
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
print("\n  REALM at authored nf=1000 measured <= 0.000743 rad (0.043 deg) at ~4 N of pad contact")
print("  (wrapdiff_robolabcfg.log). If RoboLab's residual is the same order at comparable force,")
print("  the constraint is equally HARD in both builds -- mimic softness is then NOT what makes")
print("  RoboLab's fingers curl, and the search should move off the mimic constraint entirely.")
OUT["result"] = dict(residual_final=last["res"], max_abs_residual_rad=mx,
                     max_abs_residual_deg=float(np.degrees(mx)),
                     q_final=last["q"], q_lead_cmd=float(np.pi / 4))
OUT["sweep"] = traj

os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
with open(args_cli.out, "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True, default=str)
print(f"\nwrote {args_cli.out}")
print("WRAPDIFF_ROBOLAB_SQUEEZE_OK")
simulation_app.close()
