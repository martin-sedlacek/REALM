"""Build a "curl grip" variant of the robolab Robotiq 2F-85 asset: a softer mimic constraint.

Why
---
The real 2F-85 is underactuated. When an OPEN fingertip is loaded, the four-bar lets the tip rotate
INWARD, toward the closing axis. In `droid_robolab_v2.usd` the four inner linkage joints are PhysX
**mimic** joints slaved to `finger_joint` at `naturalFrequency 1000 / dampingRatio 0.05`. A mimic
constraint at that frequency is a near-hard equality, so under a concentrated tip load the pad
rotates by 0.048 deg -- sub-visible. Lowering `naturalFrequency` softens exactly that constraint and
nothing else.

This is the SIBLING of scripts/make_padspring_gripper_usd.py, which attacks the same behaviour by
removing the mimic from the two pad pivots and giving them a real drive. Two routes:

  * pad spring -- the pad pivots become sprung followers. Needs a custom controller
    (PadSpringGripperController) because a driven DOF must be claimed by one.
  * mimic softening (THIS FILE) -- the linkage stays closed and mimic-coupled; only the constraint's
    stiffness changes. No new controller, no new DOFs in the gripper group, no change to
    `is_grasping`'s view of the finger links. Strictly less invasive.

What this changes
-----------------
On the four INNER mimic joints only (`{left,right}_inner_finger_joint` and
`{left,right}_inner_finger_knuckle_joint`):

  * `physxMimicJoint:rotX:naturalFrequency` -> --nf
  * `physxMimicJoint:rotX:dampingRatio`     -> --dr

`right_outer_knuckle_joint` (the left/right symmetry constraint, `rotZ`, naturalFrequency 1e6) is
left alone on purpose: it is what keeps the two jaws mirrored, and softening it makes the jaw
asymmetric rather than compliant.

Optionally, with --leader-max-force, `finger_joint`'s own `drive:angular:physics:maxForce`. That is
a different mechanism and it is off by default: the leader is the ONE driven gripper DOF, and at the
authored 16.5 N.m it cannot be back-driven by a press (see the --leader-max-force help).

Two traps this file exists to get right, both measured:

  * the mimic INSTANCE TOKEN is not the joint's `physics:axis`. All six joints author axis Z, yet
    the four inner ones use `rotX` and `right_outer_knuckle_joint` uses `rotZ`. Discovered from the
    applied schemas, never guessed.
  * `naturalFrequency` / `dampingRatio` are NOT in the `PhysxMimicJointAPI` schema of Isaac Sim
    5.1.0. `omni.physx` reads them by literal token as CUSTOM attributes. They are authored in the
    source asset, so they can be Set here; if they ever are not, they are created as `custom float`.

ARM PHYSICS. Nothing here touches the arm, and that is verified rather than asserted: the final pass
reopens the written file and compares every authored attribute on `panda_joint1..7` -- and on every
`panda_link*` prim -- against the source, byte for byte, and refuses to pass if any differ. The
top-level `friction`/`armature` arrays and the `arm_0` controller block live in the robot YAML, not
in the USD, and `env_dynamic.update_robot_physics()` writes only `physxJoint:jointFriction` and
`physxJoint:armature` on `panda_link{idx}/{arm_joint_names[idx]}` `for idx in range(7)` -- arm only.

Usage (inside the container -- pxr only exists once a Kit app is up):
    python /app/scripts/make_curlgrip_gripper_usd.py --nf 200
    python /app/scripts/make_curlgrip_gripper_usd.py --nf 200 --dst /app/.../foo.usd
"""

import argparse
import os
import shutil

import omnigibson as og
import omnigibson.lazy as lazy  # noqa: E402

DEFAULT_SRC = "/app/realm/robots/panda_robotiq/droid_robolab_v2.usd"
DEFAULT_DST = "/app/realm/robots/panda_robotiq/droid_robolab_curlgrip.usd"

# The four joints of the closed four-bar, on both sides. NOT right_outer_knuckle_joint.
INNER_MIMIC = ("left_inner_finger_joint", "right_inner_finger_joint",
               "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint")
LEADER = "finger_joint"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=DEFAULT_SRC)
ap.add_argument("--dst", default=DEFAULT_DST)
ap.add_argument("--nf", type=float, default=200.0,
                help="physxMimicJoint:<inst>:naturalFrequency on the four inner mimic joints. "
                     "Authored 1000. Lower = softer constraint = more tip rotation under load. "
                     "200 is the shipped default: it is the softest rung that still passed every "
                     "grasp gate, and 100 changes the force profile substantially more for about "
                     "twice the curl.")
ap.add_argument("--dr", type=float, default=0.05,
                help="physxMimicJoint:<inst>:dampingRatio on the same four. Authored 0.05. A "
                     "dampingRatio sweep is a settled negative -- it does not move the curl -- so "
                     "this defaults to the authored value and should stay there.")
ap.add_argument("--leader-max-force", type=float, default=None,
                help="drive:angular:physics:maxForce (N.m) on finger_joint. Authored 16.5. OFF by "
                     "default. This is the LEADER's holding torque: OmniGibson drives finger_joint "
                     "at isaac_kp 1e7, so the drive saturates immediately and behaves as a constant "
                     "torque source resisting back-drive. Against a ~0.1 m tip lever arm 16.5 N.m "
                     "is ~165 N at the tip, while a full-effort arm press stalls at ~80 N -- so the "
                     "press cannot fold the four-bar and every curl measured so far is follower "
                     "deviation around a rigid leader. Lowering this is a DIFFERENT mechanism from "
                     "--nf and it trades directly against grip strength, so it is opt-in and has to "
                     "be re-gated on tasks 0 and 4 whenever it is used.")
args = ap.parse_args()


def find_joints(stage):
    out = {}
    for prim in lazy.pxr.Usd.PrimRange(stage.GetDefaultPrim()):
        if "Joint" in prim.GetTypeName():
            out.setdefault(prim.GetName(), prim)
    return out


def mimic_insts(prim):
    """PhysxMimicJointAPI instance tokens on @prim. Discovered, because the instance is NOT
    physics:axis: these joints all author axis Z and four of them use rotX."""
    return [s.split(":", 1)[1] for s in prim.GetAppliedSchemas()
            if s.startswith("PhysxMimicJointAPI:")]


def set_custom_float(prim, name, value):
    at = prim.GetAttribute(name)
    if not at.IsValid():
        at = prim.CreateAttribute(name, lazy.pxr.Sdf.ValueTypeNames.Float, custom=True)
    before = at.Get()
    at.Set(float(value))
    return before, at.Get()


def attr_snapshot(prim):
    """Every AUTHORED attribute on @prim, as {name: value}. Used for the arm byte-comparison."""
    out = {}
    for at in prim.GetAttributes():
        if at.HasAuthoredValue():
            v = at.Get()
            out[at.GetName()] = str(v)
    return out


def main():
    Usd = lazy.pxr.Usd
    assert os.path.exists(args.src), f"no source asset at {args.src}"
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("refusing to edit the shipped asset in place -- pass a different --dst")
    shutil.copyfile(args.src, args.dst)
    print(f"copied {args.src}\n    -> {args.dst}")

    stage = Usd.Stage.Open(args.dst)
    joints = find_joints(stage)
    missing = [n for n in INNER_MIMIC if n not in joints]
    assert not missing, f"{missing} not found in {args.src}"

    for name in INNER_MIMIC:
        prim = joints[name]
        insts = mimic_insts(prim)
        assert insts, f"{name} has no PhysxMimicJointAPI -- is this the right source asset?"
        print(f"\n[{name}] {prim.GetPath()}  axis={prim.GetAttribute('physics:axis').Get()}  "
              f"mimic instances={insts}")
        for inst in insts:
            for attr, val in (("naturalFrequency", args.nf), ("dampingRatio", args.dr)):
                b, a = set_custom_float(prim, f"physxMimicJoint:{inst}:{attr}", val)
                print(f"    physxMimicJoint:{inst}:{attr}  {b} -> {a}")

    if args.leader_max_force is not None:
        prim = joints[LEADER]
        at = prim.GetAttribute("drive:angular:physics:maxForce")
        assert at.IsValid(), f"{LEADER} has no drive:angular:physics:maxForce"
        b = at.Get()
        at.Set(float(args.leader_max_force))
        print(f"\n[{LEADER}] drive:angular:physics:maxForce  {b} -> {at.Get()}")

    stage.Save()

    # --- verify by reopening --------------------------------------------------------------------
    print("\nVERIFY (reopened from disk):")
    src_stage = Usd.Stage.Open(args.src)
    dst_stage = Usd.Stage.Open(args.dst)
    src_j, dst_j = find_joints(src_stage), find_joints(dst_stage)
    ok = True
    for name in list(INNER_MIMIC) + [LEADER, "right_outer_knuckle_joint"]:
        prim = dst_j[name]
        vals = {}
        for inst in mimic_insts(prim):
            for attr in ("naturalFrequency", "dampingRatio", "gearing", "offset"):
                at = prim.GetAttribute(f"physxMimicJoint:{inst}:{attr}")
                vals[f"{inst}:{attr}"] = at.Get() if at.IsValid() else None
        mf = prim.GetAttribute("drive:angular:physics:maxForce")
        print(f"  {name:<34} {vals}  maxForce={mf.Get() if mf.IsValid() else None}")
        if name in INNER_MIMIC:
            got_nf = vals.get("rotX:naturalFrequency")
            got_dr = vals.get("rotX:dampingRatio")
            ok &= (got_nf is not None and abs(got_nf - args.nf) < 1e-4 * max(1.0, args.nf))
            ok &= (got_dr is not None and abs(got_dr - args.dr) < 1e-6)
        if name == "right_outer_knuckle_joint":
            # the symmetry constraint must be UNTOUCHED
            s = src_j[name]
            for inst in mimic_insts(prim):
                for attr in ("naturalFrequency", "dampingRatio", "gearing", "offset"):
                    k = f"physxMimicJoint:{inst}:{attr}"
                    sa, da = s.GetAttribute(k), prim.GetAttribute(k)
                    same = (sa.Get() if sa.IsValid() else None) == (da.Get() if da.IsValid() else None)
                    if not same:
                        print(f"    *** {name} {k} CHANGED -- it must not be ***")
                    ok &= same

    # --- THE ARM MUST BE BYTE-IDENTICAL ---------------------------------------------------------
    # Not assumed: every authored attribute on the seven arm joints and on every panda_link* prim is
    # compared against the source. env_dynamic.update_robot_physics() writes jointFriction/armature
    # on exactly these seven at runtime, so anything the asset changes here would compound with it.
    print("\nARM PHYSICS -- comparing every authored attribute against the source:")
    arm_ok, n_attr = True, 0
    for name in ARM_JOINTS:
        assert name in src_j and name in dst_j, f"{name} missing from one of the stages"
        a, b = attr_snapshot(src_j[name]), attr_snapshot(dst_j[name])
        n_attr += len(a)
        if a != b:
            arm_ok = False
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    print(f"  *** {name}.{k}: src={a.get(k)!r}  dst={b.get(k)!r}")
        print(f"  {name:<16} {len(a):>3} authored attributes  "
              f"{'IDENTICAL' if a == b else '*** DIFFERS ***'}")
    n_links = 0
    for sp in Usd.PrimRange(src_stage.GetDefaultPrim()):
        nm = sp.GetName()
        if not nm.startswith("panda_link"):
            continue
        dp = dst_stage.GetPrimAtPath(sp.GetPath())
        n_links += 1
        if not dp.IsValid() or attr_snapshot(sp) != attr_snapshot(dp):
            arm_ok = False
            print(f"  *** {sp.GetPath()} DIFFERS")
    print(f"  {n_links} panda_link* prims compared: "
          f"{'all identical' if arm_ok else '*** SOME DIFFER ***'}")
    print(f"  {n_attr} arm-joint attributes compared in total")
    ok &= arm_ok

    print(f"\n  nf={args.nf}  dr={args.dr}  leader_max_force={args.leader_max_force}")
    print(f"CURLGRIP_ARM_{'IDENTICAL' if arm_ok else 'CHANGED'}")
    print(f"CURLGRIP_USD_{'OK' if ok else 'FAIL'} {args.dst}")


if __name__ == "__main__":
    main()
    og.shutdown()
