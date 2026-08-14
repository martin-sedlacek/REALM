"""Build a "pad spring" variant of the robolab Robotiq 2F-85 asset.

Why
---
The real 2F-85 carries a **spring at the inner-finger pivot**. That spring is what keeps the pad
face parallel through the stroke *and* what lets the pad visibly rotate when it is loaded. In
`droid_robolab_v2.usd` that same pivot -- `left_inner_finger_joint` / `right_inner_finger_joint`,
each `outer_finger -> inner_finger` -- is instead a PhysX **mimic** joint rigidly slaved to
`finger_joint` at `naturalFrequency 1000 / dampingRatio 0.05`, with `DriveAPI` stripped by
scripts/convert_robolab_gripper_usd.py. A mimic constraint at that frequency is a near-hard equality,
so the pad cannot rotate: the measured jaw yield on a 30 mm cube is 1.26 mm, which is sub-visible.

What this changes
-----------------
On the two pad pivots only:

  * `PhysxMimicJointAPI:rotX` is removed (schema, all `physxMimicJoint:rotX:*` attributes, and the
    `referenceJoint` relationship), so the pivot becomes a free DOF;
  * `UsdPhysics.DriveAPI:angular` is applied, with `maxForce = --max-effort`. Stiffness/damping are
    authored 0 deliberately: OmniGibson force-writes them from the controller config's
    `isaac_kp`/`isaac_kd` on **every** `og.sim.play()` (robot.py `update_controller_mode`), so any
    value authored here would be a lie. `maxForce` is the one drive parameter OmniGibson does *not*
    touch for a non-holonomic robot, which is why it has to live in the asset.

The rest of the linkage is left mimic-coupled on purpose: `right_outer_knuckle_joint`
(naturalFrequency 1e6, the left/right symmetry constraint) and the two `*_inner_finger_knuckle_joint`
(which only carry the cosmetic inner_knuckle leaf). Keeping the linkage closed while only the pad
pivots is both closer to the real mechanism and far better behaved than freeing everything.

Because the pivots are now driven, they MUST be claimed by a controller: OmniGibson asserts that no
un-controlled DOF has a DriveAPI (robot.py:658). That is what
realm/robots/definitions/droid_robolab_padspring/ + realm/robots/padspring_gripper_controller.py are
for -- the pad DOFs join `finger_joint` in the gripper group, and the controller feeds them the angle
the mimic relation would have produced.

Usage (inside the container -- pxr only exists once a Kit app is up):
    python /app/scripts/make_padspring_gripper_usd.py            # defaults, writes the v2 variant
    python /app/scripts/make_padspring_gripper_usd.py --max-effort 3.0 --dst /app/.../foo.usd
"""

import argparse
import os
import shutil

import omnigibson as og

DEFAULT_SRC = "/app/realm/robots/panda_robotiq/droid_robolab_v2.usd"
DEFAULT_DST = "/app/realm/robots/panda_robotiq/droid_robolab_padspring.usd"

# outer_finger -> inner_finger on each side: the pad pivot, where the real gripper has its spring.
PAD_JOINTS = ("left_inner_finger_joint", "right_inner_finger_joint")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=DEFAULT_SRC)
ap.add_argument("--dst", default=DEFAULT_DST)
ap.add_argument("--max-effort", type=float, default=2.0,
                help="drive:angular:physics:maxForce (N.m) on each pad pivot. This caps the spring: "
                     "it has to beat the pad's own weight torque (~0.01 N.m) by a wide margin, and "
                     "it is what decides whether the pad holds or folds under the pad contact force.")
ap.add_argument("--limit-deg", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="optionally retighten physics:lowerLimit/upperLimit (degrees, magnitudes; the "
                     "sign is applied per side from the mimic gearing). Default: leave at +/-180.")
args = ap.parse_args()

og.launch()

import omnigibson.lazy as lazy  # noqa: E402

Usd, UsdPhysics, PhysxSchema = lazy.pxr.Usd, lazy.pxr.UsdPhysics, lazy.pxr.PhysxSchema

MIMIC_INSTANCES = ("rotX", "rotY", "rotZ", "transX", "transY", "transZ")


def find_joint(stage, name):
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if prim.GetName() == name and "Joint" in prim.GetTypeName():
            return prim
    return None


def strip_mimic(prim):
    """Remove the PhysxMimicJointAPI (schema + attrs + referenceJoint rel). Returns what it read."""
    was = {}
    for inst in MIMIC_INSTANCES:
        if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, inst):
            continue
        for suffix in ("gearing", "offset", "naturalFrequency", "dampingRatio", "referenceJointAxis"):
            a = prim.GetAttribute(f"physxMimicJoint:{inst}:{suffix}")
            if a:
                was[f"{inst}:{suffix}"] = a.Get()
        r = prim.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
        if r:
            was[f"{inst}:referenceJoint"] = [str(t) for t in r.GetTargets()]
        prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, inst)
    for p in list(prim.GetProperties()):
        if p.GetName().startswith("physxMimicJoint:"):
            prim.RemoveProperty(p.GetName())
    return was


def add_drive(prim, max_effort):
    """Apply an angular force drive with zero gains and a capped maxForce."""
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(0.0)     # overwritten from isaac_kp on every play -- see docstring
    drive.CreateDampingAttr().Set(0.0)       # overwritten from isaac_kd on every play
    drive.CreateMaxForceAttr().Set(float(max_effort))


def main():
    assert os.path.isfile(args.src), f"no source asset at {args.src}"
    assert os.path.abspath(args.src) != os.path.abspath(args.dst), "refusing to edit the source in place"
    shutil.copyfile(args.src, args.dst)
    print(f"copied {args.src}\n    -> {args.dst}")

    stage = Usd.Stage.Open(args.dst)
    for name in PAD_JOINTS:
        prim = find_joint(stage, name)
        assert prim is not None, f"{name} not found in {args.src}"
        was = strip_mimic(prim)
        assert was, f"{name} had no PhysxMimicJointAPI -- is this the right source asset?"
        gearing = was.get("rotX:gearing", was.get("rotZ:gearing"))
        print(f"\n[{name}] {prim.GetPath()}")
        print(f"    removed mimic: {was}")
        add_drive(prim, args.max_effort)
        print(f"    added DriveAPI:angular  maxForce={args.max_effort}  stiffness=0 damping=0")
        if args.limit_deg is not None:
            lo, hi = sorted(abs(v) for v in args.limit_deg)
            # The mimic drove this DOF to gearing * q_finger, q_finger in [0, 45] deg, so the swept
            # side is the sign of the gearing. Headroom is allowed on both sides either way.
            if gearing is not None and float(gearing) < 0:
                new_lo, new_hi = -hi, +lo
            else:
                new_lo, new_hi = -lo, +hi
            prim.GetAttribute("physics:lowerLimit").Set(float(new_lo))
            prim.GetAttribute("physics:upperLimit").Set(float(new_hi))
            print(f"    limits -> [{new_lo}, {new_hi}] deg (gearing {gearing})")
    stage.Save()

    # --- verify by reopening ------------------------------------------------------------------
    print("\nVERIFY (reopened from disk):")
    stage = Usd.Stage.Open(args.dst)
    ok = True
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if "Joint" not in prim.GetTypeName():
            continue
        nm = prim.GetName()
        if not any(k in nm for k in ("finger_joint", "knuckle_joint")):
            continue
        mimic = [s for s in prim.GetAppliedSchemas() if "Mimic" in s]
        drive = [s for s in prim.GetAppliedSchemas() if "Drive" in s]
        mf = prim.GetAttribute("drive:angular:physics:maxForce")
        st = prim.GetAttribute("drive:angular:physics:stiffness")
        print(f"  {nm:<34} mimic={mimic or '-'}  drive={drive or '-'}  "
              f"maxForce={mf.Get() if mf else None}  stiffness={st.Get() if st else None}  "
              f"lim=[{prim.GetAttribute('physics:lowerLimit').Get()}, "
              f"{prim.GetAttribute('physics:upperLimit').Get()}]")
        if nm in PAD_JOINTS:
            ok &= (not mimic) and bool(drive)
        elif nm != "finger_joint":
            ok &= bool(mimic)
    print(f"\nPADSPRING_USD_{'OK' if ok else 'FAIL'} {args.dst}")


if __name__ == "__main__":
    main()
    og.shutdown()
