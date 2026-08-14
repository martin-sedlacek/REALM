"""Write a TINY variant USD that only overrides the gripper's PhysX mimic-constraint parameters.

Why this exists
---------------
`physxMimicJoint:<inst>:naturalFrequency` and `:dampingRatio` are NOT in the `PhysxMimicJointAPI`
schema of Isaac Sim 5.1.0 / omni.physx 107.3.26 -- `Usd.SchemaRegistry` lists only gearing, offset,
referenceJoint and referenceJointAxis, and `_physxSchema.so` does not contain the string at all.
omni.physx nonetheless reads them by literal token name: its bindings export
`MIMIC_JOINT_ATTRIBUTE_NAME_NATURAL_FREQUENCY_ROT{X,Y,Z}` and
`MIMIC_JOINT_ATTRIBUTE_NAME_DAMPING_RATIO_ROT{X,Y,Z}`, i.e. they are *custom* attributes the physics
parser looks up directly. Custom attributes read that way are usually consumed when the articulation
is parsed, so writing them onto a live stage may not propagate -- hence this file: a variant that is
in place BEFORE the robot is ever loaded, so the value cannot be missed.

It is a ~2 KB `.usda` that sublayers the shipped 14 MB asset and authors nothing but the overridden
floats, so the shipped `droid_robolab_v2.usd` is never touched and no 14 MB copy is made.

    python scripts/debug_probes/make_mimic_variant.py --nf 100 --dr 0.05 --out /app/tmp/variants

Feed the result to the squeeze probe with `--variant-usd <path>`, which monkeypatches
`Robot.usd_path` for the robolab asset only. Nothing under `data/` is touched: the
`omnigibson-robot-assets/models/*` symlinks are shared between worktrees, so a variant must NOT be
registered as a new robot `model`.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/app/realm/robots/panda_robotiq/droid_robolab_v2.usd")
ap.add_argument("--out", default="/app/tmp/variants")
ap.add_argument("--name", default=None, help="basename; default is derived from the values")
ap.add_argument("--nf", type=float, default=None, help="naturalFrequency on the four INNER mimic joints")
ap.add_argument("--dr", type=float, default=None, help="dampingRatio on the same four")
ap.add_argument("--outer-nf", type=float, default=None, help="naturalFrequency on right_outer_knuckle_joint")
ap.add_argument("--outer-dr", type=float, default=None, help="dampingRatio on the same")
ap.add_argument("--max-force", type=float, default=None,
                help="drive:angular:physics:maxForce on finger_joint (authored 16.5)")
ap.add_argument("--stiffness", type=float, default=None,
                help="drive:angular:physics:stiffness on finger_joint (authored 100; OmniGibson "
                     "overwrites it with isaac_kp=1e7 at play, so this alone changes nothing)")
ap.add_argument("--restore-follower-drive", action="store_true",
                help="TASK 2: put back the DriveAPI that scripts/convert_robolab_gripper_usd.py:70 "
                     "strip_mimic_drives removes from the four INNER mimic joints -- zero gains, "
                     "infinite force limit, exactly as RoboLab's asset carries it. This is the last "
                     "authored USD difference between the two stacks' grippers. OmniGibson asserts "
                     "at robot.py:658 that no uncontrolled DOF is driven, so a run using this "
                     "variant needs MODE=oglite with that assert relaxed.")
args = ap.parse_args()

# joint prim name -> PhysxMimicJointAPI instance token. The instance is NOT the physics:axis: these
# joints all author axis Z, yet four use rotX and right_outer_knuckle_joint uses rotZ. Read off the
# shipped asset (scripts/debug_probes/gripper_squeeze_compliance.py prints it at runtime too).
INNER = {
    "left_inner_finger_joint": "rotX",
    "left_inner_finger_knuckle_joint": "rotX",
    "right_inner_finger_joint": "rotX",
    "right_inner_finger_knuckle_joint": "rotX",
}
OUTER = {"right_outer_knuckle_joint": "rotZ"}


def fmt(v):
    return repr(float(v))


def joint_over(jname, inst, nf, dr):
    lines = []
    if nf is not None:
        lines.append(f'            custom float physxMimicJoint:{inst}:naturalFrequency = {fmt(nf)}')
    if dr is not None:
        lines.append(f'            custom float physxMimicJoint:{inst}:dampingRatio = {fmt(dr)}')
    if not lines:
        return ""
    body = "\n".join(lines)
    return f'        over "{jname}"\n        {{\n{body}\n        }}\n'


def drive_over(jname):
    """Re-apply the vestigial DriveAPI on a follower: zero gains, unbounded force limit.

    RoboLab's shipped gripper carries exactly this on its four inner mimic joints. The REALM
    converter strips it because OmniGibson refuses to load a robot whose uncontrolled DOFs are
    driven; what is written back here is byte-for-byte what was removed, not a stronger drive. A
    zero-stiffness zero-damping drive should exert no force at all, so if this moves the residual
    the interesting fact is that PhysX treats "drive present with zero gains" differently from
    "no drive", e.g. by giving the DOF a driven-joint solver path or a maxForce-bounded reaction.
    """
    return (f'        over "{jname}"\n        {{\n'
            f'            prepend apiSchemas = ["PhysicsDriveAPI:angular"]\n'
            f'            float drive:angular:physics:stiffness = 0\n'
            f'            float drive:angular:physics:damping = 0\n'
            f'            float drive:angular:physics:maxForce = inf\n'
            f'            uniform token drive:angular:physics:type = "force"\n'
            f'            float drive:angular:physics:targetPosition = 0\n'
            f'        }}\n')


overs = ""
for jname, inst in INNER.items():
    overs += joint_over(jname, inst, args.nf, args.dr)
if args.restore_follower_drive:
    for jname in INNER:
        overs += drive_over(jname)
for jname, inst in OUTER.items():
    overs += joint_over(jname, inst, args.outer_nf, args.outer_dr)
drive = []
if args.max_force is not None:
    drive.append(f'            float drive:angular:physics:maxForce = {fmt(args.max_force)}')
if args.stiffness is not None:
    drive.append(f'            float drive:angular:physics:stiffness = {fmt(args.stiffness)}')
if drive:
    overs += '        over "finger_joint"\n        {\n' + "\n".join(drive) + "\n        }\n"

if not overs:
    raise SystemExit("nothing to override -- pass at least one of --nf/--dr/--outer-nf/--outer-dr/"
                     "--max-force/--stiffness/--restore-follower-drive")

name = args.name or ("mimic_" + "_".join(
    f"{k}{v:g}" for k, v in (("nf", args.nf), ("dr", args.dr), ("onf", args.outer_nf),
                             ("odr", args.outer_dr), ("mf", args.max_force),
                             ("st", args.stiffness)) if v is not None).replace(".", "p")
                    + ("_followerdrive" if args.restore_follower_drive else ""))
os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, f"{name}.usda")

# `subLayers` (not a reference) so /panda keeps its identity and every child prim composes exactly as
# in the shipped file; the overs below are simply a stronger opinion on four floats. defaultPrim has
# to be restated because layer metadata does not compose up from a sublayer.
text = f'''#usda 1.0
(
    """Mimic-constraint variant of droid_robolab_v2.usd -- generated by
    scripts/debug_probes/make_mimic_variant.py. Overrides ONLY the floats listed below; every prim,
    mesh and joint comes from the sublayer, which is the shipped asset, unmodified.

    inner naturalFrequency = {args.nf}   inner dampingRatio = {args.dr}
    outer naturalFrequency = {args.outer_nf}   outer dampingRatio = {args.outer_dr}
    finger_joint maxForce = {args.max_force}   stiffness = {args.stiffness}
    """
    defaultPrim = "panda"
    subLayers = [
        @{args.src}@
    ]
)

over "panda"
{{
    over "Joints"
    {{
{overs}    }}
}}
'''
with open(path, "w") as f:
    f.write(text)
print(f"wrote {path}")
print(text)
