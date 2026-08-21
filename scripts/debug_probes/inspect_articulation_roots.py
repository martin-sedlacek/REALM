"""Report every articulation root and every joint in a USD, so a mounted asset can be compared to a
bare one WITHOUT booting Isaac.

WHY THIS EXISTS: DROID_robolab_v2 (mounted) dies during construction with

    robot.py:2816  list(self.joints.keys()).index(name)
    ValueError: 'panda_joint1' is not in list

reached via _default_arm_ik_controller_configs -> arm_control_idx. That says the articulation
OmniGibson enumerated does not contain the arm joints. This script answers the prior question --
what does the asset actually declare -- by reading the USD directly.

RESOLVED 2026-08-21 -- it was a dangling fixed joint whose body targets no longer existed, and whose
BASENAME still matched the arm's real root link. Full account, including the two wrong turns and why a
present-but-dangling relationship defeats existence checks: docs/code_archaeology.md, "The mounted
robolab_v2 asset's dangling joint". This script stays as the diagnostic, not as a one-off.

pxr ONLY, no omnigibson import: `omnigibson.lazy` is a LazyImporter that CACHES NEGATIVE LOOKUPS, so
touching lazy.pxr before og.launch() poisons it for the rest of the process. Importing pxr straight
from Isaac's python sidesteps that entirely and needs no GPU, so this runs on a login node.

    apptainer exec --userns $REALM_SIF python inspect_articulation_roots.py <a.usd> [<b.usd> ...]
"""

import sys

from pxr import Usd, UsdPhysics


def report(path):
    print("=" * 100)
    print(f"STAGE: {path}")
    stage = Usd.Stage.Open(path)
    if stage is None:
        print("  !! could not open")
        return

    roots, joints, arts = [], [], []
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        # An articulation root is where PhysX starts solving. OmniGibson wraps ONE of these in an
        # ArticulationView; if the asset declares several, which one it picks decides which joints
        # end up in robot.joints -- and therefore whether 'panda_joint1' is findable.
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(p)
        if prim.IsA(UsdPhysics.Joint):
            body0 = prim.GetRelationship("physics:body0").GetTargets()
            body1 = prim.GetRelationship("physics:body1").GetTargets()
            joints.append((p, prim.GetTypeName(), bool(body0), bool(body1)))
        if prim.IsA(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            arts.append(p)

    print(f"\n  ARTICULATION ROOTS ({len(roots)}):")
    for r in roots or ["    (none)"]:
        print(f"    {r}")

    # Only the movable joints get a DOF and land in robot.joints. A fixed joint is structure.
    movable = [j for j in joints if j[1] != "PhysicsFixedJoint"]
    print(f"\n  JOINTS: {len(joints)} total, {len(movable)} movable")
    for p, t, b0, b1 in joints:
        flag = "" if (b0 and b1) else "   <-- MISSING BODY REL"
        print(f"    {t:22s} {p}{flag}")

    print(f"\n  panda_joint* present: {sorted(p.rsplit('/', 1)[-1] for p, *_ in joints if 'panda_joint' in p)}")
    print(f"  rigid bodies: {len(arts)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for a in sys.argv[1:]:
        report(a)
