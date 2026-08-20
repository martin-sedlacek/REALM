"""De-articulate the mounted table so `/panda` is the sole articulation root. Edits the USD IN PLACE.

THE BUG. droid_robolab_v2_mounted.usd declares an articulation root on BOTH /panda and /panda/table.
PhysX allows one articulation root per hierarchy, so the articulation OmniGibson wraps is the inner
/panda/table, whose only joint is the fixed panda_table_joint. robot.joints therefore never contains
the arm joints and construction dies in arm_control_idx with

    ValueError: 'panda_joint1' is not in list

Measured apiSchemas, mounted vs bare:

    /panda        BOTH assets   [PhysicsArticulationRootAPI, PhysxArticulationAPI, RobotAPI]
    /panda/table  mounted only  [PhysicsRigidBodyAPI, PhysicsMassAPI,
                                 PhysicsArticulationRootAPI, PhysxArticulationAPI,   <-- duplicates
                                 MaterialBindingAPI]                                     /panda's root

The table was handed a duplicate of the root /panda already has. Dropping BOTH articulation APIs from
the table leaves it a rigid body with mass and a material, mounted by its fixed joint -- which is what
a link should be -- and leaves /panda as the sole root, matching the bare asset exactly.

The earlier fixroot variant removed only PhysicsArticulationRootAPI and left PhysxArticulationAPI
behind. That is why this script exists as well: PhysxArticulationAPI is the PhysX-side companion to the
root and has no meaning on a prim that is no longer one.

WHY Sdf AND Save(), NOT Usd AND Export(). Export() builds a NEW stage and copies metadata across, and
that copy warns on Omniverse-only fields it cannot represent (hide_in_stage_window, no_delete) -- it
composes references, of which this asset has 16. Editing the root layer's prim spec and calling
Save() rewrites only this layer, touching nothing that composition brought in.

REFUSES TO WRITE unless the result has exactly one articulation root, and re-opens the saved file to
confirm. The original is recoverable from git (it is a tracked file) -- check `git status` is clean on
it before running, so HEAD holds the pristine copy.

    python fix_mounted_articulation_root.py <asset.usd> [--prim /panda/table] [--dry-run]
"""

import argparse
import sys

from pxr import Sdf

# Removing these two is the whole fix. The rigid-body, mass and material APIs stay: the table is still
# a physical object, it just stops being its own articulation.
STRIP = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]


def api_schemas(layer, path):
    spec = layer.GetPrimAtPath(path)
    if spec is None:
        return None, None
    if "apiSchemas" not in spec.ListInfoKeys():
        return spec, []
    return spec, list(spec.GetInfo("apiSchemas").explicitItems)


def roots(layer):
    """Every prim spec declaring an articulation root, by Sdf inspection (no stage composition)."""
    found = []

    def walk(spec):
        if "apiSchemas" in spec.ListInfoKeys():
            if "PhysicsArticulationRootAPI" in spec.GetInfo("apiSchemas").explicitItems:
                found.append(spec.path.pathString)
        for child in spec.nameChildren:
            walk(child)

    walk(layer.pseudoRoot)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("--prim", default="/panda/table")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layer = Sdf.Layer.FindOrOpen(args.asset)
    if layer is None:
        sys.exit(f"could not open {args.asset}")

    before = roots(layer)
    print(f"articulation roots before ({len(before)}): {before}")
    if len(before) < 2:
        sys.exit(f"expected >=2 roots (the bug); found {len(before)}. Nothing to do -- refusing.")

    spec, schemas = api_schemas(layer, args.prim)
    if spec is None:
        sys.exit(f"no prim spec at {args.prim}")
    print(f"{args.prim} apiSchemas before: {schemas}")

    keep = [s for s in schemas if s not in STRIP]
    removed = [s for s in schemas if s in STRIP]
    if not removed:
        sys.exit(f"{args.prim} carries none of {STRIP} -- refusing")
    print(f"  removing: {removed}")
    print(f"  keeping:  {keep}")

    if args.dry_run:
        print("--dry-run: not writing")
        return

    listop = Sdf.TokenListOp()
    listop.explicitItems = keep
    spec.SetInfo("apiSchemas", listop)

    after = roots(layer)
    print(f"articulation roots after  ({len(after)}): {after}")
    if len(after) != 1:
        # The layer is dirty but unsaved; abandoning the process leaves the file untouched on disk.
        sys.exit(f"REFUSING to save: expected exactly 1 root, got {len(after)}. File NOT modified.")

    layer.Save()
    print(f"saved {args.asset}")

    # Save() returning is not proof the edit serialised. Re-open from disk and re-check.
    layer.Reload()
    reopened = Sdf.Layer.FindOrOpen(args.asset)
    v = roots(reopened)
    _, vs = api_schemas(reopened, args.prim)
    print(f"verify roots ({len(v)}): {v}")
    print(f"verify {args.prim} apiSchemas: {vs}")
    if len(v) != 1 or any(s in vs for s in STRIP):
        sys.exit("VERIFY FAILED -- restore from git and investigate")
    print("OK")


if __name__ == "__main__":
    main()
