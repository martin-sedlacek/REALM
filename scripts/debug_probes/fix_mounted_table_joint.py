"""Repoint the mounted asset's panda_table_joint at prims that actually exist. Edits USD in place.

THE BUG. droid_robolab_v2_mounted.usd carries

    /panda/table/panda_table_joint   (PhysicsFixedJoint)
        physics:body0 -> /panda/droid_mounted/droid_mounted/base_link    MISSING
        physics:body1 -> /panda/droid_mounted/panda_link0                MISSING

There is no /panda/droid_mounted prim in this asset -- both targets are leftovers from an earlier
hierarchy, so the joint attaches nothing to nothing. Two consequences, and the second is the one that
actually kills construction:

1. The `table` link is bolted to nothing: it is a free-floating rigid body.
2. entity_prim.py:229 derives a link name from the joint's body1 with
   `b1[0].pathString.split("/")[-1]`, a BASENAME. The dangling path still ends in "panda_link0", so
   panda_link0 lands in joint_children even though the joint is broken. That removes the arm's real
   root from `valid_root_links = links - joint_children`, leaving `table` -- the one link nothing joints
   into -- as the sole candidate. root_link_name becomes "table", and
   ArticulationView("/panda/table") is built on a prim with no arm joints beneath it.

Hence, on construction:

    robot.py:2816  list(self.joints.keys()).index(name)
    ValueError: 'panda_joint1' is not in list

Measured root_link_name, replicating entity_prim.py:203-241 offline:

    droid_robolab_v2.usd         (bare)          valid_root=['panda_link0'] -> 'panda_link0'   works
    droid_robolab_v2_mounted.usd (mounted)       valid_root=['table']       -> 'table'         fails
    droid_mounted.usd            (stock mounted) valid_root=3 candidates    -> 'base_link' via
                                                 the len!=1 fallback, which happens to be the name
                                                 of its real mount link -- which is why the stock
                                                 asset survives the same class of wiring.

THE FIX: body0 -> /panda/table, body1 -> /panda/panda_link0. The table becomes the arm's mount, so
panda_link0 is a genuine joint child, `table` stays the only un-jointed link and is therefore still
the inferred root -- but now it is actually part of the articulation, so the view at /panda/table sees
the whole arm.

This is the second of two edits the asset needed. The first (fix_mounted_articulation_root.py) removed
a duplicate ArticulationRootAPI from /panda/table; necessary, because with the joint repaired that
nested root would split the arm off into its own articulation -- but on its own it fixed nothing.

    python fix_mounted_table_joint.py <asset.usd> [--dry-run]
"""

import argparse
import sys

from pxr import Sdf

JOINT = "/panda/table/panda_table_joint"
WANT = {"physics:body0": "/panda/table", "physics:body1": "/panda/panda_link0"}


def targets(spec, name):
    for rel in spec.relationships:
        if rel.name == name:
            return [str(t) for t in rel.targetPathList.explicitItems]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layer = Sdf.Layer.FindOrOpen(args.asset)
    if layer is None:
        sys.exit(f"could not open {args.asset}")

    joint = layer.GetPrimAtPath(JOINT)
    if joint is None:
        sys.exit(f"no joint spec at {JOINT}")

    # Every replacement target must already exist, or we would be trading one dangling path for another.
    for path in WANT.values():
        if layer.GetPrimAtPath(path) is None:
            sys.exit(f"refusing: replacement target {path} does not exist in this layer")

    changed = False
    for name, want in WANT.items():
        have = targets(joint, name)
        print(f"  {name}: {have} -> ['{want}']")
        if have == [want]:
            print("    already correct, skipping")
            continue
        if have is None:
            sys.exit(f"refusing: {JOINT} has no {name} relationship at all")
        # Only ever repoint a target that is genuinely dangling; a valid target means the asset is
        # wired differently from what this script was written for and should not be rewritten blind.
        for t in have:
            if layer.GetPrimAtPath(t) is not None:
                sys.exit(f"refusing: {name} target {t} EXISTS -- this is not the dangling-path bug")
        if args.dry_run:
            changed = True
            continue
        rel = next(r for r in joint.relationships if r.name == name)
        rel.targetPathList.explicitItems[:] = [Sdf.Path(want)]
        changed = True

    if not changed:
        print("nothing to do")
        return
    if args.dry_run:
        print("--dry-run: not writing")
        return

    layer.Save()
    print(f"saved {args.asset}")

    reopened = Sdf.Layer.FindOrOpen(args.asset)
    reopened.Reload()
    j = reopened.GetPrimAtPath(JOINT)
    for name, want in WANT.items():
        got = targets(j, name)
        print(f"verify {name}: {got}")
        if got != [want]:
            sys.exit(f"VERIFY FAILED for {name} -- restore from git")
    print("OK")


if __name__ == "__main__":
    main()
