"""Remove a NESTED UsdPhysics.ArticulationRootAPI from a robot USD, writing a new file.

THE BUG THIS FIXES. droid_robolab_v2_mounted.usd declares ArticulationRootAPI on BOTH /panda and
/panda/table -- a root nested inside a root. PhysX allows one articulation root per hierarchy, so the
articulation OmniGibson ends up wrapping is /panda/table, whose only joint is the fixed
panda_table_joint. robot.joints therefore never contains the arm joints and construction dies in
arm_control_idx with:

    ValueError: 'panda_joint1' is not in list

Measured, with scripts/debug_probes/inspect_articulation_roots.py:

    droid_robolab_v2.usd         (bare, works)  1 root:  /panda
    droid_robolab_v2_mounted.usd (fails)        2 roots: /panda AND /panda/table   <-- nested
    droid_mounted.usd            (stock)        2 roots: /panda/panda_link0, /panda/base_link
                                                         -- SIBLINGS, and /panda is not a root

So the stock mounted asset is not a counterexample: sibling roots are legal, a root inside a root is
not. Stripping the inner one leaves /panda as the sole root, which is exactly the bare asset's
topology.

WHAT THIS DOES NOT TOUCH: no joint, drive, limit, mass or inertia is modified -- only the presence of
one API schema on one prim. The arm's own physics is left alone. It does change how PhysX PARTITIONS
the hierarchy (table + arm become one articulation instead of two), which is what "mounted" should
mean, so this is a behavioural change to a configuration that currently cannot construct at all.

Writes a NEW file and never overwrites the input, so the original asset stays authoritative until a
smoke run says otherwise.

    python strip_nested_articulation_root.py <in.usd> <out.usd> </panda/table>
"""

import sys

from pxr import Usd, UsdPhysics


def roots(stage):
    return [p.GetPath().pathString for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)]


def main(src, dst, target):
    stage = Usd.Stage.Open(src)
    if stage is None:
        sys.exit(f"could not open {src}")

    before = roots(stage)
    print(f"roots before ({len(before)}): {before}")
    if target not in before:
        sys.exit(f"{target} does not have ArticulationRootAPI -- nothing to strip")

    prim = stage.GetPrimAtPath(target)
    if not prim:
        sys.exit(f"no prim at {target}")

    # RemoveAPI drops the schema from the prim's apiSchemas list. The prim itself, its transform and
    # its fixed joint all stay -- the table is still mounted, it just stops being its own articulation.
    ok = prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print(f"RemoveAPI({target}) -> {ok}")

    after = roots(stage)
    print(f"roots after  ({len(after)}): {after}")
    if len(after) != 1:
        sys.exit(f"REFUSING to write: expected exactly 1 root after the strip, got {len(after)}")

    stage.Export(dst)
    print(f"wrote {dst}")

    # Re-open the written file and re-check, because Export() succeeding is not proof the schema
    # change survived serialisation.
    v = roots(Usd.Stage.Open(dst))
    print(f"verify reopened ({len(v)}): {v}")
    if len(v) != 1:
        sys.exit("VERIFY FAILED: reopened file does not have exactly 1 articulation root")
    print("OK")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
