"""Do all members of a vector build place the SAME object the SAME way?

Every other test in tests/ asks "did the run produce its artifacts". None of them looks at the
scene. That gap is not hypothetical: measured 2026-08-16 on `realm_og391_v2.sif` at MODE=stock,
open_drawer at num_envs=2 put member 0's cabinet at orientation (0.0014, -0.0143, 0.0882, 0.9960)
and member 1's at (0.7044, 0.0616, 0.0616, 0.7044) -- 90 degrees apart, i.e. scene 0's cabinet
lying on its back -- while `tests/test_single_task.py --task_id 8` and
`tests/test_vector_integrity.py --cells 8:Default,9:Default` BOTH PASSED on that same build.

The mechanism, from OG-lite's `USDObject._preapply_articulation_root`: referencing a layer whose
`upAxis` disagrees with the stage's makes Kit's metrics assembler append
`xformOp:rotateX:unitsResolve` to the referencing prim's `xformOpOrder`. No OmniGibson pose setter
writes or strips that op, so it silently post-multiplies every pose set on the prim -- and the
assembler's UnitsAdjust layer is content-hash keyed, so it is materialised for the FIRST reference
to the asset only. Hence the asymmetry between members rather than a uniform error.

WHAT IS ASSERTED, and why it is cross-member agreement rather than an absolute pose: the members
are the same task config built N times, so whatever the right orientation is, every member must
agree on it. That invariant needs no golden value, cannot drift when a task config is retuned, and
is exactly what the defect breaks. The `unitsResolve` check is the direct form of the same thing.

    MODE=oglite ./scripts/clara/interactive/rr python -u tests/test_scene_object_placement.py
    MODE=stock  ./scripts/clara/interactive/rr python -u tests/test_scene_object_placement.py

Costs one vector build (~4 min). Defaults to open_drawer because that is the only asset in the
repo authored upAxis=Y; pass --task_cfg_path to check another.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

import omnigibson as og

from realm.environments.env_vector import RealmVectorEnvironment
from realm.sim_config import set_sim_config

ORI_TOL = 1e-3


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=2,
                   help="2 is the minimum that can see this: the defect is an asymmetry between "
                        "the FIRST reference to an asset and the rest.")
    p.add_argument("--task_cfg_path", default="REALM_DROID10/open_drawer/default.yaml")
    p.add_argument("--robot", default="DROID_robolab_v2")
    args = p.parse_args()

    set_sim_config(robot=args.robot)
    vec = RealmVectorEnvironment(args.num_envs, task_cfg_path=args.task_cfg_path,
                                 perturbations=["Default"], robot=args.robot,
                                 rendering_mode="rt")

    failures = []
    rows = []
    for i, env in enumerate(vec.envs):
        mo = env.main_objects[0]
        prim = mo.prim
        order = prim.GetAttribute("xformOpOrder").Get() if prim.HasAttribute("xformOpOrder") else None
        order = list(order) if order else []
        resolve = [o for o in order if "unitsResolve" in o]
        pos, ori = mo.get_position_orientation()
        rows.append((i, mo.name, resolve, [round(float(v), 4) for v in ori]))
        print(f"member {i}: {mo.name}  unitsResolve={resolve or 'NONE'}  ori={ori}", flush=True)
        if resolve:
            failures.append(
                f"member {i}'s '{mo.name}' prim carries {resolve} in its xformOpOrder. No "
                f"OmniGibson pose setter writes or strips that op, so every pose set on this prim "
                f"is silently post-multiplied by it.")

    # Cross-member agreement. No golden value: the members are one config built N times.
    if len(rows) > 1:
        ref_i, ref_name, _, ref_ori = rows[0]
        for i, name, _, ori in rows[1:]:
            worst = max(abs(a - b) for a, b in zip(ref_ori, ori))
            print(f"member {i} vs member {ref_i}: max |dq| = {worst:.4f}", flush=True)
            if worst > ORI_TOL:
                failures.append(
                    f"member {i}'s '{name}' orientation {ori} disagrees with member {ref_i}'s "
                    f"{ref_ori} by {worst:.4f} > {ORI_TOL}. Same task config, same asset -- the "
                    f"members must agree on where the object goes.")

    print("\n" + "=" * 78, flush=True)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print(f"PASSED -- all {len(rows)} members place '{rows[0][1]}' identically and no member "
              f"carries a unitsResolve op", flush=True)
    print("=" * 78, flush=True)
    og.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
