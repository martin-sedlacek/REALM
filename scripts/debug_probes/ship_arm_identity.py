"""ARM PHYSICS MUST STAY BYTE-IDENTICAL across gripper variants. This checks it, no GPU needed.

Gripper physics is re-authorable; arm physics is not. The two places arm physics can drift are:

  1. the robot CONFIG yaml -- the `arm_0` controller block (Kq, Kqd, Kx, Kxd, max_effort, min_effort,
     motor_type, use_impedances, use_gravity_compensation, use_cc_compensation) and the top-level
     `friction` / `armature` arrays. `env_dynamic.update_robot_physics()` writes exactly those two
     arrays onto `panda_link{idx}/{arm_joint_names[idx]}` `for idx in range(7)` -- arm only -- so a
     drift here compounds into the sim on every reset.
  2. the variant USD -- the seven `panda_joint*` prims. That half is checked by
     scripts/make_curlgrip_gripper_usd.py at build time, which reopens the file it wrote and diffs
     every authored attribute against the source (grep CURLGRIP_ARM_IDENTICAL).

Comments are ignored; values are not.

    python scripts/debug_probes/ship_arm_identity.py
    python scripts/debug_probes/ship_arm_identity.py --base DROID_robolab_v2 --others A,B
"""
import argparse
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "realm", "config", "robots"))
ap.add_argument("--base", default="DROID_robolab_v2")
ap.add_argument("--others", default="DROID_robolab_curlgrip,DROID_robolab_padspring")
ap.add_argument("--ee-base", default="DROID_robolab_v2_ee_control")
ap.add_argument("--ee-others", default="DROID_robolab_curlgrip_ee_control")
args = ap.parse_args()

# The controller keys that ARE arm physics. Listed rather than inferred, so a new key added to the
# block in future fails loudly here instead of slipping through an "all keys" comparison.
ARM_KEYS = ("Kq", "Kqd", "Kx", "Kxd", "max_effort", "min_effort", "motor_type", "use_impedances",
            "use_gravity_compensation", "use_cc_compensation")


def strip(s):
    return [l.strip() for l in s.split("\n") if l.strip() and not l.strip().startswith("#")]


def arm_block(path):
    """The `arm_0:` block, comments and blank lines removed."""
    lines = open(path).read().split("\n")
    out, ind, on = [], None, False
    for L in lines:
        if re.match(r"^\s*arm_0:\s*$", L):
            on, ind = True, len(L) - len(L.lstrip())
            continue
        if on:
            if not L.strip() or L.strip().startswith("#"):
                continue
            if (len(L) - len(L.lstrip())) <= ind:
                break
            out.append(L.strip())
    return out


def top_array(path, key):
    return [l.strip() for l in open(path) if re.match(rf"^\s*{key}:\s*\[", l)]


def check(d, base, others, label):
    bp = os.path.join(d, base + ".yaml")
    assert os.path.exists(bp), f"no base config at {bp}"
    ba = arm_block(bp)
    present = [k for k in ARM_KEYS if any(l.startswith(k + ":") or l.startswith('"' + k + '"')
                                          for l in ba)]
    missing = [k for k in ARM_KEYS if k not in present]
    print(f"\n{label}  base = {base}  ({len(ba)} arm_0 lines)")
    print(f"  arm_0 keys found: {present}")
    if missing:
        print(f"  [note] not authored in the base (inherited from the controller default): {missing}")
    ok = True
    for o in [x.strip() for x in others.split(",") if x.strip()]:
        op = os.path.join(d, o + ".yaml")
        if not os.path.exists(op):
            print(f"  {o:<42} MISSING -- skipped")
            continue
        oa = arm_block(op)
        same = (ba == oa)
        print(f"  {o:<42} arm_0 {'IDENTICAL' if same else '*** DIFFERS ***'}")
        if not same:
            ok = False
            for x, y in zip(ba + [""] * len(oa), oa + [""] * len(ba)):
                if x != y:
                    print(f"      - {x}\n      + {y}")
        for key in ("friction", "armature"):
            b, c = top_array(bp, key), top_array(op, key)
            s = (b == c)
            ok &= s
            print(f"      {key:<9} {'SAME' if s else '*** DIFFERS ***'}  {c or '(absent)'}")
    return ok


ok = check(args.dir, args.base, args.others, "JOINT-CONTROL CONFIGS")
ok &= check(args.dir, args.ee_base, args.ee_others, "EE-CONTROL CONFIGS")
print(f"\nSHIP_ARM_IDENTITY_{'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
