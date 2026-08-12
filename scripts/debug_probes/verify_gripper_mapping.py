"""Verify a robot's gripper command mapping BEFORE spending a batch on it.

Drop into the pump's inbox (see README.md). Holds each binary gripper command for 60 steps and reads
back both the physical jaw gap and the `gripper_state` the policy is actually handed, then refuses
to run rollouts if either is inverted.

Why this exists
---------------
On 2026-08-11 the robolab 2F-85 was run for a full 10-repeat batch with the gripper command
inverted -- REALM's CLOSE opened the jaw -- because "pad separation" had been measured between the
two inner_finger LINK ORIGINS, which at the time sat at the gripper mount rather than on the pads.
A 2F-85 is a four-bar linkage: the knuckles swing APART as the pads close TOGETHER, so a separation
measured anywhere but the pads reports the exact opposite of the truth. Independently,
ROBOT_OBS_PROFILES had gripper_open_qpos/gripper_closed_qpos swapped, so pi0.5 -- which is
closed-loop on gripper_state -- was told "closed" whenever the hand was open.

Both are silent failures: nothing raises, the policy simply never commits to a grasp. This check
costs ~10 s of sim and catches both.

Two preconditions for the measurement to mean anything:
  1. the two finger links' origins must actually be ON the pads. If they are not, run
     scripts/fix_robolab_link_origins.py first -- otherwise you are measuring the linkage again.
  2. read gripper_state through extract_from_obs, not the raw joint, since the profile is the thing
     under test.
"""

# Expected jaw gap (mm) between the finger link origins at each extreme. Robolab measured
# 116.2 mm open / 33.0 mm shut; set OPEN_MM/SHUT_MM for the asset under test.
OPEN_MM = 80.0   # a gap above this counts as "open"
SHUT_MM = 50.0   # a gap below this counts as "shut"

PASS_ALL = True


def _sep_mm():
    import numpy as np
    fl = list(env.robot_finger_links)  # noqa: F821 -- provided by the pump namespace
    a = fl[0].get_position_orientation()[0]
    b = fl[1].get_position_orientation()[0]
    a = np.asarray(a.cpu() if hasattr(a, "cpu") else a)
    b = np.asarray(b.cpu() if hasattr(b, "cpu") else b)
    return 1000.0 * float(np.linalg.norm(a - b))


def verify():
    """Returns True iff both commands drive the jaw the right way AND report the right state."""
    global PASS_ALL, obs  # noqa: F821
    import numpy as np
    from omnigibson.controllers.controller_view import ControllerView

    gk, _ = robot._controllers["gripper_0"]  # noqa: F821
    g = ControllerView._controller_groups[gk]
    print(f"gripper controller override: _open_qpos={g._open_qpos} _closed_qpos={g._closed_qpos}")

    prof = __import__("realm.inference.utils", fromlist=["x"]).get_robot_obs_profile(robot.name)  # noqa: F821
    print(f"obs profile: open_qpos={prof['gripper_open_qpos']} closed_qpos={prof['gripper_closed_qpos']}")

    fl = list(env.robot_finger_links)  # noqa: F821
    l8 = robot.links["panda_link8"].get_position_orientation()[0]  # noqa: F821
    l8 = np.asarray(l8.cpu() if hasattr(l8, "cpu") else l8)
    p = fl[0].get_position_orientation()[0]
    p = np.asarray(p.cpu() if hasattr(p, "cpu") else p)
    d = float(np.linalg.norm(p - l8))
    print(f"finger origin to flange: {d:.5f} m")
    if d < 0.05:
        print("  WARNING: origins look like they are at the MOUNT, not the pads. Any separation "
              "measured here reports the linkage, not the jaw. Run fix_robolab_link_origins.py.")
        PASS_ALL = False

    q = robot.get_joint_positions()  # noqa: F821
    q = (q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)).copy()
    arm = np.asarray(q[:7], dtype=np.float32)

    print("\ncommand -> jaw, and what the policy is told:")
    for label, cmd, want_open in (("REALM OPEN  cmd=-1", -1.0, True),
                                  ("REALM CLOSE cmd=+1", +1.0, False)):
        act = np.concatenate([arm, [cmd]]).astype(np.float32)
        for _ in range(60):
            obs, _, _, _, _ = env.step(act)  # noqa: F821
        sep = _sep_mm()
        _, _, _, _, _, _, gs = extract_from_obs(obs, robot_name=robot.name)  # noqa: F821
        jaw_open = sep > OPEN_MM
        jaw_shut = sep < SHUT_MM
        state_ok = (gs < 0.2) if want_open else (gs > 0.8)
        ok = ((jaw_open if want_open else jaw_shut) and state_ok)
        PASS_ALL &= ok
        print(f"  {label}: sep={sep:6.1f} mm -> {'OPEN' if jaw_open else 'SHUT' if jaw_shut else '???'}"
              f"   gripper_state={gs:.3f}   {'OK' if ok else 'WRONG'}")

    print(f"\nVERIFY: {'PASS' if PASS_ALL else 'FAIL -- do not run a batch'}")
    return PASS_ALL


if verify():
    print("\n--- 3 rollouts ---", flush=True)
    import numpy as np
    tps = []
    for i in range(3):
        r = rollout(max_steps=300, horizon=8, verbose=False)  # noqa: F821
        tps.append(r["task_progression"])
        print(f"  run {i}: tp={r['task_progression']:.3f} steps={r['steps']}", flush=True)
    print(f"\nmean task_progression = {np.mean(tps):.3f}")
else:
    print("\nskipping rollouts: gripper mapping is wrong, fix it before spending GPU time")
