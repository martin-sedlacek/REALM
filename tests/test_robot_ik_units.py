"""Unit checks for realm/robots: the IK solver's conversion algebra and gain preparation.

WHY: these are the pure parts of the controller stack, and they had zero coverage while carrying
the constants every EE rollout depends on. This pins their behaviour BEFORE the planned
controller base-class extraction, so that refactor has something to diff against.

WHAT IT NEEDS: the container (dm_control + MuJoCo for RobotIKSolver, torch + omnigibson for
prepare_gain), but NO GPU, NO dataset and NO running simulator -- og.sim is stubbed the same way
tests/test_joint_reset_batching.py stubs it. Run inside the container:

    python -u tests/test_robot_ik_units.py

Script-style like the rest of the suite: verdict line on stdout, exit 1 on failure.
"""
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

import numpy as np

failures = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def main():
    # ---- gains.prepare_gain (needs torch + og.sim.device; stub the sim, never launch it) ----
    import omnigibson as og
    if og.sim is None:
        og.sim = types.SimpleNamespace(device="cpu")
    import torch as th
    from realm.robots.gains import prepare_gain

    diag = prepare_gain([1.0, 2.0, 3.0])
    check("1-D gain becomes its diagonal matrix",
          diag.shape == th.Size([3, 3]) and th.equal(th.diag(diag), th.tensor([1.0, 2.0, 3.0])))
    full = prepare_gain(th.eye(6) * 4.0)
    check("2-D gain is used as given", th.equal(full, th.eye(6) * 4.0))
    check("result never tracks gradients", not diag.requires_grad and not full.requires_grad)
    try:
        prepare_gain(3.0)
        check("0-D gain raises ValueError", False, "did not raise")
    except ValueError:
        # NOTE: the module docstring says "scalars" are accepted; the code has always raised on a
        # bare scalar (0-D tensor). Pinned as-is -- every config passes lists or matrices.
        check("0-D gain raises ValueError", True)

    # ---- RobotIKSolver (dm_control/MuJoCo only; no omnigibson involved) ----
    from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver

    solver = RobotIKSolver()

    check("arm name is the string 'franka', not a bound method",
          isinstance(solver._arm.name, str) and solver._arm.name == "franka",
          repr(solver._arm.name))

    rng = np.random.default_rng(0)

    # delta <-> velocity are exact inverses inside the unit ball
    for _ in range(20):
        v = rng.uniform(-1, 1, 6)
        v[:3] /= max(1.0, np.linalg.norm(v[:3]))   # keep inside the unit balls
        v[3:] /= max(1.0, np.linalg.norm(v[3:]))
        round_trip = solver.cartesian_delta_to_velocity(solver.cartesian_velocity_to_delta(v))
        if not np.allclose(round_trip, v, atol=1e-12):
            check("cartesian velocity<->delta round-trip", False, f"{v} -> {round_trip}")
            break
    else:
        check("cartesian velocity<->delta round-trip", True)

    jd = rng.uniform(-solver.max_joint_delta, solver.max_joint_delta, 7)
    check("joint delta<->velocity round-trip",
          np.allclose(solver.joint_velocity_to_delta(solver.joint_delta_to_velocity(jd)), jd,
                      atol=1e-12))

    # over-unit velocities are clipped to exactly the max delta, direction preserved
    big = np.array([10.0, 0, 0, 0, 0, 0])
    d = solver.cartesian_velocity_to_delta(big)
    check("linear velocity clip", np.isclose(d[0], solver.max_lin_delta) and np.allclose(d[1:], 0))
    big_rot = np.array([0, 0, 0, 0, 10.0, 0])
    d = solver.cartesian_velocity_to_delta(big_rot)
    check("angular velocity clip", np.isclose(d[4], solver.max_rot_delta)
          and np.allclose(np.delete(d, 4), 0))

    # list and ndarray inputs agree (the controllers pass both)
    v = rng.uniform(-0.5, 0.5, 6)
    check("list input matches ndarray input",
          np.array_equal(solver.cartesian_velocity_to_delta(list(v)),
                         solver.cartesian_velocity_to_delta(v)))

    # a full IK solve is finite, 7-D and deterministic (same inputs -> same output)
    home = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.0])
    state = {"joint_positions": home, "joint_velocities": np.zeros(7)}
    out1 = np.asarray(solver.cartesian_velocity_to_joint_velocity(np.array([0.2, 0, 0, 0, 0, 0]), state))
    out2 = np.asarray(solver.cartesian_velocity_to_joint_velocity(np.array([0.2, 0, 0, 0, 0, 0]), state))
    check("solve returns 7 finite joint velocities", out1.shape == (7,) and np.all(np.isfinite(out1)))
    check("solve is deterministic", np.array_equal(out1, out2))
    moved = np.asarray(solver.cartesian_velocity_to_joint_velocity(np.array([0.0, 0.2, 0, 0, 0, 0]), state))
    check("different command, different solution", not np.array_equal(out1, moved))

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s): {', '.join(failures)}")
        print("=" * 78)
        return 1
    print("PASSED -- IK conversion algebra and gain preparation behave as pinned")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
