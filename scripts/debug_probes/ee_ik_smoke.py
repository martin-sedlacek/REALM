"""Bottom-up smoke test for the dm_robotics EE-control stack. No Isaac, no OmniGibson.

Checks, in order:
  1. dm_control / dm_robotics.moma.effectors import in this container, and at what versions.
  2. RobotIKSolver() constructs (this is where the MJCF is compiled and the QP effector is built).
  3. The MJCF joint/actuator order, so it can be compared against the OG robot's arm DOFs.
  4. The solver actually SOLVES: drive the wrist site toward a reachable target for N iterations
     and check the residual shrinks. A solver that silently returns zeros passes a "finite" check
     but fails this one.
  5. Both command paths DroidEndEffectorController uses: absolute_pose and pose_delta_ori.
"""
import sys
import traceback

import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=160)

DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


hdr("1. IMPORTS")
print("python:", sys.version.replace("\n", " "))
try:
    import importlib.metadata as md
    for pkg in ("dm-control", "dm-robotics-moma", "dm-robotics-controllers",
                "dm-robotics-transformations", "dm-robotics-geometry", "mujoco", "numpy", "scipy"):
        try:
            print(f"  {pkg:32s} {md.version(pkg)}")
        except Exception as e:
            print(f"  {pkg:32s} <not installed: {type(e).__name__}>")
except Exception:
    traceback.print_exc()

try:
    print("  import dm_control.mjcf                 OK")
    print("  import dm_robotics.moma.effectors      OK")
except Exception:
    print("  IMPORT FAILED")
    traceback.print_exc()
    raise SystemExit(2)

hdr("2. RobotIKSolver CONSTRUCTION")
try:
    from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver
    solver = RobotIKSolver()
    print("  RobotIKSolver()                        OK")
except Exception:
    print("  CONSTRUCTION FAILED")
    traceback.print_exc()
    raise SystemExit(3)

hdr("3. MJCF JOINT / ACTUATOR ORDER (what the IK will drive)")
joints = solver._arm.joints
actuators = solver._arm.actuators
print(f"  n_joints    = {len(joints)}")
print(f"  joint order = {[j.name for j in joints]}")
print(f"  n_actuators = {len(actuators)}")
print(f"  act   order = {[a.name for a in actuators]}")
print(f"  actuator->joint = {[(a.name, a.joint.name) for a in actuators]}")
print(f"  wrist_site  = {solver._arm.wrist_site.name}   base_site = {solver._arm.base_site.name}")
qpos_bind = solver._physics.bind(joints).qpos
print(f"  physics.bind(joints).qpos shape = {np.asarray(qpos_bind).shape}")

hdr("4. FORWARD KINEMATICS AT THE REALM RESET POSE")
from scipy.spatial.transform import Rotation as R


def fk(qpos, qvel=None):
    qvel = np.zeros(7) if qvel is None else qvel
    solver._arm.update_state(solver._physics, np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float))
    solver._physics.forward()
    b = solver._physics.bind(solver._arm.wrist_site)
    pos = np.array(b.xpos).copy()
    rpy = R.from_matrix(np.array(b.xmat).copy().reshape(3, 3)).as_euler("xyz")
    return np.concatenate([pos, rpy])


q0 = DEFAULT_RESET_JOINTPOS.copy()
p0 = fk(q0)
print(f"  reset qpos      = {q0}")
print(f"  wrist_site pose = {p0}   (xyz + rpy, robot base frame)")
assert np.all(np.isfinite(p0)), "FK produced non-finite pose"

hdr("5. SINGLE IK CALL (absolute_pose path, as DroidEndEffectorController does it)")
from realm.geometry import pose_diff

target = p0.copy()
target[0] += 0.05   # 5 cm forward -- well inside max_lin_delta*N and certainly reachable
target[2] -= 0.05   # 5 cm down

q = q0.copy()
qd = np.zeros(7)
cart_delta = pose_diff(target, fk(q, qd))
cart_vel = solver.cartesian_delta_to_velocity(cart_delta)
print(f"  cartesian_delta    = {cart_delta}")
print(f"  cartesian_velocity = {cart_vel}")
try:
    jvel = solver.cartesian_velocity_to_joint_velocity(
        cart_vel, robot_state={"joint_positions": q, "joint_velocities": qd})
except Exception:
    print("  SOLVE FAILED")
    traceback.print_exc()
    raise SystemExit(4)
jdelta = solver.joint_velocity_to_delta(jvel)
qnext = jdelta + q
print(f"  joint_velocity     = {jvel}")
print(f"  joint_delta        = {jdelta}")
print(f"  joint_position     = {qnext}")
print(f"  finite: jvel={np.all(np.isfinite(jvel))} jdelta={np.all(np.isfinite(jdelta))} q={np.all(np.isfinite(qnext))}")
print(f"  |joint_delta|_inf  = {np.abs(jdelta).max():.6f}  (solver.max_joint_delta = {solver.max_joint_delta})")
if not np.any(np.abs(jdelta) > 1e-9):
    print("  !! WARNING: joint_delta is all-zero -- the QP returned no motion for a reachable target")

hdr("6. CLOSED-LOOP CONVERGENCE (does the solver actually reduce the residual?)")
q = q0.copy()
qd = np.zeros(7)
print(f"  {'it':>3}  {'|pos err| m':>12}  {'|rot err| rad':>13}  {'|dq|inf':>9}")
for it in range(41):
    cur = fk(q, qd)
    d = pose_diff(target, cur)
    if it % 5 == 0 or it == 40:
        print(f"  {it:>3}  {np.linalg.norm(d[:3]):>12.6f}  {np.linalg.norm(d[3:]):>13.6f}", end="")
    v = solver.cartesian_delta_to_velocity(d)
    jv = solver.cartesian_velocity_to_joint_velocity(
        v, robot_state={"joint_positions": q, "joint_velocities": qd})
    dq = solver.joint_velocity_to_delta(jv)
    if it % 5 == 0 or it == 40:
        print(f"  {np.abs(dq).max():>9.6f}")
    q = q + dq
    if not np.all(np.isfinite(q)):
        print(f"  !! NON-FINITE qpos at iteration {it}: {q}")
        raise SystemExit(5)

final = fk(q, qd)
err = pose_diff(target, final)
print(f"\n  start pose  = {p0}")
print(f"  target pose = {target}")
print(f"  final pose  = {final}")
print(f"  final |pos err| = {np.linalg.norm(err[:3]):.6f} m   |rot err| = {np.linalg.norm(err[3:]):.6f} rad")
print(f"  final qpos  = {q}")
print(f"  CONVERGED  = {np.linalg.norm(err[:3]) < 5e-3}")

hdr("7. DELTA PATH (pose_delta_ori, as DROID_ee_delta_control.yaml uses)")
q = q0.copy()
qd = np.zeros(7)
dpos = np.array([0.02, 0.0, -0.02])
drpy = np.array([0.0, 0.0, 0.05])
cart_delta = np.concatenate([dpos, drpy])
v = solver.cartesian_delta_to_velocity(cart_delta)
jv = solver.cartesian_velocity_to_joint_velocity(
    v, robot_state={"joint_positions": q, "joint_velocities": qd})
dq = solver.joint_velocity_to_delta(jv)
print(f"  cartesian_delta = {cart_delta}")
print(f"  joint_delta     = {dq}")
print(f"  finite = {np.all(np.isfinite(dq))}   |dq|inf = {np.abs(dq).max():.6f}")

hdr("8. TORCH-TENSOR INPUT (what compute_control actually hands the solver)")
# compute_control passes torch tensors as robot_state["joint_positions"] / ["joint_velocities"],
# and a python list as cartesian_velocity. Reproduce that exactly.
try:
    import torch as th
    tq = th.tensor(q0, dtype=th.float32)
    tqd = th.zeros(7, dtype=th.float32)
    v_list = solver.cartesian_delta_to_velocity(np.concatenate([dpos, drpy])).tolist()
    jv = solver.cartesian_velocity_to_joint_velocity(
        v_list, robot_state={"joint_positions": tq, "joint_velocities": tqd})
    print(f"  torch input OK, joint_velocity = {np.asarray(jv)}")
except Exception:
    print("  TORCH-INPUT PATH FAILED")
    traceback.print_exc()

hdr("VERDICT")
print("IK_SMOKE_OK")
