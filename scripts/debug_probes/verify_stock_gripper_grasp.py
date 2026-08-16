"""Prove the STOCK DROID gripper closes, bites, and registers a grasp -- end to end, in one probe.

Drop into the pump's inbox (see README.md) with `REALM_ROBOT=DROID`. Takes ~15 s of sim.

WHY THIS EXISTS
---------------
"Stock DROID is broken -- every rollout ends at REACH and the gripper never closes" has been raised
more than once, and each time the evidence was a run under `--model_type debug`. That client returns
a hardcoded `np.zeros(8)` (realm/inference/client.py:33), whose last element binarises to gripper
= -1 = OPEN, and whose first seven ask the arm to hold a fixed pose nowhere near the object. So the
gripper is held open BY CONSTRUCTION and no success condition can fire. tests/test_integrity.py says
exactly this in its own docstring. A REACH-only result under `debug` is not evidence about the asset.

This probe is the counter-evidence, and it is cheap enough that nobody should ever have to re-derive
it. Measured 2026-08-16, job 191495, MODE=oglite, `droid_mounted` (which is what
realm/config/robots/DROID.yaml resolves to on a REALM_DROID10 task -- see env_config.py:130):

    gripper DOF limits            prismatic [0, 0.05] m,   revolute [0, 0.7854] rad
    gripper_control_idx           [8, 7, 10, 9]     (NOT arange(7, 11) -- see droid_mounted.yaml)
    REALM action -1 ("close")     -> [0, 0, 0, 0]                  -> jaw 80.1 mm, pads apart
    REALM action +1 ("open")      -> [.0485, .0485, .7614, .7614]  -> jaw  7.1 mm, pads in contact
    +1 with a 30 mm block held between the pads
                                  -> arrests at 22.9 mm, 2/2 pads on the block,
                                     is_touching True, is_grasping True,
                                     progression 0.200 -> 0.600 (REACH, GRASP, LIFT_SLIGHT),
                                     block lifted +0.262 m and held for 80 steps

THE NAMES LIE. `rollout.binarize_gripper` calls +1 "open" and the controller's branch is called
`should_open`, but REALM's +1 is CLOSE for both DROID assets -- DROID_robolab_v2.yaml records the
same inversion of the same identifier. Do not "fix" it by adding `inverted: true`; that would send
the close command to the joints' LOWER limit, which is the open jaw.

MEASURE THE PADS, NOT THE LINKAGE. The direction check below is `pads in mutual contact`, which is
geometry-free. Link-origin separation agrees for this asset (the two inner_finger origins sit
133.3 mm from panda_link8, i.e. on the pads) but that is the exact measurement that read BACKWARDS
on robolab and inverted a full batch on 2026-08-11 -- a four-bar linkage swings its knuckles apart
as the pads close. Both observables are printed so a disagreement is visible rather than silent.
"""

# Jaw separation (mm) below which the pads count as shut, for the link-origin cross-check only.
SHUT_MM = 20.0
OPEN_MM = 60.0

PASS_ALL = True


def _fail(msg):
    global PASS_ALL
    PASS_ALL = False
    print(f"  FAIL: {msg}")


def verify_stock_gripper(close_steps=40, lift_steps=60):
    import numpy as np
    from omnigibson.utils.usd_utils import RigidContactAPI

    r = robot  # noqa: F821
    e = env  # noqa: F821
    global obs, PASS_ALL  # noqa: F824
    fl = list(e.robot_finger_links)
    mo = e.main_objects[0]
    gidx = np.asarray(r.gripper_control_idx[r.default_arm])

    def _np(x):
        return np.asarray(x.cpu() if hasattr(x, "cpu") else x)

    def midpoint():
        return 0.5 * (_np(fl[0].get_position_orientation()[0])
                      + _np(fl[1].get_position_orientation()[0]))

    def sep_mm():
        return 1000.0 * float(np.linalg.norm(
            _np(fl[0].get_position_orientation()[0]) - _np(fl[1].get_position_orientation()[0])))

    def pads_touch():
        return len(RigidContactAPI.get_contact_pairs(
            scene_idx=r.scene.idx, query_set={fl[0]}, with_set={fl[1]}, current_only=True)) > 0

    def n_obj_contacts():
        return len({f for _, f in RigidContactAPI.get_contact_pairs(
            scene_idx=mo.scene.idx, query_set={mo}, with_set=e.robot_finger_links,
            current_only=True)})

    print(f"robot={r.name}  prim={r.prim_path}  n_dof={r.n_dof} n_joints={r.n_joints}")
    print(f"gripper_control_idx={gidx.tolist()}  finger_joints={r.finger_joint_names[r.default_arm]}")
    print(f"finger_links={[l.name for l in fl]}")
    l8 = _np(r.links["panda_link8"].get_position_orientation()[0])
    d8 = [1000.0 * float(np.linalg.norm(_np(l.get_position_orientation()[0]) - l8)) for l in fl]
    print(f"finger origin -> panda_link8: {np.round(d8, 1).tolist()} mm "
          f"({'on the pads' if min(d8) > 50 else 'AT THE MOUNT -- separation is meaningless'})")
    if r.n_dof != r.n_joints:
        _fail(f"n_dof {r.n_dof} != n_joints {r.n_joints}: a loop-closing joint gained a DOF, so "
              f"gripper_control_idx no longer indexes what the definition thinks it does")

    obs, _ = e.reset()
    obs, _, _, _, _ = e.warmup(obs)
    q = _np(r.get_joint_positions()).copy()
    arm = np.asarray(q[:7], dtype=np.float32)

    # ---- 1. direction -------------------------------------------------------------------------
    print("\n[1] which command shuts the jaw")
    state = {}
    for label, cmd in (("-1", -1.0), ("+1", +1.0)):
        for _ in range(close_steps):
            obs, _, _, _, _ = e.step(np.concatenate([arm, [cmd]]).astype(np.float32))
        state[label] = (sep_mm(), pads_touch(), _np(r.get_joint_positions())[gidx].copy())
        print(f"  action {label}: qpos={np.round(state[label][2], 4).tolist()} "
              f"sep={state[label][0]:.1f} mm  pads_touching={state[label][1]}")
    if not state["+1"][1]:
        _fail("REALM's +1 (CLOSE) did not bring the pads into mutual contact")
    if state["-1"][1]:
        _fail("REALM's -1 (OPEN) left the pads in contact")
    if not (state["+1"][0] < SHUT_MM < OPEN_MM < state["-1"][0]):
        _fail(f"link-origin separation disagrees with the contact test: "
              f"+1 -> {state['+1'][0]:.1f} mm, -1 -> {state['-1'][0]:.1f} mm. One of the two "
              f"observables is lying; believe the contact test and re-check the link origins.")

    # ---- 2. grasp -----------------------------------------------------------------------------
    # Hold the block at the jaw midpoint until both pads bite, then hand physics back. Without this
    # it simply falls: after warmup the jaws are ~46 cm above the table.
    print("\n[2] grasp on a held block")
    for _ in range(20):
        obs, _, _, _, _ = e.step(np.concatenate([arm, [-1.0]]).astype(np.float32))
    _, mq = mo.get_position_orientation()
    close_act = np.concatenate([arm, [+1.0]]).astype(np.float32)
    held = True
    for _ in range(close_steps + 20):
        if held:
            mo.set_position_orientation(position=midpoint(), orientation=mq)
        obs, prog, _, _, _ = e.step(close_act)
        if held and n_obj_contacts() == 2:
            held = False
    e.capture_mo_reference()
    nc, touch, grasp = n_obj_contacts(), bool(e.is_touching(obs, mo)), bool(e.is_grasping(obs, mo))
    print(f"  jaw arrested at {sep_mm():.1f} mm on a "
          f"{1000*float(_np(mo.aabb_extent)[0]):.0f} mm object")
    print(f"  pads on object={nc}/2  is_touching={touch}  is_grasping={grasp}  progression={prog:.3f}")
    print(f"  stages={[k for k, v in e.task_progression.items() if v]}")
    if nc != 2:
        _fail(f"only {nc}/2 pads on the object -- is_grasping needs exactly two")
    if not grasp:
        _fail("is_grasping did not fire with both pads on the object")

    # ---- 3. lift ------------------------------------------------------------------------------
    print("\n[3] lift")
    lift = arm.copy()
    lift[1] -= 0.30
    lift[3] += 0.30
    z0 = float(_np(mo.get_position_orientation()[0])[2])
    for _ in range(lift_steps):
        obs, prog, _, _, _ = e.step(np.concatenate([lift, [+1.0]]).astype(np.float32))
    dz = float(_np(mo.get_position_orientation()[0])[2]) - z0
    stages = [k for k, v in e.task_progression.items() if v]
    print(f"  block dz={dz:+.4f} m  still grasped={bool(e.is_grasping(obs, mo))}  "
          f"progression={prog:.3f}  stages={stages}")
    if dz < 0.02:
        _fail(f"block did not come off the table (dz={dz:+.4f} m)")
    if "GRASP" not in stages:
        _fail("GRASP never latched in the task progression")

    print(f"\n{'STOCK_GRIPPER_OK' if PASS_ALL else 'STOCK_GRIPPER_FAILED'}")
    return PASS_ALL


verify_stock_gripper()
