"""Why do a drawer task's cabinet joints not reach the commanded openness?

`RealmEnvironmentBase.reset_joints()` commands the task cabinet's openable joints to their per-task
start state via `utils.reset_joints_batched` -- normalized -1.0 (fully closed) for all of them on
`open_drawer`, and the same except `INIT_OPENNESS_FRACTION` on the TARGET drawer for `close_drawer`,
whose whole point is that the robot has to push an open drawer back in. That call TELEPORTS the joint
(`JointPrim.set_pos(..., normalized=True)` with drive=False writes `set_joint_positions`) and then
steps. A joint that does not stay where it was put is therefore being pushed back out by
depenetration, not failing to be driven -- so this probe reports, per member:

  * the cabinet's root-link pose (world AND scene frame), its entity-prim pose, and the local
    transform between them -- cabinet.usd authors base_link away from the entity origin, which is
    what OG-lite 7c59ed5 is about, so the two are not interchangeable here
  * every openable joint: limits, commanded position, achieved position, residual
  * `init_openness_fraction`, the number `open_drawer`/`close_drawer` are actually scored against
  * what the cabinet's links are in CONTACT with at the stuck pose, and which bodies' AABBs
    OVERLAP the target drawer link right after it is teleported home -- i.e. the obstruction

    ./run python -u scripts/clara/interactive/t13_drawer_stop.py --num_envs 2 --task_id 8

`--dz a,b,...` re-runs the joint reset after shifting member i's cabinet by dz[i] in z. That is the
causality test for the "scene 0's cabinet sits 44 mm higher than scene 1's" lead: if where the
joints stop does not move with the cabinet, the height is a coincidence and the obstruction is
something else.
"""
import argparse

import torch as th

import omnigibson as og
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import get_local_pose

from realm.environments.contact_utils import get_impulse_contacts
from realm.environments.env_base import run_joint_resets
from realm.environments.env_vector import RealmVectorEnvironment
from realm.environments.joint_reset import INIT_OPENNESS_FRACTION
from realm.environments.utils import get_openable_joints
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config


def f3(v):
    return tuple(round(float(x), 4) for x in (v.cpu() if hasattr(v, "cpu") else v))


def expected_targets(env):
    """The normalized position each openable joint SHOULD hold after a reset, per TASK TYPE.

    `close_drawer` is the one task that does not start fully closed: its target drawer starts at
    INIT_OPENNESS_FRACTION, because the robot's job is to push it back in. Reading that off the env
    instead of assuming -1.0 for everything is what lets one probe judge both drawer tasks -- the
    hardcoded -1.0 this replaced called a CORRECT task 9 reset "DID NOT REACH", and called the
    actual defect (task 9 starting closed) "OK -- every joint home".
    """
    init_open = env.task_type == "close_drawer"
    return {
        j.joint_name: (INIT_OPENNESS_FRACTION if (init_open and j is env.mo_joint) else -1.0)
        for j in get_openable_joints(env.main_objects[0])
    }


def pose_block(env, i):
    cabinet = env.main_objects[0]
    scene = env.omnigibson_env.scene
    scene_pos = scene._pose_info["pos_ori"][0]
    root_w = cabinet.get_position_orientation()
    root_s = cabinet.get_position_orientation(frame="scene")
    ent_w = XFormPrim.get_position_orientation(cabinet)
    root_local = get_local_pose(cabinet.root_link.prim_path)
    print(f"  member {i} (scene {scene.idx}) cabinet={cabinet.name!r} prim={cabinet.prim_path}")
    print(f"      scene prim pos      {f3(scene_pos)}")
    print(f"      root link  (world)  pos={f3(root_w[0])} ori={f3(root_w[1])}")
    print(f"      root link  (scene)  pos={f3(root_s[0])} ori={f3(root_s[1])}")
    print(f"      entity prim (world) pos={f3(ent_w[0])} ori={f3(ent_w[1])}")
    print(f"      root_local (base_link rel. entity prim) pos={f3(root_local[0])} ori={f3(root_local[1])}")
    print(f"      root_link_name={cabinet.root_link_name!r}  fixed_base={cabinet.fixed_base}  "
          f"scale={f3(cabinet.scale)}")
    aabb_lo, aabb_hi = cabinet.aabb
    print(f"      cabinet aabb        lo={f3(aabb_lo)} hi={f3(aabb_hi)}")


def drive_state(j):
    """This joint's position/velocity target and its PhysX drive gains, or why they are unreadable.

    Wrapped because both reads go through the articulation view, which raises rather than returning
    a sentinel when the physics handle is not live -- and a probe that dies reporting the diagnosis
    is worse than one that prints the reason it cannot.
    """
    try:
        tpos, tvel = j.get_target()
        tgt = f"pos={float(tpos[0]):+.4f} vel={float(tvel[0]):+.4f}"
    except Exception as e:
        tgt = f"<unreadable: {type(e).__name__}>"
    try:
        kps, kds = j._articulation_view.get_gains(joint_indices=j.dof_indices)
        tgt += f" kp={float(kps[0][0]):.1f} kd={float(kds[0][0]):.1f}"
    except Exception as e:
        tgt += f" gains=<unreadable: {type(e).__name__}>"
    return tgt


def joint_block(env, i, commanded=None):
    """Per-joint commanded vs achieved. @commanded maps joint name -> normalized target."""
    cabinet = env.main_objects[0]
    joints = get_openable_joints(cabinet)
    print(f"  member {i}: {len(joints)} openable joints; target={env.mo_joint.joint_name!r}")
    rows = []
    for j in joints:
        lo, hi = j.lower_limit, j.upper_limit
        pos, vel, _ = j.get_state()
        pos = float(pos[0])
        npos = 2.0 * (pos - lo) / (hi - lo) - 1.0
        tgt_n = -1.0 if commanded is None else commanded.get(j.joint_name, -1.0)
        tgt = (tgt_n + 1.0) / 2.0 * (hi - lo) + lo
        mark = "  <== TARGET" if j is env.mo_joint else ""
        star = "" if abs(npos - tgt_n) < 1e-3 else "  ** DID NOT REACH **"
        print(f"      {j.joint_name:<12s} [{j.joint_type:<16s}] limits=[{lo:+.4f},{hi:+.4f}] "
              f"cmd={tgt:+.4f}({tgt_n:+.2f}) got={pos:+.4f}({npos:+.4f}) "
              f"resid={pos - tgt:+.4f} vel={float(vel[0]):+.4f}{mark}{star}")
        # The DRIVE state, not just the position. OG 3.9.1 only writes a joint's position TARGET
        # when JointPrim.driven is True, and driven is `HasAPI(DriveAPI) and load_config["driven"]`
        # -- and EntityPrim.is_driven is hardcoded False, so a cabinet's joints are never "driven"
        # even when the USD authors a drive. OG 1.1.1's set_pos wrote the target unconditionally.
        # If a teleported-open drawer is being pulled shut, a stale target plus a live stiffness is
        # how, so print both rather than inferring.
        print(f"        {'':<12s} driven={j.driven} control_type={j._control_type} "
              f"target={drive_state(j)}")
        rows.append((j.joint_name, pos, npos, tgt, tgt_n))
    print(f"      init_openness_fraction={float(env.init_openness_fraction):.4f} "
          f"joint_range={float(env.joint_range):.4f}")
    return rows


def contact_block(env, i):
    cabinet = env.main_objects[0]
    scene_idx = env.omnigibson_env.scene.idx
    links = list(cabinet.links.values())
    # threshold 0: we want every reported contact, including resting ones -- a depenetration
    # contact that has already been resolved carries almost no impulse but is the whole story.
    contacts = get_impulse_contacts(scene_idx, links, impulse_threshold=0.0)
    print(f"  member {i} contacts on cabinet links:")
    if not contacts:
        print("      (none reported)")
    for path in sorted(contacts):
        others = sorted(contacts[path])
        print(f"      {path.split('/')[-1]:<16s} -> {others}")


def overlap_block(env, i, link):
    """Every body in the member's scene whose AABB intersects @link's, right now."""
    scene = env.omnigibson_env.scene
    lo, hi = link.aabb
    hits = []
    for obj in scene.objects:
        for lname, l in obj.links.items():
            if l.prim_path == link.prim_path:
                continue
            try:
                olo, ohi = l.aabb
            except Exception:
                continue
            if bool(th.all(lo < ohi) and th.all(olo < hi)):
                inter = th.minimum(hi, ohi) - th.maximum(lo, olo)
                hits.append((float(th.min(inter)), obj.name, lname, f3(inter)))
    robot = env.robot
    for lname, l in robot.links.items():
        olo, ohi = l.aabb
        if bool(th.all(lo < ohi) and th.all(olo < hi)):
            inter = th.minimum(hi, ohi) - th.maximum(lo, olo)
            hits.append((float(th.min(inter)), robot.name, lname, f3(inter)))
    print(f"  member {i} AABB overlaps with {link.prim_path.split('/')[-1]} "
          f"(lo={f3(lo)} hi={f3(hi)}):")
    if not hits:
        print("      (none)")
    for depth, oname, lname, inter in sorted(hits, reverse=True):
        print(f"      {oname}/{lname:<20s} min_penetration={depth:+.4f} overlap_extent={inter}")


def report(tag, vec_env):
    print(f"\n########## {tag} ##########", flush=True)
    for i, env in enumerate(vec_env.envs):
        pose_block(env, i)
    print()
    for i, env in enumerate(vec_env.envs):
        joint_block(env, i, commanded=expected_targets(env))
    print()
    for i, env in enumerate(vec_env.envs):
        contact_block(env, i)
    print(flush=True)


def teleport_home_and_look(vec_env):
    """Teleport every openable joint home, look at the overlaps BEFORE physics resolves them."""
    print("\n########## teleport to the task's init state, pre-step overlap ##########", flush=True)
    for i, env in enumerate(vec_env.envs):
        cabinet = env.main_objects[0]
        targets = expected_targets(env)
        for j in get_openable_joints(cabinet):
            # The TASK's init state, not a hardcoded -1.0: on close_drawer the obstruction that
            # matters is the one a FULLY OPEN drawer runs into, and -1.0 never puts it there.
            j.set_pos(targets[j.joint_name], normalized=True)
            j.set_vel(0)
    og.sim.render()  # flush the physx->fabric sync so the AABBs below are the teleported ones
    for i, env in enumerate(vec_env.envs):
        cabinet = env.main_objects[0]
        joints = get_openable_joints(cabinet)
        print(f"  -- member {i} right after teleport, before any step:")
        for j in joints:
            pos, _, _ = j.get_state()
            print(f"      {j.joint_name:<12s} got={float(pos[0]):+.4f}")
        tgt_link = cabinet.links[env.mo_joint.body1.split("/")[-1]]
        overlap_block(env, i, tgt_link)
    print(flush=True)


def hold_and_watch(vec_env, steps, every=25):
    """Free-run @steps sim steps and trace each member's openness, as a rollout's first steps would.

    `init_openness_fraction` is captured INSIDE the reset, before the settle loop, but a frame is
    rendered -- and the policy's first actions are taken -- many steps later. A drawer that is open
    at reset and drifts shut on its own before then invalidates the eval cell just as thoroughly as
    one that never opened, while leaving init_openness_fraction reading a perfect 1.0. Distinguishing
    those two is the whole point of this loop, so it reports the SCORED quantity
    (get_mo_joint_openness_fraction) next to the reference and the delta the rubric thresholds on.
    """
    print(f"\n########## hold: {steps} free sim steps after the reset ##########", flush=True)
    print("  step | " + " | ".join(f"m{i} openness (delta)" for i in range(len(vec_env.envs))),
          flush=True)

    def row(k):
        cells = []
        for env in vec_env.envs:
            frac = float(env.get_mo_joint_openness_fraction())
            cells.append(f"{frac:.4f} ({float(env.init_openness_fraction) - frac:+.4f})")
        print(f"  {k:>5d} | " + " | ".join(cells), flush=True)

    row(0)
    # No render: nothing here reads a camera, and the robot holds its last position targets, so it
    # does not collapse. This is the drawer's own behaviour under an idle policy.
    with og.sim.render_on_step(False):
        for k in range(1, steps + 1):
            og.sim.step()
            if k % every == 0 or k == steps:
                row(k)
    print(flush=True)


def slide_axis_test(vec_env):
    """Does the target drawer LINK actually translate when its joint does, and along which axis?

    A joint position that reads 0.30 m is not the same claim as a drawer that has moved 0.30 m: the
    joint value is a DOF reading, while what a camera photographs -- and what the 1.1.1 reference
    frame shows engulfing the exterior camera -- is the link's pose. So teleport the target joint to
    both ends of its range and difference the link's AABB centre. Expect |delta| ~= the joint range
    along one horizontal axis. A near-zero delta means the link is not following the joint at all; a
    delta along z means the drawer slides vertically, which is the up-axis signature confined to the
    drawer link rather than the whole cabinet.
    """
    print("\n########## slide-axis test: link displacement per unit of joint travel ##########",
          flush=True)
    for i, env in enumerate(vec_env.envs):
        cabinet = env.main_objects[0]
        j = env.mo_joint
        link = cabinet.links[j.body1.split("/")[-1]]
        centres = {}
        for tag, npos in (("closed(-1)", -1.0), ("open(+1)", 1.0)):
            j.set_pos(npos, normalized=True)
            j.set_vel(0)
            # render() flushes the physx->fabric sync; without it the AABB read below is the
            # PREVIOUS pose and the delta comes out zero for a purely bookkeeping reason.
            og.sim.render()
            lo, hi = link.aabb
            centres[tag] = (lo + hi) / 2.0
            print(f"  member {i} {tag:<10s} joint={float(j.get_state()[0][0]):+.4f} "
                  f"link_aabb lo={f3(lo)} hi={f3(hi)} extent={f3(hi - lo)}")
        d = centres["open(+1)"] - centres["closed(-1)"]
        rng = float(j.upper_limit - j.lower_limit)
        print(f"  member {i} link centre delta={f3(d)} |delta|={float(th.linalg.norm(d)):.4f} "
              f"(joint range {rng:.4f}) link={link.prim_path.split('/')[-1]}")
        axis = int(th.argmax(th.abs(d)))
        print(f"  member {i} dominant axis={'xyz'[axis]} "
              f"{'** LINK DOES NOT FOLLOW THE JOINT **' if float(th.linalg.norm(d)) < 0.5 * rng else ('** SLIDES VERTICALLY **' if axis == 2 else 'horizontal slide, as expected')}")
    print(flush=True)


def main(num_envs, task_id, robot, perturbation, dz, resets, hold_steps):
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[perturbation],
        robot=robot,
    )
    report("after construction (reset_joints has run once)", vec_env)
    slide_axis_test(vec_env)
    # slide_axis_test leaves the target joint wherever its last teleport put it.
    for env in vec_env.envs:
        env.reset_joints()
    run_joint_resets(vec_env.envs)

    if hold_steps:
        hold_and_watch(vec_env, hold_steps)
        report(f"after {hold_steps} free steps", vec_env)
        # Back to a defined start state before anything below runs.
        for env in vec_env.envs:
            env.reset_joints()
        run_joint_resets(vec_env.envs)

    for r in range(resets):
        vec_env.reset()
        report(f"after vec_env.reset() #{r + 1}", vec_env)

    teleport_home_and_look(vec_env)
    # Put the cabinets back to a settled state before the dz experiment.
    for env in vec_env.envs:
        env.reset_joints()
    run_joint_resets(vec_env.envs)
    report("after a re-driven reset_joints()", vec_env)

    if dz:
        print(f"\n########## dz experiment: {dz} ##########", flush=True)
        for i, env in enumerate(vec_env.envs):
            if i >= len(dz) or dz[i] == 0.0:
                continue
            cabinet = env.main_objects[0]
            pos, ori = cabinet.get_position_orientation(frame="scene")
            want = pos.clone()
            want[2] += dz[i]
            cabinet.set_position_orientation(position=want, orientation=ori, frame="scene")
            got = cabinet.get_position_orientation(frame="scene")[0]
            print(f"  member {i}: asked scene z {float(pos[2]):+.4f} -> {float(want[2]):+.4f}, "
                  f"read back {float(got[2]):+.4f} "
                  f"({'MOVED' if abs(float(got[2]) - float(want[2])) < 1e-3 else 'DID NOT TAKE'})")
        for _ in range(5):
            og.sim.step()
        for env in vec_env.envs:
            env.reset_joints()
        run_joint_resets(vec_env.envs)
        report(f"after dz={dz} + reset_joints()", vec_env)

    print("\n########## VERDICT ##########", flush=True)
    for i, env in enumerate(vec_env.envs):
        cabinet = env.main_objects[0]
        targets = expected_targets(env)
        # What init_openness_fraction must read for THIS task: 1.0 where close_drawer starts its
        # target drawer open, 0.0 where open_drawer starts it shut. The value the progression
        # stages are scored against, so a wrong one silently invalidates the whole eval cell.
        want_frac = (INIT_OPENNESS_FRACTION + 1.0) / 2.0 if env.task_type == "close_drawer" else 0.0
        bad = []
        for j in get_openable_joints(cabinet):
            lo, hi = j.lower_limit, j.upper_limit
            want = (targets[j.joint_name] + 1.0) / 2.0 * (hi - lo) + lo
            pos = float(j.get_state()[0][0])
            if abs(pos - want) > 1e-3:
                bad.append(f"{j.joint_name}={pos:.4f}(want={want:.4f})")
        got_frac = float(env.init_openness_fraction)
        frac_ok = abs(got_frac - want_frac) < 1e-3
        print(f"  member {i} (scene {env.omnigibson_env.scene.idx}) {env.task_type}: "
              f"init_openness_fraction={got_frac:.4f} (want {want_frac:.4f}) "
              f"{'OK' if frac_ok else '** WRONG START STATE **'}; "
              f"{'every joint at its task target' if not bad else 'OFF TARGET: ' + ', '.join(bad)}")
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    p.add_argument("--resets", type=int, default=0)
    p.add_argument("--dz", type=str, default="", help="comma-separated per-member z shift, e.g. -0.044,0.044")
    p.add_argument("--hold_steps", type=int, default=0,
                   help="free-run this many sim steps after the reset and trace the openness. Use "
                        "~300 to cover a render probe's pre-render budget: it separates 'the drawer "
                        "never opened' from 'it opened and then drifted shut before the frame'.")
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.robot, a.perturbation,
         [float(x) for x in a.dz.split(",")] if a.dz else [], a.resets, a.hold_steps)
