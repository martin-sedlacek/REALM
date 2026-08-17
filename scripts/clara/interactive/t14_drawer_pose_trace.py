"""WHEN does the drawer cabinet's pose stop being the one the task config asked for?

t13_drawer_stop.py measured the outcome: in scene 0 the cabinet's root link ends up at
ori ~ identity (lying on its back) instead of the config's [0.7044, 0.0616, 0.0616, 0.7044], its
drawers therefore slide vertically, and closing them drives them into the floor. In scene 1 the same
config lands exactly right. This probe finds the phase that does it.

It re-implements RealmVectorEnvironment.__init__ inline, phase by phase, and after each phase prints
every member's cabinet entity-prim pose, root-link pose, root_local, and the state of the fixed
rootJoint. It also traces every set_position_orientation() write that lands on the cabinet, with the
caller, so a phase that moves it can be attributed to a call site rather than guessed at.

    ./run python -u scripts/clara/interactive/t14_drawer_pose_trace.py --num_envs 2
"""
import argparse
import traceback


import omnigibson as og
import omnigibson.lazy as lazy  # noqa: F401 -- extension/lazy-loader side effects on import
from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import get_local_pose

from realm.environments.env_base import run_joint_resets
from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config

WATCH = "/drawer"
TRACE = True


def f3(v):
    return tuple(round(float(x), 4) for x in (v.cpu() if hasattr(v, "cpu") else v))


def _caller():
    out = []
    for fr in traceback.extract_stack()[:-2][-6:]:
        out.append(f"{fr.filename.split('/')[-1]}:{fr.lineno}:{fr.name}")
    return " < ".join(reversed(out))


def install_trace():
    xf_orig = XFormPrim.set_position_orientation
    ep_orig = EntityPrim.set_position_orientation

    def xf_patched(self, position=None, orientation=None, frame="world"):
        r = xf_orig(self, position=position, orientation=orientation, frame=frame)
        if TRACE and self.prim_path.endswith(WATCH):
            print(f"    [TRACE XFormPrim.set_po {self.prim_path} frame={frame}] "
                  f"pos={f3(position) if position is not None else None} "
                  f"ori={f3(orientation) if orientation is not None else None}\n"
                  f"        from {_caller()}", flush=True)
        return r

    def ep_patched(self, position=None, orientation=None, frame="world"):
        if TRACE and self.prim_path.endswith(WATCH):
            print(f"    [TRACE EntityPrim.set_po {self.prim_path} frame={frame}] "
                  f"pos={f3(position) if position is not None else None} "
                  f"ori={f3(orientation) if orientation is not None else None} "
                  f"stopped={og.sim.is_stopped() if og.sim else '?'}\n"
                  f"        from {_caller()}", flush=True)
        return ep_orig(self, position=position, orientation=orientation, frame=frame)

    XFormPrim.set_position_orientation = xf_patched
    EntityPrim.set_position_orientation = ep_patched


def joint_attrs(prim_path):
    prim = og.sim.stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return "rootJoint: <missing>"
    def g(n):
        a = prim.GetAttribute(n)
        return a.Get() if a else None
    b0 = prim.GetRelationship("physics:body0").GetTargets() if prim.GetRelationship("physics:body0") else None
    b1 = prim.GetRelationship("physics:body1").GetTargets() if prim.GetRelationship("physics:body1") else None
    return (f"rootJoint: body0={[str(p) for p in (b0 or [])]} body1={[str(p) for p in (b1 or [])]} "
            f"localPos0={g('physics:localPos0')} localRot0={g('physics:localRot0')} "
            f"localPos1={g('physics:localPos1')} localRot1={g('physics:localRot1')} "
            f"excludeFromArticulation={g('physics:excludeFromArticulation')} enabled={g('physics:jointEnabled')}")


def show(tag, envs):
    print(f"\n===== {tag} =====", flush=True)
    for i, env in enumerate(envs):
        scene = env.omnigibson_env.scene
        cab = scene.object_registry("name", "drawer")
        if cab is None:
            print(f"  member {i}: no 'drawer' in registry yet")
            continue
        ent = XFormPrim.get_position_orientation(cab)
        root = cab.get_position_orientation()
        try:
            rl = get_local_pose(cab.root_link.prim_path)
            rl = f"pos={f3(rl[0])} ori={f3(rl[1])}"
        except Exception as e:
            rl = f"<{type(e).__name__}: {e}>"
        lo, hi = cab.aabb
        print(f"  member {i} (scene {scene.idx}) stopped={og.sim.is_stopped()}")
        print(f"      entity(world) pos={f3(ent[0])} ori={f3(ent[1])}")
        print(f"      root  (world) pos={f3(root[0])} ori={f3(root[1])}")
        print(f"      root_local    {rl}")
        print(f"      aabb lo={f3(lo)} hi={f3(hi)}  extent={f3(hi - lo)}")
        try:
            print(f"      root_link lin_vel={f3(cab.root_link.get_linear_velocity())} "
                  f"ang_vel={f3(cab.root_link.get_angular_velocity())}")
        except Exception as e:
            print(f"      root_link vel <{type(e).__name__}>")
        print(f"      kinematic_only={cab.kinematic_only} fixed_base={cab.fixed_base} "
              f"art_root={cab.articulation_root_path} n_joints={cab.n_joints} "
              f"n_fixed_joints={getattr(cab, 'n_fixed_joints', '?')}")
        print(f"      {joint_attrs(f'{cab.prim_path}/rootJoint')}")
    print(flush=True)


def main(num_envs, task_id, robot, perturbation):
    global TRACE
    set_sim_config(robot=robot)
    install_trace()

    kwargs = dict(task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
                  perturbations=[perturbation], robot=robot)
    envs = []
    for i in range(num_envs):
        print(f"\n########## building member {i} ##########", flush=True)
        envs.append(RealmEnvironmentDynamic(in_vec_env=True, **kwargs))
        show(f"member {i} built (sim still stopped)", envs)

    TRACE = True
    og.sim.play()
    show("after og.sim.play()", envs)

    for env in envs:
        env.omnigibson_env.post_play_load()
    show("after post_play_load() (includes og.Environment.reset())", envs)

    for env in envs:
        env.bind_scene_handles()
    show("after bind_scene_handles()", envs)

    og.sim.stop()
    show("after og.sim.stop() for the scene fixes", envs)
    for env in envs:
        env.apply_scene_fixes_from_cfg(manage_sim_state=False)
    show("after apply_scene_fixes_from_cfg()", envs)
    og.sim.play()
    show("after the shared og.sim.play()", envs)

    for env in envs:
        env.rebase_initial_file()
    for env in envs:
        env.finalize_setup()
    show("after finalize_setup() (reset_joints recorded)", envs)

    run_joint_resets(envs)
    show("after run_joint_resets()", envs)

    for _ in range(30):
        og.sim.step()
    show("after 30 free steps", envs)

    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.robot, a.perturbation)
