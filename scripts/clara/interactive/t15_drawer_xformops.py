"""What is the extra Rx(90) that scene 0's cabinet prim carries and scene 1's does not?

t14_drawer_pose_trace.py traced `_set_xform_properties` reading the SAME asset as

    scene 0:  pos=(0.0,     0.8098, 0.0)  ori=(0.7071, 0, 0, 0.7071)   <- Rx(+90)
    scene 1:  pos=(25.2472, 0.8098, 0.0)  ori=(0, 0, 0, 1)             <- identity

and every later pose write on scene 0's prim coming back as `requested ∘ Rx(90)`. This dumps the
prim's actual transform state -- the xformOp order, every xformOp attribute with its type, the USD
local transform, the Fabric local/world transform, and the prim stack -- for both members, so the
extra rotation can be attributed to an op rather than inferred from the residual.

Builds the members and stops; never plays. ~1 scene load per member.

    ./run python -u scripts/clara/interactive/t15_drawer_xformops.py --num_envs 2
"""
import argparse


import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import get_local_pose, get_world_pose

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config


def f(v):
    return tuple(round(float(x), 4) for x in (v.cpu() if hasattr(v, "cpu") else v))


def dump(prim_path, label):
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    print(f"\n  ---- {label}: {prim_path} ----")
    if not prim.IsValid():
        print("      <invalid prim>")
        return
    xf = lazy.pxr.UsdGeom.Xformable(prim)
    order = xf.GetXformOpOrderAttr().Get()
    print(f"      xformOpOrder = {list(order) if order else None}")
    for name in sorted(n for n in prim.GetPropertyNames() if n.startswith("xformOp")):
        attr = prim.GetAttribute(name)
        print(f"      {name:<45s} type={attr.GetTypeName()} value={attr.Get()}")
    local = xf.GetLocalTransformation()
    print(f"      USD GetLocalTransformation():")
    for i in range(4):
        print(f"          {tuple(round(float(v), 5) for v in local.GetRow(i))}")
    print(f"      Fabric get_local_pose:  pos={f(get_local_pose(prim_path)[0])} "
          f"ori={f(get_local_pose(prim_path)[1])}")
    print(f"      Fabric get_world_pose:  pos={f(get_world_pose(prim_path)[0])} "
          f"ori={f(get_world_pose(prim_path)[1])}")
    print(f"      prim stack (strongest first):")
    for spec in prim.GetPrimStack():
        print(f"          {spec.layer.identifier}   path={spec.path}")


def main(num_envs, task_id, robot, perturbation):
    set_sim_config(robot=robot)
    kwargs = dict(task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
                  perturbations=[perturbation], robot=robot)
    envs = []
    for i in range(num_envs):
        envs.append(RealmEnvironmentDynamic(in_vec_env=True, **kwargs))

    print("\n########## cabinet prim transform state, per member ##########", flush=True)
    for i, env in enumerate(envs):
        scene = env.omnigibson_env.scene
        dump(f"{scene.prim_path}", f"member {i} SCENE prim")
        dump(f"{scene.prim_path}/drawer", f"member {i} cabinet ENTITY prim")
        dump(f"{scene.prim_path}/drawer/base_link", f"member {i} cabinet base_link")

    print("\n########## the asset on its own ##########", flush=True)
    for path in ("/app/custom_assets/impact_drawer/usd/cabinet.usd",):
        st = lazy.pxr.Usd.Stage.Open(path)
        dp = st.GetDefaultPrim()
        print(f"  {path}")
        print(f"      upAxis={lazy.pxr.UsdGeom.GetStageUpAxis(st)} "
              f"metersPerUnit={lazy.pxr.UsdGeom.GetStageMetersPerUnit(st)} defaultPrim={dp.GetPath()}")
        xf = lazy.pxr.UsdGeom.Xformable(dp)
        order = xf.GetXformOpOrderAttr().Get()
        print(f"      defaultPrim xformOpOrder = {list(order) if order else None}")
        for name in sorted(n for n in dp.GetPropertyNames() if n.startswith("xformOp")):
            a = dp.GetAttribute(name)
            print(f"      {name:<45s} type={a.GetTypeName()} value={a.Get()}")
        bl = st.GetPrimAtPath(f"{dp.GetPath()}/base_link")
        if bl.IsValid():
            xfb = lazy.pxr.UsdGeom.Xformable(bl)
            ob = xfb.GetXformOpOrderAttr().Get()
            print(f"      base_link xformOpOrder = {list(ob) if ob else None}")
            for name in sorted(n for n in bl.GetPropertyNames() if n.startswith("xformOp")):
                a = bl.GetAttribute(name)
                print(f"      base_link {name:<35s} type={a.GetTypeName()} value={a.Get()}")

    print(f"\n  STAGE upAxis={lazy.pxr.UsdGeom.GetStageUpAxis(og.sim.stage)} "
          f"metersPerUnit={lazy.pxr.UsdGeom.GetStageMetersPerUnit(og.sim.stage)}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.robot, a.perturbation)
