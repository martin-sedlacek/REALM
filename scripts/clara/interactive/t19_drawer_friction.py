"""What friction does the drawer's collision geometry actually have at runtime?

`custom_assets/impact_drawer/usd/cabinet.usd` binds `/cabinet/Materials/DummyPhysicsMaterial` -- which
does carry `PhysicsMaterialAPI` -- to 35 collision meshes, 7 per drawer link across all five drawers,
including all 15 handle cylinders the gripper has to grasp. On the 1.1.1 tree that material authors
`physics:staticFriction = physics:dynamicFriction = 0.8`. On og391 it authors **neither**, because the
commit that added them (`36f5028`) is not an ancestor of `port-to-og391`.

Read off the authored file, og391 therefore resolves both to **0.0** -- the UsdPhysics schema fallback
-- and with `physxMaterial:frictionCombineMode = min` a 0.0 on the drawer side makes the handle
frictionless against any gripper pad, however grippy the pad is.

That last step is exactly what this probe exists to check rather than assume, for two reasons:

  * Omniverse PhysX may substitute its OWN default (commonly 0.5) for an absent attribute instead of
    honouring the schema fallback of 0.0. Those two possibilities differ by the whole question.
  * the binding is the general-purpose `material:binding`, not `material:binding:physics`, so whether
    PhysX picks the material up for these meshes at all is a runtime question.

So: dump, on the live stage, the material prim's resolved friction, whether OmniGibson replaced or
re-bound it, and the friction of the gripper's own pad material for comparison. Reports what it finds;
asserts nothing.

    ./scripts/clara/interactive/rr python -u scripts/clara/interactive/t19_drawer_friction.py \
        --num_envs 1 --task_id 8
"""
import argparse

import omnigibson as og
import omnigibson.lazy as lazy

from realm.environments.env_vector import RealmVectorEnvironment
from realm.eval import SUPPORTED_PERTURBATIONS, SUPPORTED_TASKS
from realm.sim_config import set_sim_config


def material_row(prim, label):
    api = lazy.pxr.UsdPhysics.MaterialAPI(prim)
    if not prim.IsValid():
        print(f"    {label}: <invalid prim>")
        return
    has = prim.HasAPI(lazy.pxr.UsdPhysics.MaterialAPI)
    print(f"    {label}: {prim.GetPath()}")
    print(f"        PhysicsMaterialAPI={has} schemas={list(prim.GetAppliedSchemas())}")
    if not has:
        return
    for name, at in (("staticFriction", api.GetStaticFrictionAttr()),
                     ("dynamicFriction", api.GetDynamicFrictionAttr()),
                     ("restitution", api.GetRestitutionAttr())):
        print(f"        {name:<16} authored={str(at.HasAuthoredValue()):<6} resolved={at.Get()}")
    for extra in ("physxMaterial:frictionCombineMode", "physxMaterial:restitutionCombineMode"):
        a = prim.GetAttribute(extra)
        if a:
            print(f"        {extra} = {a.Get()}")


def main(num_envs, task_id, robot, perturbation):
    set_sim_config(robot=robot)
    vec_env = RealmVectorEnvironment(
        num_envs,
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml",
        perturbations=[perturbation],
        robot=robot,
    )
    stage = og.sim.stage
    print(f"\n########## t19 drawer friction: task {task_id} "
          f"({SUPPORTED_TASKS[task_id]}) ##########", flush=True)

    for i, env in enumerate(vec_env.envs):
        cabinet = env.main_objects[0]
        print(f"\n  ===== member {i}: {cabinet.prim_path} =====")

        # Every physics material anywhere under the cabinet, however it got there.
        print("  physics materials under the cabinet prim:")
        found = 0
        for prim in lazy.pxr.Usd.PrimRange(stage.GetPrimAtPath(cabinet.prim_path)):
            if prim.HasAPI(lazy.pxr.UsdPhysics.MaterialAPI):
                material_row(prim, f"material[{found}]")
                found += 1
        if not found:
            print("    none -- OmniGibson did not carry the asset's physics material onto the stage")

        # What the collision meshes are actually bound to, both binding purposes.
        print("\n  bindings on the drawer's collision meshes (first 4 of each purpose):")
        link_name = env.mo_joint.body1.split("/")[-1]
        link_prim = stage.GetPrimAtPath(cabinet.links[link_name].prim_path)
        seen = {"material:binding": [], "material:binding:physics": []}
        for prim in lazy.pxr.Usd.PrimRange(link_prim):
            if not prim.IsA(lazy.pxr.UsdGeom.Mesh):
                continue
            for rel_name in seen:
                r = prim.GetRelationship(rel_name)
                if r and r.GetTargets():
                    seen[rel_name].append((prim.GetName(), [str(t) for t in r.GetTargets()]))
        for rel_name, entries in seen.items():
            print(f"    {rel_name}: {len(entries)} mesh(es)")
            for nm, tgts in entries[:4]:
                print(f"        {nm} -> {tgts}")

        # The gripper pad, for the `min` combine comparison.
        print("\n  gripper-side physics materials (for the frictionCombineMode=min comparison):")
        robot_obj = env.omnigibson_env.robots[0]
        npad = 0
        for prim in lazy.pxr.Usd.PrimRange(stage.GetPrimAtPath(robot_obj.prim_path)):
            if prim.HasAPI(lazy.pxr.UsdPhysics.MaterialAPI):
                material_row(prim, f"robot material[{npad}]")
                npad += 1
                if npad >= 4:
                    print("        (stopping after 4)")
                    break
        if not npad:
            print("    none found under the robot prim")

    print("\n########## t19 done ##########", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--robot", type=str, default="DROID_robolab")
    p.add_argument("--perturbation", type=str, default=SUPPORTED_PERTURBATIONS[0])
    a = p.parse_args()
    main(a.num_envs, a.task_id, a.robot, a.perturbation)
