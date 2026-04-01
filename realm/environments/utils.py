import yaml
import os
from collections import OrderedDict

import omnigibson as og
from omnigibson.object_states.open_state import _get_relevant_joints
from omnigibson.prims.joint_prim import JointPrim, JointType
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.objects.dataset_object import DatasetObject


_current_dir = os.path.dirname(os.path.abspath(__file__))
_yaml_path = os.path.join(_current_dir, "../config/tasks/task_progressions.yaml")

def load_task_progressions():
    with open(_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    task_progressions = {}
    for task, stages in data.items():
        task_progressions[task] = OrderedDict((stage, False) for stage in stages)

    return task_progressions


def reset_joints(
        joints: list[JointPrim],
        reset_states: list[float] = None,
        closing_steps: int = 10,
        still_steps: int = 5
):
    if reset_states is None:
        reset_states = [-1.0 for _ in joints]
    assert len(joints) == len(reset_states), f"{len(joints)=}, {len(reset_states)=}"
    for step in range(closing_steps):
        for j, target_state in zip(joints, reset_states):
            j.set_pos(target_state, normalized=True)
            j.set_vel(0)
            j.set_effort(0)
        og.sim.step()
    for step in range(still_steps):
        for j in joints:
            j.keep_still()
        og.sim.step()


def get_openable_joints(cabinet: DatasetObject) -> list[JointPrim]:
    relevant_joints = _get_relevant_joints(cabinet)[1]
    openable_joints = []
    for j in relevant_joints:
        if j.joint_type in (JointType.JOINT_PRISMATIC, JointType.JOINT_REVOLUTE):
            openable_joints.append(j)
    return openable_joints


def get_target_drawer_joint(cabinet: DatasetObject, target_drawer_loc: str) -> JointPrim:
    assert target_drawer_loc in ("top", "middle", "bottom"), f"{target_drawer_loc=}"

    links: list[RigidPrim] = list(cabinet.links.values())
    joints: list[JointPrim] = _get_relevant_joints(cabinet)[1]
    path2link = {l.prim_path: l for l in links}
    drawer_heights = []
    for j in joints:
        if j.joint_type != JointType.JOINT_PRISMATIC:
            continue
        drawer_link_path = j.body1
        link = path2link[drawer_link_path]
        z = link.aabb_center[-1].item()
        drawer_heights.append((j, z))
    drawer_heights = sorted(drawer_heights, key=lambda x: x[1], reverse=True)[:3]  # take top 3 drawers by height
    if len(drawer_heights) == 0:
        all_joint_types = [(j.joint_name, j.joint_type) for j in joints]
        raise ValueError(
            f"No prismatic (drawer) joints found in cabinet '{cabinet.name}'. "
            f"Available joints: {all_joint_types}. "
            f"Check that the asset has drawer joints (not just revolute/door joints)."
        )
    if  len(drawer_heights) == 2:
        if target_drawer_loc == "top":
            target_joint = max(drawer_heights, key=lambda x: x[1])[0]
        elif target_drawer_loc == "middle":
            sorted_drawers = sorted(drawer_heights, key=lambda x: x[1])
            target_joint = sorted_drawers[0][0]
    else:
        if target_drawer_loc == "top":
            target_joint = max(drawer_heights, key=lambda x: x[1])[0]
        elif target_drawer_loc == "bottom":
            target_joint = min(drawer_heights, key=lambda x: x[1])[0]
        elif target_drawer_loc == "middle":
            assert len(drawer_heights) == 3, f"{len(drawer_heights)=}"
            sorted_drawers = sorted(drawer_heights, key=lambda x: x[1])
            target_joint = sorted_drawers[1][0]

    return target_joint
