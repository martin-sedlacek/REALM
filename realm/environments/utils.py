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

    reset_joints_batched([(joints, reset_states)], closing_steps=closing_steps, still_steps=still_steps)


def reset_joints_batched(
        programs: list[tuple[list[JointPrim], list[float] | None]],
        closing_steps: int = 10,
        still_steps: int = 5
):
    """Drive SEVERAL members' joint sets to their targets off ONE shared step loop.

    og.sim.step() advances every scene in the simulator, not the caller's. Running the single-member
    loop once per member therefore costs (closing_steps + still_steps) * N global steps and, worse,
    steps each member's scene N times per reset while its own joints are being driven for only one
    of those passes. Interleaving instead -- write every member's targets, then step once -- gives
    each member exactly the sequence of writes and steps it sees single-env, for a total of
    closing_steps + still_steps steps regardless of N.

    With a single program the emitted call sequence is identical to the pre-batching loop, so
    single-env behaviour is unchanged.
    """
    normalized = []
    for joints, reset_states in programs:
        if reset_states is None:
            reset_states = [-1.0 for _ in joints]
        assert len(joints) == len(reset_states), f"{len(joints)=}, {len(reset_states)=}"
        normalized.append((joints, reset_states))

    # Pure settle -- no camera is read, so skip the render pass on every step.
    with og.sim.render_on_step(False):
        for _ in range(closing_steps):
            for joints, reset_states in normalized:
                for j, target_state in zip(joints, reset_states):
                    j.set_pos(target_state, normalized=True)
                    j.set_vel(0)
                    j.set_effort(0)
            og.sim.step()
        for _ in range(still_steps):
            for joints, _ in normalized:
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
    """The prismatic joint of @cabinet's top / middle / bottom drawer, by drawer height.

    Only the top three drawers (by height) are considered. With exactly two drawers, "middle"
    historically meant the LOWER of the two (kept as-is), and "bottom" is refused as ambiguous --
    it used to crash with UnboundLocalError. With one drawer, "top" and "bottom" both name it and
    "middle" fails.
    """
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

    if len(drawer_heights) == 0:
        all_joint_types = [(j.joint_name, j.joint_type) for j in joints]
        raise ValueError(
            f"No prismatic (drawer) joints found in cabinet '{cabinet.name}'. "
            f"Available joints: {all_joint_types}. "
            f"Check that the asset has drawer joints (not just revolute/door joints)."
        )

    by_height = sorted(drawer_heights, key=lambda x: x[1], reverse=True)[:3]  # highest first
    n = len(by_height)

    if target_drawer_loc == "top":
        return by_height[0][0]
    if target_drawer_loc == "bottom":
        if n == 2:
            raise ValueError(
                f"cabinet '{cabinet.name}' has exactly 2 drawer joints, so 'bottom' is ambiguous "
                f"with 'middle' -- use 'top' or 'middle' for this asset")
        return by_height[-1][0]
    # middle
    if n == 2:
        return by_height[-1][0]  # the lower of the two, matching historical behaviour
    assert n == 3, f"{n=}"
    return by_height[1][0]
