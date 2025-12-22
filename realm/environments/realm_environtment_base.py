import numpy as np
import torch

from realm.tasks.task_progressions import TASK_PROGRESSIONS
from realm.helpers import compute_rot_diff_magnitude
from realm.robots.droid_joint_controller import IndividualJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController
import omnigibson as og
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.object_states.open_state import _get_relevant_joints
from omnigibson.utils.object_utils import compute_base_aligned_bboxes, compute_bbox_offset
from omnigibson.prims.joint_prim import JointPrim, JointType
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.objects.dataset_object import DatasetObject

REGISTERED_CONTROLLERS["CustomJointController"] = IndividualJointPDController
REGISTERED_CONTROLLERS["CustomGripperController"] = MultiFingerGripperController
INIT_OPENNESS_FRACTION = 1.0 #0.5


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
            #print(f"MYDEBUG: {__name__}: {step=}, {j.name=}, {j.get_state(normalized=True)=}, {target_state=}")
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
    drawer_heights = drawer_heights[-3:] # take max top 3 drawers, on the default nddvba this drops the hidden bottom one
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


class RealmEnvironmentBase:
    def __init__(
        self,
        main_objects,
        target_objects,
        task,
        task_type,
        robot,
        use_droid_with_base,
        mo_cfgs
    ):
        self.use_droid_with_base = use_droid_with_base

        self.main_objects = main_objects
        self.target_objects = target_objects

        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.task = task
        self.task_type = task_type
        self.robot = robot
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

        self.was_lifted = False
        # assert not (task_type in TASK_PROGRESSIONS and task in TASK_PROGRESSIONS)
        if task_type in TASK_PROGRESSIONS:
            self.task_progression = TASK_PROGRESSIONS[task_type]
        elif task in TASK_PROGRESSIONS:
            self.task_progression = TASK_PROGRESSIONS[task]
        else:
            self.task_progression = None

        # if task_type in ["open_close_drawer"]:
        #     #cabinet = self.omnigibson_env.scene.object_registry("name", "cabinet")
        #     cabinet = main_objects[0]
        #     assert cabinet.name == "cabinet" # TODO: get object name dynamically
        #     relevant_joints = _get_relevant_joints(cabinet)[1]
        #     self.mo_joint = relevant_joints[0] # TODO: get joint index dynamically
        #     self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
        #     self.init_openness_fraction = (self.mo_joint.get_state()[0][0] - self.mo_joint.lower_limit) / self.joint_range
        # else:
        #     self.mo_joint = None

        self.reset_joints()

        self.success_conditions = {
            "REACH": self.check_reach_condition,
            "GRASP": self.check_grasp_condition,
            "TOUCH": self.check_touch_condition,
            "LIFT_SLIGHT": self.check_lift_slight_condition,
            "LIFT_LARGE": self.check_lift_large_condition,
            "ROTATED": self.check_rotated,
            "PUSH": self.check_push,
            "MOVE_CLOSE": self.check_move_close_condition,
            "PLACE_INTO": self.check_place_condition,
            "PLACE_ONTO": self.check_place_onto_condition,
            "TOUCH_AND_MOVE_JOINT": self.check_touching_and_moved_mo_joint,
            "OPEN_JOINT_SMALL": self.check_opened_mo_joint_small,
            "OPEN_JOINT_LARGE": self.check_opened_mo_joint_large,
            "OPEN_JOINT_FULL": self.check_opened_mo_joint_full,
            "CLOSE_JOINT_SMALL": self.check_closed_mo_joint_small,
            "CLOSE_JOINT_LARGE": self.check_closed_mo_joint_large,
            "CLOSE_JOINT_FULL": self.check_closed_mo_joint_full,
            "MOVE_JOINT_SMALL": self.check_moved_mo_joint_small, # TODO: turn faucet
            "MOVE_JOINT_LARGE": self.check_moved_mo_joint_large,
            "MOVE_JOINT_FULL": self.check_moved_mo_joint_full,
            "TOGGLED_ON": self.check_toggled_on_condition,
            "POURED": self.check_pour # TODO: pouring
        }

    def  reset_joints(self, target_drawer_loc: str = "top"):
        if self.task_type in ["open_drawer", "close_drawer"]:
            cabinet = self.main_objects[0]
            #assert cabinet.category in ("bottom_cabinet", "bottom_cabinet_no_top")
            # relevant_joints = _get_relevant_joints(cabinet)[1]
            init_state_open = self.task_type == "close_drawer"
            self.mo_joint = get_target_drawer_joint(cabinet, target_drawer_loc=target_drawer_loc)

            #self.mo_joint._articulation_view.set_max_velocities(torch.tensor([[0.2]]), joint_indices=self.mo_joint.dof_indices)
            self.mo_joint._articulation_view.set_max_efforts(torch.tensor([[1.0e8]], dtype=torch.float32), joint_indices=self.mo_joint.dof_indices)
            self.mo_joint._articulation_view.set_gains(kps=torch.tensor([[0.0]]), joint_indices=self.mo_joint.dof_indices)
            self.mo_joint._articulation_view.set_gains(kds=torch.tensor([[1000.0]]), joint_indices=self.mo_joint.dof_indices)
            # self.mo_joint.set_attribute("physxJoint:jointFriction", 10.0)
            # if og.sim.is_playing():
            #     self.mo_joint._articulation_view.set_friction_coefficients(torch.tensor([[10.0]]), joint_indices=self.mo_joint.dof_indices)

            openable_joints = get_openable_joints(cabinet)
            reset_states = [-1 for _ in openable_joints]
            target_joint_ind = openable_joints.index(self.mo_joint)
            reset_states[target_joint_ind] = INIT_OPENNESS_FRACTION if init_state_open else -1
            reset_joints(openable_joints, reset_states=reset_states)
            self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
            self.init_openness_fraction = (self.mo_joint.get_state()[0][
                                               0] - self.mo_joint.lower_limit) / self.joint_range
            for _ in range(30):
                og.sim.step()
            for j in cabinet.joints.values():
                j: JointPrim
                j.keep_still()
            for _ in range(10):
                og.sim.step()

        else:
            self.mo_joint = None

    # ============================== [SUCCESS METRICS] ==============================
    def is_grasping(self, obs, candidate_obj):
        finger_joints = obs['franka']['proprio'][7:9].cpu().numpy()
        is_either_finger_closing = (0.45 - finger_joints[0] > 1e-3 or 0.45 - finger_joints[1] > 1e-3)
        is_both_fingers_touching_obj = len(
            candidate_obj.states[ContactBodies].get_value().intersection(self.robot_finger_links)) == 2
        is_robot_touching_obj = self.is_touching(obs, candidate_obj)

        if is_both_fingers_touching_obj and is_robot_touching_obj and is_either_finger_closing:
            return True
        return False

    def is_touching(self, obs, candidate_obj):
        is_robot_touching_obj = self.robot.states[og.object_states.Touching].get_value(candidate_obj)
        return is_robot_touching_obj

    def recompute_task_progression(self, obs):
        reward = 0.0

        if self.task_progression is not None:
            for stage, is_completed_flag in self.task_progression.items():
                checker_function = self.success_conditions.get(stage)
                if is_completed_flag or checker_function(obs):
                    if not is_completed_flag:
                        self.task_progression[stage] = True
                    reward += 1 / len(self.task_progression.keys())
                else:
                    break
            assert 0.0 <= reward <= 1.0
        return reward

    def check_reach_condition(self, obs):
        mo = self.main_objects[0]

        if self.task_progression in ["open_close_drawer"]:
            return self.is_touching(obs, mo)

        pos1 = mo.get_position_orientation()[0]
        finger1 = list(self.robot_finger_links)[0]
        pos_finger1 = finger1.get_position_orientation()[0]
        finger2 = list(self.robot_finger_links)[1]
        pos_finger2 = finger2.get_position_orientation()[0]

        distance_1 = np.linalg.norm(pos1 - pos_finger1)
        distance_2 = np.linalg.norm(pos1 - pos_finger2)

        dist = 0.1
        return distance_1 < dist or distance_2 < dist or self.check_touch_condition(obs)

        # TODO: make the distance computation bbox dependent
        # obj_pos = mo.get_position_orientation()[0]
        # xmin, ymin, zmin = (obj_pos - self.mo_bbox_orig / 2).tolist()
        # xmax, ymax, zmax = (obj_pos + self.mo_bbox_orig / 2).tolist()
        #
        # finger_distances = []
        # for finger in list(self.robot_finger_links):
        #     finger_pos = finger.get_position_orientation()[0]
        #     finger_x, finger_y, finger_z = finger_pos.tolist()
        #
        #     closest_x = max(xmin, min(finger_x, xmax))
        #     closest_y = max(ymin, min(finger_y, ymax))
        #     closest_z = max(zmin, min(finger_z, zmax))
        #
        #     dx = finger_x - closest_x
        #     dy = finger_y - closest_y
        #     dz = finger_z - closest_z
        #     dist = np.sum(np.abs([dx, dy, dz]))
        #     finger_distances.append(dist)
        # print(finger_distances)

    def check_grasp_condition(self, obs):
        return self.is_grasping(obs, self.main_objects[0])

    def check_touch_condition(self, obs):
        return self.is_touching(obs, self.main_objects[0])

    def get_mo_joint_openness_fraction(self):
        assert self.mo_joint is not None
        return (self.mo_joint.get_state()[0][0] - self.mo_joint.lower_limit) / self.joint_range

    def get_mo_joint_delta(self):
        openness_fraction = self.get_mo_joint_openness_fraction()
        delta_openness_fraction = self.init_openness_fraction - openness_fraction
        return delta_openness_fraction

    def check_touching_and_moved_mo_joint(self, obs, threshold=0.025):
        delta_openness_fraction = self.get_mo_joint_delta()
        if self.task == "open_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction < threshold
        elif self.task == "close_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction > threshold
        else:
            raise NotImplementedError()

    def check_opened_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.125

    def check_opened_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.65

    def check_opened_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() > 0.95

    def check_closed_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.875

    def check_closed_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.35

    def check_closed_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction() < 0.05

    def check_moved_mo_joint_small(self, obs):
        return self.check_closed_mo_joint_small(obs) or self.check_opened_mo_joint_small(obs)

    def check_moved_mo_joint_large(self, obs):
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)

    def check_moved_mo_joint_full(self, obs):
        return self.check_closed_mo_joint_large(obs) or self.check_opened_mo_joint_large(obs)

    # NOTE: switched to checking Z axis rotation only, possible it is still bad but seems to be working well now
    def check_rotated(self, obs, rot_threshold=1.1):
        mo = self.main_objects[0]
        mo_rot_curr = mo.get_position_orientation()[1]

        rot_diff = compute_rot_diff_magnitude(self.mo_rot_orig, mo_rot_curr)
        print(mo_rot_curr, self.mo_rot_orig, rot_diff)

        return abs(rot_diff) > rot_threshold

    def check_lift_and_distance_condition(self, distance_threshold=0.05, lift_threshold=0.01):
        mo = self.main_objects[0]
        mo_pos_curr = mo.get_position_orientation()[0]

        distance = np.linalg.norm(mo_pos_curr - self.mo_pos_orig)

        return mo_pos_curr[2] - self.mo_pos_orig[2] > lift_threshold and distance > distance_threshold

    def check_lift_slight_condition(self, obs):
        return self.check_lift_and_distance_condition()  # lifted at least 1cm and traveled at least 5cm

    def check_lift_large_condition(self, obs):
        return self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=0.075)

    def check_push(self, obs):
        mo = self.main_objects[0]
        push_cond = self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=-0.05)
        is_lifted = self.check_lift_and_distance_condition(distance_threshold=-0.05, lift_threshold=0.05)
        self.was_lifted = is_lifted or self.was_lifted
        is_robot_touching_obj = self.robot.states[og.object_states.Touching].get_value(mo)
        return push_cond and is_robot_touching_obj and not self.was_lifted

    def check_move_close_condition(self, obs):
        assert len(self.main_objects) == 1
        assert len(self.target_objects) == 1

        mo = self.main_objects[0]
        pos1 = mo.get_position_orientation()[0]

        target = self.target_objects[0]
        pos2 = target.get_position_orientation()[0]

        distance = np.linalg.norm(pos1 - pos2)
        return distance < 0.125 #0.075 #TODO: adjust for size of receiver, this might not always be enough it seems

    def check_place_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        inside_or_on_top = mo.states[og.object_states.OnTop].get_value(target) or mo.states[og.object_states.Inside].get_value(target)
        return inside_or_on_top and not self.is_grasping(obs, mo)

    def check_place_onto_condition(self, obs):
        mo = self.main_objects[0]
        target = self.target_objects[0]
        return mo.states[og.object_states.OnTop].get_value(target) and not self.is_grasping(obs, mo)

    def check_toggled_on_condition(self, obs):
        mo = self.main_objects[0]
        return mo.states[og.object_states.ToggledOn].get_value()

    def check_pour(self):
        return False

