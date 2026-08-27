
import copy

import numpy as np

import omnigibson as og

from realm.environments.utils import load_task_progressions
from realm.geometry import compute_rot_diff_magnitude

TASK_PROGRESS_RUBRICS = load_task_progressions()

DRAWER_TASK_TYPES = ("open_drawer", "close_drawer")


class TaskProgressionMixin:


    def _init_task_progression(self, task_type):

        self.was_lifted = False
        self.task_progression = (
            copy.deepcopy(TASK_PROGRESS_RUBRICS[task_type])
            if task_type in TASK_PROGRESS_RUBRICS
            else None
        )
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
            "MOVE_JOINT_SMALL": self.check_moved_mo_joint_small,  # TODO: turn faucet
            "MOVE_JOINT_LARGE": self.check_moved_mo_joint_large,
            "MOVE_JOINT_FULL": self.check_moved_mo_joint_full,
            "TOGGLED_ON": self.check_toggled_on_condition,
            "POUR": self.check_pour,  # TODO: pouring
        }

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

        if self.task_type in DRAWER_TASK_TYPES:
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

    def check_grasp_condition(self, obs):
        return self.is_grasping(obs, self.main_objects[0])

    def check_touch_condition(self, obs):
        return self.is_touching(obs, self.main_objects[0])

    def check_rotated(self, obs, rot_threshold=1.1):
        mo = self.main_objects[0]
        mo_rot_curr = mo.get_position_orientation()[1]

        rot_diff = compute_rot_diff_magnitude(self.mo_rot_orig, mo_rot_curr)

        return abs(rot_diff) > rot_threshold

    def check_lift_and_distance_condition(self, distance_threshold=0.05, lift_threshold=0.01):

        mo = self.main_objects[0]
        mo_pos_curr = mo.get_position_orientation()[0]

        distance = np.linalg.norm(mo_pos_curr - self.mo_pos_orig)

        return mo_pos_curr[2] - self.mo_pos_orig[2] > lift_threshold and distance > distance_threshold

    def check_lift_slight_condition(self, obs):
        return self.check_lift_and_distance_condition()

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
        return distance < 0.125

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

    def check_pour(self, obs):

        return False

    def get_mo_joint_openness_fraction(self):
        assert self.mo_joint is not None
        return (self.mo_joint.get_state()[0][0] - self.mo_joint.lower_limit) / self.joint_range

    def get_mo_joint_delta(self):

        openness_fraction = self.get_mo_joint_openness_fraction()
        delta_openness_fraction = self.init_openness_fraction - openness_fraction
        return delta_openness_fraction

    def check_touching_and_moved_mo_joint(self, obs, threshold=0.025):
        delta_openness_fraction = self.get_mo_joint_delta()
        if self.task_type == "open_drawer":
            return self.check_touch_condition(obs) and delta_openness_fraction < threshold
        elif self.task_type == "close_drawer":
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
