
import numpy as np

import omnigibson as og
from omnigibson.utils.usd_utils import RigidContactAPI

from realm.environments.contact_utils import get_impulse_contacts
from realm.environments.joint_reset import JointResetMixin
from realm.environments.task_progression import TaskProgressionMixin
# Avoid importing inference transport dependencies through the package.
from realm.inference.utils import arm_names, finger_proprio_indices, get_robot_obs_profile
from realm.robots.controller_registry import register_realm_controllers

# Re-exported for tests/test_joint_reset_batching.py.
from realm.environments.joint_reset import (  # noqa: F401
    JOINT_HOLD_STEPS,
    JOINT_SETTLE_STEPS,
    JointResetPlan,
    run_joint_resets,
)
from realm.environments.task_progression import TASK_PROGRESS_RUBRICS  # noqa: F401

register_realm_controllers()



def robot_finger_link_prims(robot, arm=None):
    """Finger link prims of `arm`, or of EVERY arm when None. Single-arm robots: the default arm's two
    fingers, exactly as before; the bimanual YAM: all four."""
    arms = list(robot.arm_names) if arm is None else [arm]
    return {robot._links[link] for a in arms for link in robot.finger_link_names[a]}


class RealmEnvironmentBase(JointResetMixin, TaskProgressionMixin):
    in_vec_env = False

    def __init__(
        self,
        main_objects,
        target_objects,
        task_type,
        robot,
        mo_cfgs
    ):
        self.main_objects = main_objects
        self.target_objects = target_objects

        # Reset refreshes pose references after perturbations replace the main object.
        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        # Bounding-box perturbations remain anchored to the authored size.
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.task_type = task_type
        self.robot = robot
        self.robot_finger_links = robot_finger_link_prims(self.robot)

        self._init_task_progression(task_type)
        self.reset_joints()

    def capture_mo_reference(self):

        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()

    def get_ee_pose(self):
        ee_link_name = self.robot.eef_link_names[self.robot.default_arm]
        ee_link = self.robot.links[ee_link_name]
        return ee_link.get_position_orientation()

    def _adjacent_link_pairs(self):

        if not hasattr(self, "_robot_adjacent_links"):
            self._robot_adjacent_links = set()
            if hasattr(self.robot, "joints"):
                for joint in self.robot.joints.values():
                    b0 = joint.body0
                    b1 = joint.body1
                    if b0 and b1:
                        self._robot_adjacent_links.add(frozenset((b0, b1)))
        return self._robot_adjacent_links

    def check_collisions(self):

        self_collision = False
        env_collision = False

        adjacent_link_pairs = self._adjacent_link_pairs()

        robot_links = list(self.robot.links.values())
        robot_link_paths = set(l.prim_path for l in robot_links)
        robot_prim_path = self.robot.prim_path

        # Prefixes, so links and geoms belonging to a manipulation target are all covered.
        ignore_obj_roots = [obj.prim_path for obj in self.main_objects + self.target_objects]

        queried_links = [link for link in robot_links if link.name != self.robot.root_link_name]
        contacts_by_link = get_impulse_contacts(self.robot.scene.idx, queried_links)

        for link in queried_links:
            for other_path in contacts_by_link.get(link.prim_path, ()):
                is_robot = other_path in robot_link_paths or other_path.startswith(robot_prim_path)

                if is_robot:
                    if other_path in robot_link_paths:
                        if frozenset((link.prim_path, other_path)) in adjacent_link_pairs:
                            continue
                    self_collision = True
                else:
                    is_ignored = any(other_path.startswith(root) for root in ignore_obj_roots)
                    if not is_ignored:
                        env_collision = True

            if self_collision and env_collision:
                break

        return self_collision, env_collision

    def _finger_closure_threshold(self):

        profile = get_robot_obs_profile(self.robot.name)
        open_q, closed_q = profile["gripper_open_qpos"], profile["gripper_closed_qpos"]
        return open_q + 9.0 * (closed_q - open_q)

    def _is_either_finger_closing(self, finger_joints):
        """The closure test, written for grippers whose closed position is ABOVE the open one (DROID's
        Robotiq 0 -> 0.785, YAMLab's fingers -0.0475 -> 0): a finger counts as closing while it is below
        the threshold. The expression is kept verbatim for those robots. A gripper that closes TOWARD the
        lower end (ABC's crank fingers, open at +0.0475, closed at 0) gets the mirror image; without it the
        threshold sits far below the joint range and no grasp could ever be detected."""
        profile = get_robot_obs_profile(self.robot.name)
        thresh = self._finger_closure_threshold()
        if profile["gripper_closed_qpos"] > profile["gripper_open_qpos"]:
            return thresh - finger_joints[0] > 1e-3 or thresh - finger_joints[1] > 1e-3
        return finger_joints[0] - thresh > 1e-3 or finger_joints[1] - thresh > 1e-3

    def is_grasping(self, obs, candidate_obj):
        """True when SOME arm holds `candidate_obj`: both of that arm's fingers touch it, the robot's
        Touching state agrees, and at least one finger is past the closure threshold. Single-arm robots
        evaluate exactly the one arm they have."""
        is_robot_touching_obj = self.is_touching(obs, candidate_obj)
        for arm in arm_names(self.robot.name):
            # The two finger DOFs follow the arm joints in the proprio vector (7 for DROID, 6 for YAM);
            # multi-arm profiles look them up by name.
            idx = finger_proprio_indices(self.robot.name, arm)
            finger_joints = obs[self.robot.name]['proprio'][idx].cpu().numpy()
            is_either_finger_closing = self._is_either_finger_closing(finger_joints)
            finger_links = (self.robot_finger_links if arm is None
                            else robot_finger_link_prims(self.robot, arm))
            contact_pairs = RigidContactAPI.get_contact_pairs(
                scene_idx=candidate_obj.scene.idx,
                query_set={candidate_obj},
                with_set=finger_links,
                current_only=True,
            )
            is_both_fingers_touching_obj = len({finger_path for _, finger_path in contact_pairs}) == 2

            if is_both_fingers_touching_obj and is_robot_touching_obj and is_either_finger_closing:
                return True
        return False

    def is_touching(self, obs, candidate_obj):
        is_robot_touching_obj = self.robot.states[og.object_states.Touching].get_value(candidate_obj)
        return is_robot_touching_obj
