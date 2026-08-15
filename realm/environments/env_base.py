"""The task-agnostic half of a REALM environment.

``RealmEnvironmentBase`` owns what a rollout is SCORED against: the start-of-rollout reference pose
of the main object, and the robot/contact predicates the stage checkers call. The stages themselves
live in ``task_progression.py``; the drawer reset in ``joint_reset.py``; building the scene all of
this reads is ``env_dynamic.py``'s job.
"""
import numpy as np

import omnigibson as og
from omnigibson.utils.usd_utils import RigidContactAPI  # replaces ContactBodies, removed in OG 3.9.1

from realm.environments.contact_utils import get_impulse_contacts
from realm.environments.joint_reset import JointResetMixin
from realm.environments.task_progression import TaskProgressionMixin
# Imported from the module rather than the package so that pulling in a REALM environment does not
# also pull in the inference client and its transport deps.
from realm.inference.utils import get_robot_obs_profile
from realm.robots.controller_registry import register_realm_controllers

# Re-exported: these five are imported from here by tests/test_joint_reset_batching.py and by
# scripts/clara/interactive/{t9,t13}*.py, which predate the split.
from realm.environments.joint_reset import (  # noqa: F401
    JOINT_HOLD_STEPS,
    JOINT_SETTLE_STEPS,
    JointResetPlan,
    run_joint_resets,
)
from realm.environments.task_progression import TASK_PROGRESS_RUBRICS  # noqa: F401

# OG 3.9.1 requires a default controller config entry per custom controller, not just a
# REGISTERED_CONTROLLERS entry -- see realm/robots/controller_registry.py.
register_realm_controllers()


class RealmEnvironmentBase(JointResetMixin, TaskProgressionMixin):
    # Per-instance in practice -- RealmEnvironmentDynamic.__init__ sets it before og.Environment is
    # built -- but declared here because RealmEnvironmentBase.__init__ itself calls reset_joints(),
    # which has to read it.
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

        # Build-time seed only. Correct while main_objects[0] is still the object the task config
        # declared, which stops being true the moment a perturbation swaps it -- every reset
        # re-takes these from the live object via capture_mo_reference() below.
        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        # Build-time and STAYS build-time, unlike the two above. See capture_mo_reference().
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.task_type = task_type
        self.robot = robot
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

        self._init_task_progression(task_type)
        self.reset_joints()

    def capture_mo_reference(self):
        """Re-take mo_pos_orig / mo_rot_orig from whatever main_objects[0] points at RIGHT NOW.

        These two are the START-OF-ROLLOUT reference the progression stages are judged against:
        check_lift_and_distance_condition() (LIFT_SLIGHT, LIFT_LARGE, PUSH) measures both the lift
        and the travel from mo_pos_orig, and check_rotated() (ROTATED) measures against mo_rot_orig.
        __init__ seeds them from the task config, which is only right while main_objects[0] is still
        the object the config declared -- and SB-NOUN, VSB-NOBJ and VB-MOBJ all re-point it DURING
        reset(), after the seed. Without this the reference described one object while the checks
        read another: measured 2026-08-13 at 0.111-0.465 m of separation, with LIFT_SLIGHT answering
        True AT REST on 3 of 6 resets.

        Call this ONLY at the end of a reset, never while stepping. It records where the object
        STARTED; a reference that followed the object would drive both terms to zero and make every
        lift/distance check permanently False, silently deleting the stage instead of fixing it.

        Kept as one method rather than a line in each perturbation so a future perturbation that
        swaps the object cannot forget it: every reset path ends here. A vector env needs its own
        call in RealmVectorEnvironment.reset(), because apply_perturbations() runs there before the
        shared play -- exactly as it already needs its own settle and its own deferred post-play
        drain.

        mo_bbox_orig is DELIBERATELY not re-taken, though it is seeded on the line right after these
        two and looks like it has the same staleness shape. It is an ANCHOR on the task config, not
        a description of the current object, and re-taking it would turn VB-MOBJ's per-reset draw
        into a multiplicative random walk. The full argument, the measurement above, and what would
        have to change if perturbations were ever composed are in docs/vector_env/PERTURBATIONS.md
        under "The scoring reference".
        """
        # Stored as OmniGibson hands them back (torch, cloned -- RigidDynamicPrim.get_position_
        # orientation defaults to clone=True, so this is a snapshot and not a view onto the physics
        # buffer). Deliberately NOT converted to numpy: this is byte-for-byte what warmup() has
        # always stored, so no historical number moves.
        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()

    # ============================== [STATUS] ==============================
    def get_ee_pose(self):
        ee_link_name = self.robot.eef_link_names[self.robot.default_arm]
        ee_link = self.robot.links[ee_link_name]
        return ee_link.get_position_orientation()

    def _adjacent_link_pairs(self):
        """Robot link pairs joined by a joint, whose mutual contact is not a self-collision."""
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
        """(self_collision, env_collision) for the robot this step.

        Contact with a manipulation target does not count as an environment collision, and contact
        between two links the same joint connects does not count as a self-collision. The root link
        is skipped outright -- it is usually touching the mount or the floor.

        OG 3.9.1 removed RigidPrim.contact_list(), so contacts and their impulses come from the
        contact matrix instead: one batched query for all links rather than a per-link call. See
        realm/environments/contact_utils.py.
        """
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
                    # Only an exact link path can be tested for adjacency; anything else that
                    # merely starts with the robot's prim path is assumed to be a real
                    # self-collision.
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

    # ============================== [SUCCESS METRICS] ==============================
    def _finger_closure_threshold(self):
        """Finger joint value below which the gripper counts as "closing" on something.

        The original test compared a bare literal 0.45 against the raw finger joint value, with no
        units and no normalisation by joint range, so what it meant depended entirely on the asset.
        On droid.usd the finger joints are PRISMATIC in metres over [0, 0.05], where 0.45 is 9x the
        entire travel and the test is VACUOUSLY TRUE. On the robolab 2F-85 the same proprio indices
        are REVOLUTE in radians over [0, 0.7854], where 0.45 lands mid-range and the test becomes
        "less than ~57% closed" -- which a real grasp violates. Measured 2026-08-11 (job 189066):
        with both pads on the block and the block lifted, finger_joint sits at 0.507-0.528, so this
        rejected 78/78 genuine grasp steps and the asset could never score a GRASP. Because
        recompute_task_progression breaks at the first unmet stage, that also froze LIFT/MOVE/PLACE
        on rollouts that visibly completed the task.

        Scaling by the robot's own open->closed range instead reproduces 0.45 EXACTLY for droid.usd
        (9 * 0.05), so every historical result is bit-identical; for robolab it becomes 7.07 rad and
        the test is vacuous there too, matching the behaviour the stock asset has always had.

        NOTE: 0.45 is very likely a typo for 0.045, i.e. "the fingers stopped short of full closure,
        so an object is between them" (90% of droid.usd's travel), which is a meaningful test rather
        than a no-op. Deliberately NOT adopted here: it would make the guard bite on droid.usd for
        the first time and could move every historical REALM number. Decide that separately, with a
        measurement.
        """
        profile = get_robot_obs_profile(self.robot.name)
        open_q, closed_q = profile["gripper_open_qpos"], profile["gripper_closed_qpos"]
        return open_q + 9.0 * (closed_q - open_q)

    def is_grasping(self, obs, candidate_obj):
        """Both fingers on @candidate_obj, the robot touching it, and the gripper still closing."""
        finger_joints = obs[self.robot.name]['proprio'][7:9].cpu().numpy()
        thresh = self._finger_closure_threshold()
        is_either_finger_closing = (thresh - finger_joints[0] > 1e-3 or thresh - finger_joints[1] > 1e-3)
        # OG 3.9.1 removed the ContactBodies object state; query the contact matrix directly instead.
        # get_contact_pairs returns (query_prim_path, with_prim_path) tuples, so the second element of
        # each pair is the finger link that candidate_obj is touching.
        contact_pairs = RigidContactAPI.get_contact_pairs(
            scene_idx=candidate_obj.scene.idx,
            query_set={candidate_obj},
            with_set=self.robot_finger_links,
            current_only=True,
        )
        is_both_fingers_touching_obj = len({finger_path for _, finger_path in contact_pairs}) == 2
        is_robot_touching_obj = self.is_touching(obs, candidate_obj)

        if is_both_fingers_touching_obj and is_robot_touching_obj and is_either_finger_closing:
            return True
        return False

    def is_touching(self, obs, candidate_obj):
        is_robot_touching_obj = self.robot.states[og.object_states.Touching].get_value(candidate_obj)
        return is_robot_touching_obj
