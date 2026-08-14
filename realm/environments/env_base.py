import copy
from collections import namedtuple

import numpy as np
import torch

from realm.environments.utils import *
from realm.geometry import compute_rot_diff_magnitude
from realm.environments.contact_utils import get_impulse_contacts
from realm.robots.controller_registry import register_realm_controllers
# Per-robot gripper conventions, so is_grasping's finger-closure test is not hardcoded to
# droid.usd's units. Imported from the module rather than the package to avoid pulling in the
# inference client (and its transport deps) on the environment side.
from realm.inference.utils import get_robot_obs_profile
import omnigibson as og
from omnigibson.utils.usd_utils import RigidContactAPI  # replaces the ContactBodies object state, removed in OG 3.9.1
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.prims.joint_prim import JointPrim
from omnigibson.prims.rigid_prim import RigidPrim


# OG 3.9.1 also requires a default controller config entry per custom controller, not just a
# REGISTERED_CONTROLLERS entry -- see realm/robots/controller_registry.py.
register_realm_controllers()
INIT_OPENNESS_FRACTION = 1.0 #0.5
TASK_PROGRESS_RUBRICS = load_task_progressions()

# The free-run half of a drawer reset: let the cabinet come to rest, tell every one of its joints to
# hold, then let that take effect. Named rather than inline so the cost of a drawer reset is legible
# at a glance -- these two plus utils.reset_joints' 10 + 5 are the ~55 global steps per reset that
# run_joint_resets exists to stop multiplying by the member count.
JOINT_SETTLE_STEPS = 30
JOINT_HOLD_STEPS = 10

# One member's pending drawer reset: what reset_joints() worked out per member, minus every
# og.sim.step() -- those are global, so run_joint_resets() issues them once for all members.
JointResetPlan = namedtuple("JointResetPlan", ["cabinet", "joints", "reset_states"])


def run_joint_resets(envs):
    """Work every pending joint reset in @envs off ONE shared set of og.sim.step() calls.

    RealmEnvironmentBase.reset_joints() issues ~55 og.sim.step()s on a drawer task (10 + 5 driving
    the joints home in utils.reset_joints, then 30 + 10 free-running here). og.sim.step() is GLOBAL,
    so a vector env running that per member costs 55*N global steps per reset and advances every
    member's scene N times over while only one member's joints are being driven -- the same defect
    that the per-member settle loops and the per-member og.sim.stop()/play() cycles were hoisted out
    of RealmVectorEnvironment.reset() for. This is that hoist for the joint reset.

    Each member still experiences exactly the sequence a single env gives it: its own writes, then a
    step, N members' writes at a time. With one member the emitted calls are identical to the
    pre-batching straight-line version, so single-env behaviour is unchanged.

    VERIFIED END TO END 2026-08-14, once open_drawer/close_drawer started loading (they are the only
    task types that reach a non-empty @pending; everything else takes the early return above). This
    docstring used to say UNVERIFIED, on the grounds that neither task could build. Measured on
    task 8, Default, with the openness of every drawer read back after each reset:

        num_envs=2  reset issues 57 og.sim.step() calls  (2 per-member reset obs + 55 shared)
        num_envs=1  reset issues 56                      (1 + 55)

    57 rather than 110 is the batching. And the outcome is right, not just the count: member 1
    lands every one of its five drawer joints on the commanded normalized -1.0000, and member 0
    lands on exactly the state a num_envs=1 run of the same task produces, joint for joint. Both
    halves matter -- the count alone would also be satisfied by a loop that stepped once and wrote
    nothing.

    SEPARATE, PRE-EXISTING, NOT THIS FUNCTION'S DOING: in SCENE 0 the drawers do not reach the
    commanded position at all (target joint settles ~0.17-0.19 m of a 0.30 m range, and joints 02/03
    stop at 0.2289/0.2288 m in every run, which is a geometric stop rather than a control failure).
    That reproduces identically at num_envs=1 with this function bypassed, so it is a property of
    the task in scene 0 and not of the batching. Scene 0's cabinet also sits 44 mm higher than
    scene 1's (root-link z 0.544 vs 0.500) and scene 1's drawers close perfectly, which is where to
    start looking.
    """
    pending = [env for env in envs if env.pending_joint_reset is not None]
    if not pending:
        return

    reset_joints_batched(
        [(env.pending_joint_reset.joints, env.pending_joint_reset.reset_states) for env in pending]
    )
    # Between the two loops, exactly where the straight-line version read it: the openness the
    # drawer actually settled into once driven home, which is what the joint progression stages are
    # scored against -- not the openness it was commanded to.
    for env in pending:
        env._record_joint_openness()

    # Pure settle -- no camera is read, so skip the render pass on all 40 steps.
    # (gm.HEADLESS only removes the window; step() still renders without this context.)
    with og.sim.render_on_step(False):
        for _ in range(JOINT_SETTLE_STEPS):
            og.sim.step()
        for env in pending:
            for j in env.pending_joint_reset.cabinet.joints.values():
                j: JointPrim
                j.keep_still()
        for _ in range(JOINT_HOLD_STEPS):
            og.sim.step()

    # Clear last: the loop above reads pending_joint_reset.cabinet, and a member whose plan is still
    # set after this returns is the "recorded but never run" case RealmVectorEnvironment asserts on.
    for env in pending:
        env.pending_joint_reset = None


class RealmEnvironmentBase:
    # Per-instance in practice -- RealmEnvironmentDynamic.__init__ sets it before og.Environment is
    # built -- but declared here because RealmEnvironmentBase.__init__ itself calls reset_joints(),
    # which has to read it.
    in_vec_env = False
    # A JointResetPlan recorded by reset_joints() and drained by RealmVectorEnvironment. Always None
    # outside a vector env, where reset_joints() runs the plan inline before returning.
    pending_joint_reset = None

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
        # Build-time and STAYS build-time, unlike the two above. Not an oversight -- see the last
        # paragraph of capture_mo_reference() for why re-taking this one would be a regression.
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.task_type = task_type
        self.robot = robot
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}

        self.was_lifted = False
        if task_type in TASK_PROGRESS_RUBRICS:
            # deepcopy is load-bearing: TASK_PROGRESS_RUBRICS is built ONCE at module import, and
            # recompute_task_progression MUTATES this dict (`self.task_progression[stage] = True`).
            # Assigning the module-level object gave every environment in the process the SAME
            # progression state. Harmless with one env per process, catastrophic in a vector env:
            # member A grasping set GRASP=True for all members, and because
            # recompute_task_progression short-circuits on `is_completed_flag or checker(obs)` an
            # already-True stage is never re-checked per member. Progression became an OR across
            # members and stuck there, so every member reported the same timestamps, the same
            # stage, and SR=1 whenever ANY member succeeded. It also made the 15-step terminal
            # countdown start simultaneously for all of them, which looked like the members
            # naturally converging. Measured 2026-08-13: it inflated a 25-rollout vectorized
            # pi0.5 eval to SR 0.960 (an upper bound over waves of 4), reported by Martin from
            # the videos -- rollouts scored SUCCESS with the block never grasped.
            self.task_progression = copy.deepcopy(TASK_PROGRESS_RUBRICS[task_type])
        else:
            self.task_progression = None

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

    def capture_mo_reference(self):
        """Re-take mo_pos_orig / mo_rot_orig from whatever main_objects[0] points at RIGHT NOW.

        These two are the START-OF-ROLLOUT reference the progression stages are judged against:
        check_lift_and_distance_condition() (LIFT_SLIGHT, LIFT_LARGE, PUSH) measures both the lift
        `pos.z - mo_pos_orig.z` and the travel `||pos - mo_pos_orig||`, and check_rotated() (ROTATED)
        measures against mo_rot_orig. __init__ seeds them from the task config, which is only right
        while main_objects[0] is still the object the config declared -- and several perturbations
        change that DURING reset(), after the seed:

            SB-NOUN   pops a random distractor and swaps it into main_objects[0] (sb_noun.py)
            VSB-NOBJ  replaces main_objects[0] with a freshly sampled object (vsb_nobj.py)
            VB-MOBJ   replaces main_objects[0] with a rescaled copy (vb_mobj.py)

        Without this the reference described one object while the checks read another. Measured
        2026-08-13, SB-NOUN on task 0, 6 resets (scripts/clara/interactive/t11_mopos_ref.py): right
        after reset() the reference sat 0.111-0.465 m (mean 0.285 m) from the object being scored,
        and LIFT_SLIGHT answered True AT REST on 3 of 6 resets -- progression that never happened.

        Call this ONLY at the end of a reset, never while stepping. It records where the object
        STARTED; a reference that followed the object would drive both terms to zero and make every
        lift/distance check permanently False, silently deleting the stage instead of fixing it.
        t11_mopos_ref.py's [FROZEN] section tests that direction explicitly.

        Kept as one method rather than a line in each perturbation so a future perturbation that
        swaps the object cannot forget it: every reset path ends here. The call sites are
        RealmEnvironmentDynamic.apply_perturbations() (the phase that does the swapping, and the
        tail of reset()) plus both warmups. A vector env needs its own call in
        RealmVectorEnvironment.reset(), because apply_perturbations() runs there before the shared
        play -- exactly as it already needs its own settle and its own deferred post-play drain.

        mo_bbox_orig is DELIBERATELY not re-taken here, though it is seeded on the line right after
        these two and looks like it has the same staleness shape. It does not, for three separate
        reasons, any one of which is enough:

          - It is an ANCHOR, not a description of the current object. Its only reader is VB-MOBJ,
            which computes `mo_bbox_orig * U(0.5,1.5)^3` EVERY reset and then rescales
            (PrimitiveObject) or removes-and-re-adds (DatasetObject) main_objects[0] at that size.
            Re-taking it would make each reset scale relative to the previous reset's already-scaled
            object -- a multiplicative random walk that ends up pinned against vb_mobj.py's
            [0.02, 0.175] m clip. Anchoring on the task config is what keeps VB-MOBJ's draw
            independent per reset, which is also what the harness's `size` observable assumes.
          - The staleness itself is unreachable. The perturbations that re-point main_objects[0] at
            a DIFFERENT object are SB-NOUN and VSB-NOBJ, and REALM runs exactly one perturbation per
            process (eval.py builds `[SUPPORTED_PERTURBATIONS[perturbation_id]]`, vector_eval.py
            `[perturbation]`), so neither can ever precede VB-MOBJ. VB-MOBJ's own swap is
            same-category/same-model, and is the swap the anchor exists to survive.
          - There is nothing sound to capture. For a PrimitiveObject vb_mobj.py assigns this value
            to `mo.scale`, which is a scale FACTOR; it only coincides with an extent because
            primitives are authored at scale 1. get_position_orientation() has no analogue for it.

        If perturbations are ever COMPOSED -- the same caveat v_view.py records -- SB-NOUN followed
        by VB-MOBJ would leave mo_bbox_orig describing an object that is no longer the target. The
        fix then belongs in the perturbation that does the swapping (re-seed from the new object's
        CONFIG), not here: this method reads the live object, which is exactly what mo_bbox_orig
        must not do.
        """
        # Stored as OmniGibson hands them back (torch, cloned -- RigidDynamicPrim.get_position_
        # orientation defaults to clone=True, so this is a snapshot and not a view onto the physics
        # buffer). Deliberately NOT converted to numpy: this is byte-for-byte what warmup() has
        # always stored, so no historical number moves.
        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()

    def reset_joints(self, target_drawer_loc: str = "top"):
        """Put this member's cabinet back to the task's starting drawer state.

        In a vector env this only RECORDS the plan and returns; RealmVectorEnvironment drains it and
        runs one shared step loop for every member. Same reason and same shape as
        perturbations/_helpers.settle(): og.sim.step() advances EVERY scene, and the loop this
        method drives is ~55 of them -- nearly twice the settle loop that was already hoisted out
        for exactly this reason -- so per member it costs 55*N global steps per reset. See
        run_joint_resets().

        Recording rather than no-oping is deliberate, for the reason settle() raises a flag: a
        member that never asked for a joint reset must not silently acquire one, and a plan that is
        never drained must fail loudly -- RealmVectorEnvironment.reset() asserts the queue is empty
        -- rather than quietly leave a drawer in the wrong start state and score the rollout
        against it.

        Only open_drawer/close_drawer reach any of this; every other task takes the early return
        below, which is what it always did and costs nothing either way. Both drawer tasks load as
        of 2026-08-14, so the batching is measured rather than assumed -- see run_joint_resets().
        """
        if self.task_type not in ("open_drawer", "close_drawer"):
            self.mo_joint = None
            return

        self.pending_joint_reset = self._prepare_joint_reset(target_drawer_loc)
        if not self.in_vec_env:
            run_joint_resets([self])

    def _prepare_joint_reset(self, target_drawer_loc: str) -> JointResetPlan:
        """The half of reset_joints() that touches only THIS member: pick the joint, set its drive.

        Deliberately contains no og.sim.step(): that is what lets a vector env run this for every
        member up front and then step once for all of them. It stays at the reset_joints() call site
        rather than being deferred with the stepping, so a caller that reads self.mo_joint straight
        afterwards still sees the joint this reset selected.
        """
        cabinet = self.main_objects[0]
        init_state_open = self.task_type == "close_drawer"
        self.mo_joint = get_target_drawer_joint(cabinet, target_drawer_loc=target_drawer_loc)

        self.mo_joint._articulation_view.set_max_efforts(torch.tensor([[1.0e8]], dtype=torch.float32), joint_indices=self.mo_joint.dof_indices)
        self.mo_joint._articulation_view.set_gains(kps=torch.tensor([[0.0]]), joint_indices=self.mo_joint.dof_indices)
        self.mo_joint._articulation_view.set_gains(kds=torch.tensor([[1000.0]]), joint_indices=self.mo_joint.dof_indices)

        openable_joints = get_openable_joints(cabinet)
        reset_states = [-1 for _ in openable_joints]
        target_joint_ind = openable_joints.index(self.mo_joint)
        reset_states[target_joint_ind] = INIT_OPENNESS_FRACTION if init_state_open else -1
        return JointResetPlan(cabinet=cabinet, joints=openable_joints, reset_states=reset_states)

    def _record_joint_openness(self):
        """Capture the openness reference the joint progression stages are measured against.

        Called by run_joint_resets() between the driving loop and the free-run loop, which is where
        the pre-batching straight-line version read it.
        """
        self.joint_range = self.mo_joint.upper_limit - self.mo_joint.lower_limit
        self.init_openness_fraction = (self.mo_joint.get_state()[0][
                                           0] - self.mo_joint.lower_limit) / self.joint_range

    # ============================== [STATUS] ==============================
    def get_ee_pose(self):
        ee_link_name = self.robot.eef_link_names[self.robot.default_arm]
        ee_link = self.robot.links[ee_link_name]
        return ee_link.get_position_orientation()

    def check_collisions(self):
        self_collision = False
        env_collision = False

        # Cache adjacent links to ignore self-collisions between connected bodies
        if not hasattr(self, "_robot_adjacent_links"):
            self._robot_adjacent_links = set()
            if hasattr(self.robot, "joints"):
                for joint in self.robot.joints.values():
                    b0 = joint.body0
                    b1 = joint.body1
                    if b0 and b1:
                        self._robot_adjacent_links.add(frozenset((b0, b1)))

        robot_links = list(self.robot.links.values())
        robot_link_paths = set(l.prim_path for l in robot_links)
        robot_prim_path = self.robot.prim_path

        # Objects to ignore for environment collision (manipulation targets)
        # We use prefixes to catch links and geoms belonging to these objects
        ignore_obj_roots = [obj.prim_path for obj in self.main_objects + self.target_objects]

        # Skip the root link (usually touching mount/floor). OG 3.9.1 removed RigidPrim.contact_list(),
        # so contacts (and their impulses) are read from the contact matrix instead -- one batched
        # query for all links rather than a per-link call. See realm/environments/contact_utils.py.
        queried_links = [link for link in robot_links if link.name != self.robot.root_link_name]
        contacts_by_link = get_impulse_contacts(self.robot.scene.idx, queried_links)

        for link in queried_links:
            for other_path in contacts_by_link.get(link.prim_path, ()):
                # Check if other_path belongs to the robot
                is_robot = other_path in robot_link_paths or other_path.startswith(robot_prim_path)

                if is_robot:
                    # Ignore collisions between adjacent links
                    # Only applicable if we have exact link paths; otherwise assume it's a valid self-collision
                    if other_path in robot_link_paths:
                        if frozenset((link.prim_path, other_path)) in self._robot_adjacent_links:
                            continue
                    self_collision = True
                else:
                    # Check if it's an allowed environment contact (belongs to main/target objects)
                    is_ignored = any(other_path.startswith(root) for root in ignore_obj_roots)
                    if not is_ignored:
                        env_collision = True

            if self_collision and env_collision:
                break

        return self_collision, env_collision

    # ============================== [SUCCESS METRICS] ==============================
    def is_grasping(self, obs, candidate_obj):
        finger_joints = obs[self.robot.name]['proprio'][7:9].cpu().numpy()
        # This test used a bare literal 0.45 compared against the raw finger joint value, with no
        # units and no normalisation by joint range -- so what it meant depended entirely on the
        # asset. On droid.usd the finger joints are PRISMATIC in metres over [0, 0.05], so 0.45 is
        # 9x the entire travel and the test is VACUOUSLY TRUE. On the robolab 2F-85 the same proprio
        # indices are REVOLUTE in radians over [0, 0.7854], where 0.45 lands mid-range and the test
        # becomes "less than ~57% closed" -- which a real grasp violates. Measured 2026-08-11
        # (job 189066): with both pads on the block and the block lifted, finger_joint sits at
        # 0.507-0.528, so this rejected 78/78 genuine grasp steps and the asset could never score a
        # GRASP. Because recompute_task_progression breaks at the first unmet stage, that also froze
        # LIFT/MOVE/PLACE on rollouts that visibly completed the task.
        #
        # Scale the threshold by the robot's own open->closed range instead. The 9x factor is chosen
        # to reproduce 0.45 EXACTLY for droid.usd (9 * 0.05), so every historical result is
        # bit-identical; for robolab it becomes 7.07 rad and the test is vacuous there too, matching
        # the behaviour the stock asset has always had.
        #
        # NOTE: 0.45 is very likely a typo for 0.045, i.e. "the fingers stopped short of full
        # closure, so an object is between them" (90% of droid.usd's travel), which is a meaningful
        # test rather than a no-op. Deliberately NOT adopted here: it would make the guard bite on
        # droid.usd for the first time and could move every historical REALM number. Decide that
        # separately, with a measurement.
        _prof = get_robot_obs_profile(self.robot.name)
        _open_q, _closed_q = _prof["gripper_open_qpos"], _prof["gripper_closed_qpos"]
        _thresh = _open_q + 9.0 * (_closed_q - _open_q)
        is_either_finger_closing = (_thresh - finger_joints[0] > 1e-3 or _thresh - finger_joints[1] > 1e-3)
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

        # TODO: make the distance computation bbox dependent rather than centre-to-centre

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

    # NOTE: switched to checking Z axis rotation only, possible it is still bad but seems to be working well now
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

